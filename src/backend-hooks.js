// TF.js backend-level instrumentation (no ONNX equivalent: ORT exposes node placement in a log,
// TF.js does not). Wraps methods on the active backend *instance*:
//
//   shouldExecuteOnCPU(inputs) -> true means the kernel about to run is executed by TF.js's JS
//     CPU kernels instead of the GPU backend ("CPU forwarding"). Counted per kernel name.
//   readSync(dataId)           -> a blocking GPU->CPU read. Called by tensor.dataSync() and by
//     kernels that need values on the CPU (GatherV2, StridedSlice, TopK, Tile, ...). Each call is
//     a CPU-GPU execution boundary.
//   read(dataId)               -> an asynchronous GPU->CPU read (tensor.data()).
//   engine.runKernel           -> CPU time of each outermost kernel call (performance.now around it):
//     on GPU backends the JS-side cost of preparing/encoding the kernel (plus any forwarded CPU
//     kernel or blocking readSync inside it); on WASM the kernel's execution itself. TF.js's
//     counterpart of ORT's per-node CPU time, measured directly.
//
// Events are attributed to the outermost kernel executing at the time, found by wrapping
// engine.runKernel (the entry point every op uses). Sub-programs a backend runs inside a kernel
// are attributed to that kernel.
//
// Training study additions:
//   phase      each outermost kernel is also attributed to a training phase, read from the engine's
//              scope stack when the kernel starts: engine.gradients() runs the forward pass in
//              tidy('forward') and backprop in tidy('backward'); the train runner applies the
//              optimizer in tidy('optimizer'). Anything else (inference, input creation) is "other".
//              Nothing is fenced or timed differently: the tag is a lookup on a JS array.
//   Math.random  calls counted while the hooks are installed (the suite uses no randomness; a
//              non-zero count would mean an unseeded RNG inside the measured work).

import { tf } from "./tfjs-setup.js";

const restores = [];
let h = null;
let klog = null;   // [kernelName, phase] of every engine.runKernel call (profile join), or null

function empty() {
  return {
    forwarded_kernels: 0,
    forwarded_by_kernel: {},
    sync_reads: 0,
    sync_read_ms: 0,
    sync_reads_by_kernel: {},
    async_reads: 0,
    kernels_run: 0,
    kernel_cpu_ms: 0,
    kernel_cpu_ms_by_kernel: {},
    kernels_run_by_phase: {},
    kernel_cpu_ms_by_phase: {},
    forwarded_by_phase: {},
    sync_reads_by_phase: {},
    math_random_calls: 0,
  };
}

const PHASES = ["optimizer", "backward", "forward"];

/** Innermost training-phase scope on the engine's scope stack, or "other". */
function phaseOf(engine) {
  const stack = engine.state.scopeStack;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (PHASES.includes(stack[i].name)) return stack[i].name;
  }
  return "other";
}

function bump(obj, key, by = 1) {
  obj[key] = (obj[key] ?? 0) + by;
}

function wrapMethod(obj, name, make) {
  if (!obj || typeof obj[name] !== "function") return false;
  const had = Object.prototype.hasOwnProperty.call(obj, name);
  const original = obj[name];
  obj[name] = make(original);
  restores.push(() => { if (had) obj[name] = original; else delete obj[name]; });
  return true;
}

export function installBackendHooks() {
  if (restores.length) throw new Error("backend hooks already installed");
  h = empty();
  const engine = tf.engine();
  const backend = tf.backend();
  let current = null;   // kernel name currently executing (outermost)
  let phase = "other";  // training phase of that kernel

  // engine.runKernel(kernelName, inputs, attrs) is the public entry for every registered kernel.
  wrapMethod(engine, "runKernel", original => function (kernelName, ...rest) {
    if (klog) {
      // Logged on completion (post-order), the order tf.profile records kernels in: a kernel that runs
      // others inside it (e.g. the gradient tape's Identity clones) is listed after them.
      const ph = phaseOf(engine);
      const log = klog;
      try {
        return runOuterOrInner.call(this, original, kernelName, rest);
      } finally {
        log.push([kernelName, ph]);
      }
    }
    return runOuterOrInner.call(this, original, kernelName, rest);
  });

  function runOuterOrInner(original, kernelName, rest) {
    const outer = current === null;
    if (!outer) return original.call(this, kernelName, ...rest);
    current = kernelName;
    phase = phaseOf(engine);
    h.kernels_run++;
    bump(h.kernels_run_by_phase, phase);
    const t0 = performance.now();
    try {
      return original.call(this, kernelName, ...rest);
    } finally {
      const dt = performance.now() - t0;
      h.kernel_cpu_ms += dt;
      h.kernel_cpu_ms_by_kernel[kernelName] = (h.kernel_cpu_ms_by_kernel[kernelName] ?? 0) + dt;
      bump(h.kernel_cpu_ms_by_phase, phase, dt);
      current = null;
      phase = "other";
    }
  }
  wrapMethod(backend, "shouldExecuteOnCPU", original => function (...args) {
    const r = original.apply(this, args);
    if (r) {
      h.forwarded_kernels++;
      const k = current ?? "(unknown)";
      h.forwarded_by_kernel[k] = (h.forwarded_by_kernel[k] ?? 0) + 1;
      bump(h.forwarded_by_phase, current ? phase : "other");
    }
    return r;
  });
  wrapMethod(backend, "readSync", original => function (...args) {
    const t0 = performance.now();
    try {
      return original.apply(this, args);
    } finally {
      h.sync_reads++;
      h.sync_read_ms += performance.now() - t0;
      const k = current ?? "(user)";
      h.sync_reads_by_kernel[k] = (h.sync_reads_by_kernel[k] ?? 0) + 1;
      bump(h.sync_reads_by_phase, current ? phase : "other");
    }
  });
  wrapMethod(backend, "read", original => function (...args) {
    h.async_reads++;
    return original.apply(this, args);
  });
  wrapMethod(Math, "random", original => function () {
    if (h) h.math_random_calls++;
    return original.call(Math);
  });
}

export function uninstallBackendHooks() {
  while (restores.length) restores.pop()();
  h = null;
}

export function resetBackendHooks() {
  if (h) h = empty();
}

/** Log every engine.runKernel call (all depths) with its phase, until readKernelLog(). */
export function startKernelLog() {
  klog = [];
}

export function readKernelLog() {
  const out = klog ?? [];
  klog = null;
  return out;
}

export function readBackendHooks() {
  return h ? JSON.parse(JSON.stringify(h)) : null;
}
