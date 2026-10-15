// Benchmark core for the custom-model study (structure ported from the sibling TF.js inference
// study's bench.js). Every value reported here is a direct measurement; anything that can't be
// measured is left out or reported as null with a reason, never estimated.
//
// runExperiment(cfg) phases, each timed:
//   backend_init     applyConfig: flags, WASM paths/threads, tf.setBackend (must resolve true and
//                    match; no fallback)
//   data_load        fetch + sha256-verify the CIFAR-10 subsets (data_fetch_ms, data_verify_ms)
//   fetch            fetch the weight file (fetch_ms, model_bytes) + sha256 check (weights_verify_ms)
//   session_create   build the model from the spec (build_ms) + write the weights (weights_assign_ms);
//                    session_create_ms = build_ms + weights_assign_ms (the sibling's graph build +
//                    weight tensor creation; GPU upload is lazy, so it lands in the first run/step)
//   precompile       optional TF.js compile-only pass: one inference (inference modes) or one full training
//                    step on a throwaway optimizer, after which the weights are rewritten (training modes;
//                    probe P0.7: later steps create no pipelines, losses unchanged)
//   run              the mode (src/infer-modes.js or src/train-modes.js)
//   verify           one extra run / step with API counters + backend hooks: the counter signature
//                    must match the requested backend, Math.random must not be called, the run must not
//                    leak tensors, and -strict configs must forward no kernel to the CPU.
import { tf, applyConfig } from "./tfjs-setup.js";
import { installCounters, uninstallCounters, resetCounters, readCounters, readGroups, verifyBackend } from "./instrument.js";
import { installBackendHooks, uninstallBackendHooks, resetBackendHooks, readBackendHooks } from "./backend-hooks.js";
import { buildModel, weightSlots } from "./models/build.js";
import { loadData } from "./data.js";
import { fetchWeights, assignWeights, verifyAssigned } from "./weights.js";
import { INFER_MODES } from "./infer-modes.js";
import { TRAIN_MODES } from "./train-modes.js";
import { makeLayersRunner } from "./layers-runner.js";
import { makeTrainRunner } from "./train-runner.js";
import { memorySnapshot } from "./memory.js";
import { withTimeout, pageMemory, RUN_COUNTERS, setWarmupCap } from "./common.js";

const SCHEMA_VERSION = 3;
const MODES = { ...INFER_MODES, ...TRAIN_MODES };
export const INFER_MODE_NAMES = Object.keys(INFER_MODES);
export const TRAIN_MODE_NAMES = Object.keys(TRAIN_MODES);

let log = msg => console.log(msg);
export function setLogger(fn) { log = fn; }

/** Browser / GPU environment facts for env.json. */
export async function probeEnv() {
  const env = {
    user_agent: navigator.userAgent,
    cross_origin_isolated: self.crossOriginIsolated,
    hardware_concurrency: navigator.hardwareConcurrency,
    time_origin_epoch_ms: performance.timeOrigin,
    timer_granularity_ms: timerGranularity(),
    tfjs_version: tf.version_core,
    webgpu_adapter: null,
    webgpu_limits: null,
    webgl: null,
  };
  if (navigator.gpu) {
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (a) {
      env.webgpu_adapter = {
        vendor: a.info.vendor, architecture: a.info.architecture,
        device: a.info.device, description: a.info.description,
        features: [...a.features].sort(),
      };
      env.webgpu_limits = { maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize };
    }
  }
  const gl = document.createElement("canvas").getContext("webgl2");
  if (gl) {
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    env.webgl = {
      version: gl.getParameter(gl.VERSION),
      renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      timer_query: !!gl.getExtension("EXT_disjoint_timer_query_webgl2"),
      max_texture_size: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    };
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
  return env;
}

/** Smallest positive step of performance.now() seen in a tight loop (5 µs when isolated). */
function timerGranularity() {
  let min = Infinity;
  let last = performance.now();
  for (let i = 0; i < 200000; i++) {
    const t = performance.now();
    if (t > last && t - last < min) min = t - last;
    last = t;
  }
  return min;
}

/** Label a failure from the raw TF.js error text. The raw message is always kept alongside. */
function classifyError(stage, message) {
  if (/data_mismatch/.test(message)) return "data_mismatch";
  if (/weights_mismatch/.test(message)) return "weights_mismatch";
  if (/not registered for backend/i.test(message)) return "kernel_unregistered_on_backend";
  if (/backend_unavailable|has not yet been initialized|not found in registry/i.test(message)) return "backend_unavailable";
  if (/wasm_threads_mismatch/.test(message)) return "wasm_threads_mismatch";
  if (/backend_mismatch/.test(message)) return "backend_mismatch";
  if (/precompile_unsupported/.test(message)) return "precompile_unsupported";
  if (/gradient function not found|Cannot compute gradient/i.test(message)) return "gradient_unsupported";
  if (/not yet supported|not supported|unsupported/i.test(message)) return "op_unsupported_on_backend";
  if (/timed out/i.test(message)) return "timeout";
  if (/out of memory|OOM|allocation failed|Array buffer allocation|Failed to allocate|Aborted\(OOM\)|memory access out of bounds/i.test(message)) return "out_of_memory";
  return `${stage}_error`;
}

/**
 * TF.js parallel pre-compilation (sibling bench.js): run once in compile-only mode (programs /
 * pipelines created, nothing dispatched), wait for compilation, leave compile-only mode.
 */
async function precompile(runner, backend) {
  const flag = backend === "webgpu" ? "WEBGPU_ENGINE_COMPILE_ONLY" : "ENGINE_COMPILE_ONLY";
  const b = tf.backend();
  tf.env().set(flag, true);
  try {
    await runner.compileOnce();
    await b.checkCompileCompletionAsync();
    if (backend === "webgl") b.getUniformLocations();
  } finally {
    tf.env().set(flag, false);
  }
}

/**
 * Run one (model, config, mode) experiment in this page.
 * cfg: {model: <spec entry>, backend, configLabel, threads?, strict?, flags?, precompile?, io?, fence?,
 *       readback?, mode, weights? ("init" | "trained"), seed, trial, instrumentRuns?, sessionTimeoutMs,
 *       inference: warmup, measure, maxMeasureS, minMeasure, inferBatch, shapeValues, batchValues, ...
 *       training:  trainWarmup, trainMeasure, trainMaxS, trainMinMeasure, trainBatch, syncPolicy, dataIo,
 *                  inferWarmup, inferMeasure, inferMaxS, inferMinMeasure, ... (see train-modes.js)}
 */
export async function runExperiment(cfg) {
  const { model: spec, backend } = cfg;
  cfg = { seed: 1, ...cfg };
  const mode = cfg.mode ?? "latency";
  const pillar = TRAIN_MODES[mode] ? "train" : "infer";
  const weightsKind = cfg.weights ?? (pillar === "train" ? "init" : "trained");
  const result = {
    schema: SCHEMA_VERSION,
    model: spec.name,
    family: spec.family,
    size: spec.size,
    params: spec.params,
    backend,
    mode,
    pillar,
    config_label: cfg.configLabel ?? backend,
    config: {
      warmup: cfg.warmup, measure: cfg.measure, seed: cfg.seed,
      warmup_min_s: cfg.warmupMinS ?? 0, warmup_max_s: cfg.warmupMaxS ?? null,
      max_measure_s: cfg.maxMeasureS ?? null, min_measure: cfg.maxMeasureS ? (cfg.minMeasure ?? 30) : null,
      infer_batch: cfg.inferBatch ?? 1,
      train_batch: cfg.trainBatch ?? spec.batch,
      train_warmup: cfg.trainWarmup ?? null, train_measure: cfg.trainMeasure ?? null,
      train_max_s: cfg.trainMaxS ?? null, train_min_measure: cfg.trainMinMeasure ?? null,
      sync_policy: cfg.syncPolicy ?? "loss", data_io: cfg.dataIo ?? "host",
      io: cfg.io ?? "cpu", fence: !!cfg.fence, readback: cfg.readback ?? "async",
      strict: !!cfg.strict,
      threads_requested: cfg.threads ?? null,
      flags: cfg.flags ?? {},
      exec: "predict",
      precompile: !!cfg.precompile,
      weights: weightsKind,
      // Instrumented launches carry per-run counters; their timings are not latency data.
      instrumented: !!cfg.instrumentRuns,
    },
    tfjs: null,
    status: "failed",
    failure: null,
    capped: [],
    model_bytes: null,
    backend_init_ms: null,
    data_fetch_ms: null,
    data_verify_ms: null,
    fetch_ms: null,
    weights_verify_ms: null,
    build_ms: null,
    weights_assign_ms: null,
    session_create_ms: null,
    precompile_ms: null,
    precompile_counters: null,
    weights_sha256: null,
    feeds: null,
    runs: [],
    verification: null,
  };

  let stage = "backend_init";
  let model = null;
  let runner = null;
  try {
    if (!MODES[mode]) throw new Error(`unknown mode ${mode}`);
    if (cfg.precompile && backend !== "webgl" && backend !== "webgpu") {
      throw new Error(`precompile is only defined for webgl/webgpu, not ${backend}`);
    }
    const b0 = performance.now();
    result.tfjs = await applyConfig(cfg);
    result.backend_init_ms = performance.now() - b0;

    stage = "data_load";
    const data = await loadData();
    result.data_fetch_ms = data.fetch_ms;
    result.data_verify_ms = data.verify_ms;
    result.data_manifest_seed = data.manifest.seed;

    // Instrumented launches count from before the weights are fetched, so weight uploads are included.
    const profileLike = mode === "profile" || mode === "train_profile";
    const instrumentedLaunch = cfg.instrumentRuns || profileLike;
    if (instrumentedLaunch) {
      installCounters();
      installBackendHooks();
    }
    if (profileLike) result.page_memory = { before_fetch: await pageMemory() };

    stage = "fetch";
    log(`[${spec.name}/${result.config_label}] ${mode}: fetching ${weightsKind} weights`);
    const w = await fetchWeights(spec.name, weightsKind);
    result.fetch_ms = w.fetch_ms;
    result.weights_verify_ms = w.verify_ms;
    result.model_bytes = w.entry.bytes;
    result.weights_sha256 = w.entry.sha256;

    stage = "session_create";
    const s0 = performance.now();
    model = await withTimeout(Promise.resolve().then(() => buildModel(spec)), cfg.sessionTimeoutMs ?? 600000, "model build");
    const slots = weightSlots(model, spec);
    const s1 = performance.now();
    assignWeights(slots, w);
    const s2 = performance.now();
    result.build_ms = s1 - s0;
    result.weights_assign_ms = s2 - s1;
    result.session_create_ms = s2 - s0;
    if (cfg.verifyWeights) {
      const v = await verifyAssigned(slots, w.entry);
      result.weights_readback = v;
      if (!v.ok) throw new Error(`weights_mismatch: read back differs for ${v.mismatches.slice(0, 5).join(", ")}`);
    }
    result.memory_after_load = memorySnapshot(instrumentedLaunch);
    if (profileLike) {
      result.profile_memory = { after_load: result.memory_after_load };
      result.page_memory.after_session_create = await pageMemory();
    }
    const ctx = { data, spec, slots };

    if (cfg.precompile) {
      stage = "precompile";
      if (cfg.instrumentRuns) { resetCounters(); resetBackendHooks(); }
      const p0 = performance.now();
      try {
        if (pillar === "train") {
          // Compile-only mode runs the optimizer's assigns on undefined data: weights are rewritten after.
          const pre = makeTrainRunner(model, slots, data, spec, cfg, cfg.trainBatch ?? spec.batch);
          await precompile({ compileOnce: async () => { const h = pre.executeSyncOnce(); h.out[0].dispose(); } }, backend);
          pre.dispose();
          assignWeights(slots, w);
        } else {
          await precompile(makeLayersRunner(model, data, cfg, cfg.inferBatch ?? 1), backend);
        }
      } catch (e) {
        throw new Error(`precompile_unsupported: ${e?.message ?? e}`);
      }
      result.precompile_ms = performance.now() - p0;
      if (cfg.instrumentRuns) {
        const c = readCounters();
        result.precompile_counters = Object.fromEntries(RUN_COUNTERS.map(k => [k, c[k]]));
      }
    }

    stage = "run";
    log(`[${spec.name}/${result.config_label}] mode ${mode}`);
    setWarmupCap({ maxMs: (cfg.warmupMaxS || Infinity) * 1000, min: 3, minMs: (cfg.warmupMinS ?? 0) * 1000,
                   floorAfterFirst: true, maxExtend: Infinity });
    runner = await MODES[mode](model, ctx, cfg, result);
    result.feeds = runner.spec;

    // Verification pass: one extra run / step with API counters and backend hooks, after (not
    // inside) the timed runs.
    stage = "verify";
    if (!instrumentedLaunch) {
      installCounters();
      installBackendHooks();
    }
    try {
      const tensorsBefore = tf.memory().numTensors;
      resetCounters();
      resetBackendHooks();
      await runner.run();
      const counters = readCounters();
      const groups = readGroups();
      result.verification = {
        counters,
        hooks: readBackendHooks(),
        dispatches_per_submit: groups.length ? groups.reduce((a, b) => a + b, 0) / groups.length : null,
        tensors_leaked: tf.memory().numTensors - tensorsBefore,
        ...verifyBackend(backend, counters),
      };
    } finally {
      uninstallBackendHooks();
      uninstallCounters();
    }
    const activeAfter = tf.getBackend();
    if (activeAfter !== backend) {
      result.verification.verified = false;
      result.verification.detail = `backend changed during the experiment: ${backend} -> ${activeAfter}`;
    }

    const v = result.verification;
    const forwarded = v.hooks?.forwarded_kernels ?? 0;
    const nonfinite = result.runs.some(r => r.nonfinite === true);
    if (!v.verified) result.status = "backend_not_verified";
    else if (cfg.strict && forwarded > 0) result.status = "strict_violation";
    else if ((v.hooks?.math_random_calls ?? 0) > 0) result.status = "rng_used";
    else if (v.tensors_leaked !== 0) result.status = "leak_detected";
    else if (nonfinite) result.status = "nonfinite_loss";
    else result.status = "ok";
    log(`[${spec.name}/${result.config_label}] ${result.status}: ${v.detail}; forwarded=${forwarded} ` +
        `sync_reads=${v.hooks?.sync_reads} leaked=${v.tensors_leaked} rng=${v.hooks?.math_random_calls}`);
  } catch (e) {
    const message = String(e?.message ?? e);
    result.failure = { stage, category: classifyError(stage, message), message };
    log(`[${spec.name}/${result.config_label}] FAILED at ${stage}: ${message}`);
  } finally {
    uninstallBackendHooks();
    uninstallCounters();
    try { runner?.dispose(); } catch { /* resident tensors / optimizer state */ }
    try { model?.dispose(); } catch { /* the model is discarded either way */ }
  }
  // What stays allocated once the model and all tensors are disposed (pool retention).
  try {
    if (result.tfjs) result.memory_after_dispose = memorySnapshot(false);
  } catch (e) {
    result.memory_after_dispose = { error: String(e?.message ?? e) };
  }
  return result;
}

export { pageMemory, withTimeout, precompile };
