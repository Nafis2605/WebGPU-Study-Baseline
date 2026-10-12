// TF.js backend selection and configuration (port of the sibling TF.js inference study's file;
// the converter is replaced by tfjs-layers, which shares the single deduplicated tfjs-core).
//
// Every backend package is imported so each registers itself; the requested one is then chosen
// explicitly with tf.setBackend(). An implicit tf.ready() would silently pick the best
// backend that initialises, so it is never relied on: setBackend() must resolve true and
// tf.getBackend() must equal the requested name, otherwise the run fails (no fallback).

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-webgpu";
import { setWasmPaths, setThreadsCount, getThreadsCount } from "@tensorflow/tfjs-backend-wasm";
import * as tfl from "@tensorflow/tfjs-layers";

export { tf, tfl };

const WASM_DIR = "/tfjs-wasm/";

// CPU forwarding runs kernels whose inputs are small and still CPU-resident (shape/scalar
// arithmetic) on TF.js's JS CPU kernels instead of the GPU backend. It is TF.js's default and the
// study's primary configs keep it (forwarded kernels are counted per run by backend-hooks.js);
// strict configs (cfg.strict) disable it so every kernel executes on the requested backend.
const STRICT_FLAGS = {
  webgl: { WEBGL_CPU_FORWARD: false },
  webgpu: { WEBGPU_CPU_FORWARD: false },
};

/**
 * Apply a config: env flags (before backend init), WASM paths/threads, then setBackend.
 * cfg = {backend, threads?, strict?, flags?: {NAME: value}}
 * Returns a record of what was actually selected; throws on any mismatch.
 */
export async function applyConfig(cfg) {
  const backend = cfg.backend;
  const flags = { ...(cfg.strict ? STRICT_FLAGS[backend] ?? {} : {}), ...(cfg.flags ?? {}) };
  for (const [name, value] of Object.entries(flags)) tf.env().set(name, value);

  if (backend === "wasm") {
    setWasmPaths({
      "tfjs-backend-wasm.wasm": `${WASM_DIR}tfjs-backend-wasm.wasm`,
      "tfjs-backend-wasm-simd.wasm": `${WASM_DIR}tfjs-backend-wasm-simd.wasm`,
      "tfjs-backend-wasm-threaded-simd.wasm": `${WASM_DIR}tfjs-backend-wasm-threaded-simd.wasm`,
    });
    if (cfg.threads) setThreadsCount(cfg.threads);
  }

  const ok = await tf.setBackend(backend);
  if (!ok) throw new Error(`backend_unavailable: tf.setBackend('${backend}') resolved false`);
  await tf.ready();
  const active = tf.getBackend();
  if (active !== backend) throw new Error(`backend_mismatch: requested ${backend}, active ${active}`);

  const info = {
    backend: active,
    tfjs_version: tf.version_core,
    tfjs_layers_version: tfl.version_layers,
    // Numerics fixed at backend init: Adam's default epsilon and the layers' clip epsilon.
    backend_epsilon: tf.backend().epsilon(),
    float_precision: tf.backend().floatPrecision(),
    cross_origin_isolated: globalThis.crossOriginIsolated === true,
    flags_set: flags,
    flags_effective: effectiveFlags(backend),
  };
  if (backend === "wasm") {
    info.wasm_threads = getThreadsCount();
    info.wasm_multithread = tf.env().getBool("WASM_HAS_MULTITHREAD_SUPPORT");
    info.wasm_simd = tf.env().getBool("WASM_HAS_SIMD_SUPPORT");
    if (cfg.threads && cfg.threads > 1 && (!info.wasm_multithread || info.wasm_threads !== cfg.threads)) {
      throw new Error(`wasm_threads_mismatch: requested ${cfg.threads}, got ${info.wasm_threads} (multithread=${info.wasm_multithread})`);
    }
  }
  if (backend === "webgpu") {
    const b = tf.backend();
    info.adapter = b.adapterInfo ? { ...b.adapterInfo } : null;
    info.device_features = [...b.device.features];
    info.timestamp_query = b.device.features.has("timestamp-query");
  }
  if (backend === "webgl") {
    info.webgl_version = tf.env().getNumber("WEBGL_VERSION");
    info.webgl_timer_reliable = tf.env().getBool("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE");
    info.webgl_render_float32 = tf.env().getBool("WEBGL_RENDER_FLOAT32_ENABLED");
    const gl = tf.backend().gpgpu.gl;
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    info.webgl_renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  }
  return info;
}

const REPORTED_FLAGS = {
  webgl: ["WEBGL_PACK", "WEBGL_CPU_FORWARD", "CPU_HANDOFF_SIZE_THRESHOLD", "WEBGL_USE_SHAPES_UNIFORMS",
    "WEBGL_FORCE_F16_TEXTURES", "WEBGL_FLUSH_THRESHOLD", "WEBGL_DELETE_TEXTURE_THRESHOLD", "ENGINE_COMPILE_ONLY"],
  webgpu: ["WEBGPU_CPU_FORWARD", "WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD", "WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE",
    "WEBGPU_ENGINE_COMPILE_ONLY", "WEBGPU_MATMUL_PROGRAM_TYPE", "WEBGPU_USE_PROFILE_TOOL"],
  wasm: [],
  cpu: [],
};

function effectiveFlags(backend) {
  const out = {};
  for (const name of REPORTED_FLAGS[backend] ?? []) {
    try { out[name] = tf.env().get(name); } catch { out[name] = null; }
  }
  return out;
}

/**
 * Wait until all GPU work queued so far has finished, without reading any tensor back.
 * WebGPU: TF.js batches dispatches into a pending command encoder, so it must be submitted
 * first (otherwise onSubmittedWorkDone resolves before the work even starts).
 * WebGL: a fence after the last draw. WASM/CPU: execution is already synchronous.
 */
export async function gpuFence() {
  const backend = tf.getBackend();
  const b = tf.backend();
  if (backend === "webgpu") {
    b.endComputePassEncoder?.();
    if (b.commandEncoder) b.submitQueue();
    await b.device.queue.onSubmittedWorkDone();
  } else if (backend === "webgl") {
    await b.gpgpu.createAndWaitForFence();
  }
}
