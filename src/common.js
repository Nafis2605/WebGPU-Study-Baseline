// Helpers shared by the inference and training modes (ported from the sibling study's bench.js).
import { tf } from "./tfjs-setup.js";
import { resetCounters, readCounters } from "./instrument.js";
import { resetBackendHooks, readBackendHooks } from "./backend-hooks.js";
import { mulberry32, fnv1a32 } from "./prng.js";

export { mulberry32, fnv1a32 };

export function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Page memory (JS heap + WASM memory + DOM) from performance.measureUserAgentSpecificMemory().
 * Needs cross-origin isolation; Chrome runs it at the next GC unless launched with
 * --enable-blink-features=ForceEagerMeasureMemory.
 */
export async function pageMemory() {
  if (!self.crossOriginIsolated) {
    return { bytes: null, reason: "measureUserAgentSpecificMemory unavailable (needs cross-origin isolation)" };
  }
  if (!performance.measureUserAgentSpecificMemory) {
    return { bytes: null, reason: "measureUserAgentSpecificMemory not implemented by this browser" };
  }
  try {
    const m = await withTimeout(performance.measureUserAgentSpecificMemory(), 10000, "memory measurement");
    return { bytes: m.bytes };
  } catch (e) {
    return { bytes: null, reason: String(e?.message ?? e) };
  }
}

/** Seeded Fisher-Yates shuffle (sibling bench.js). */
function shuffled(values, rand) {
  const a = [...values];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Shape order for the reuse study (sibling bench.js): `rounds` seeded permutations of the sweep values,
 * arranged so consecutive rounds never join on the same value.
 */
export function shapeSequence(values, rounds, seed) {
  let s = seed >>> 0;
  const rand = () => ((s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0) / 4294967296);
  const seq = [];
  for (let r = 0; r < rounds; r++) {
    let perm = shuffled(values, rand);
    if (seq.length && perm[0] === seq[seq.length - 1] && perm.length > 1) perm = [...perm.slice(1), perm[0]];
    seq.push(...perm);
  }
  return seq;
}

// Per-run counter deltas kept in instrumented launches (what a run / step compiles and uploads).
export const RUN_COUNTERS = [
  "webgl_draws", "webgl_programs_linked", "webgl_tex_uploads", "webgl_tex_upload_bytes",
  "webgl_read_pixels_calls", "webgl_read_pixels_bytes", "webgl_get_buffer_sub_data_calls", "webgl_get_buffer_sub_data_bytes",
  "webgpu_submits", "webgpu_dispatches", "webgpu_compute_passes", "webgpu_pipelines_created", "webgpu_pipelines_created_async",
  "webgpu_write_buffer_calls", "webgpu_write_buffer_bytes", "webgpu_copy_b2b_bytes",
  "webgpu_buffers_created", "webgpu_buffer_bytes_created", "webgpu_mapped_at_creation_bytes",
  "webgpu_bind_groups_created", "webgpu_map_read_calls", "webgpu_map_read_bytes",
];

/**
 * Instrumented launches only: wrap a runner so every run record carries the counter deltas and
 * backend-hook totals of that run (per training phase too). Timings from such launches are never
 * used as latency data.
 */
export function instrumented(runner) {
  return {
    ...runner,
    async run() {
      resetCounters();
      resetBackendHooks();
      const r = await runner.run();
      const c = readCounters();
      const h = readBackendHooks();
      return {
        ...r,
        counters: Object.fromEntries(RUN_COUNTERS.map(k => [k, c[k]])),
        hooks: {
          forwarded_kernels: h.forwarded_kernels, sync_reads: h.sync_reads, sync_read_ms: h.sync_read_ms,
          async_reads: h.async_reads, kernels_run: h.kernels_run, kernel_cpu_ms: h.kernel_cpu_ms,
          math_random_calls: h.math_random_calls,
          ...Object.fromEntries(["forward", "backward", "optimizer", "other"].flatMap(p => [
            [`kernels_${p}`, h.kernels_run_by_phase[p] ?? 0],
            [`kernel_cpu_ms_${p}`, h.kernel_cpu_ms_by_phase[p] ?? 0],
            [`forwarded_${p}`, h.forwarded_by_phase[p] ?? 0],
          ])),
        },
      };
    },
  };
}

/**
 * Runs appended to `records` (sibling runner.js runnerRuns, plus a minimum duration).
 * cap = {maxMs, min, minMs}: stop once `count` runs are done AND minMs has elapsed (so the GPU energy
 * window of fast models reaches >= 1 s; at most MAX_EXTEND x count runs), or once maxMs has elapsed with
 * at least `min` runs (slow pairs: fewer samples, never estimated ones).
 */
const MAX_EXTEND = 50;
export async function runnerRuns(runner, phase, count, records, extra = {}, cap = null) {
  const t0 = performance.now();
  // Floor clock: from the start, or (cap.floorAfterFirst, warm-up) from the end of the cold first run.
  let tFloor = t0;
  const maxExtend = cap?.maxExtend ?? MAX_EXTEND;
  let i = 0;
  for (;; i++) {
    const now = performance.now();
    const el = now - t0;
    if (i >= count && (!cap?.minMs || now - tFloor >= cap.minMs || i >= count * maxExtend)) break;
    if (cap && i >= cap.min && el > cap.maxMs) break;
    records.push({ phase, i, ...extra, ...(await runner.run()) });
    if (i === 0 && cap?.floorAfterFirst) tFloor = performance.now();
  }
  return { runs: i, capped: i < count };
}

/**
 * Warm-up rule of every timed mode (set per launch by bench.js): at least the configured count and at
 * least cfg.warmupMinS seconds after the first (cold) record, so fast models warm up for more than a
 * few ms (preflight: 10 records of a 0.3-3 ms inference left the first 0.2-0.3 s of the measured window
 * 3-10 % slow); bounded by cfg.warmupMaxS once >= 3 records exist (slow pairs, e.g. seconds per step on
 * WASM). A capped warm-up is flagged in result.capped; sufficiency is checked by analysis/warmup.py.
 */
let WARMUP_CAP = null;
export function setWarmupCap(c) { WARMUP_CAP = c; }
export const warmupCap = () => WARMUP_CAP;
/** Unrecorded warm-up with the same rule (load / responsiveness settle the runner before their window). */
export async function warmUp(runner, count) { await runnerRuns(runner, "warmup", count, [], {}, WARMUP_CAP); }

/** The measurement cap from a launch config: {maxMs, min, minMs} or null. */
export function capOf(maxS, min, count, minS = null) {
  if (!maxS && !minS) return null;
  return { maxMs: (maxS ?? Infinity) * 1000, min: Math.min(min ?? 30, count), minMs: (minS ?? 0) * 1000 };
}

/** Raw little-endian bytes of a typed array as base64 (sibling encodeTensor). */
export function encodeTensor(name, dtype, dims, data) {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return { name, type: dtype, dims, b64: btoa(binary) };
}

export { tf };
