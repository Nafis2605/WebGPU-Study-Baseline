// Groups 5-7: the instrumented pass (TF.js counterpart of the ONNX study's profile.js).
//
// runExperiment installs the API counters and backend hooks before the model is fetched, so weight
// uploads are counted. After warm-up, each phase runs cfg.measure inferences:
//
//   profile_counters  the normal run (tensors from CPU arrays -> execute -> data() on all outputs)
//                     with per-run: API counter deltas, dispatches per queue.submit, submission
//                     tracking (submit time, dispatches, in-flight depth, completion), backend
//                     hooks (forwarded kernels, sync/async reads, per-kernel CPU time) and
//                     logical GPU memory.
//   profile_kernels   the same inference inside tf.profile(): per-kernel time (WebGPU: compute-pass
//                     timestamps; WebGL: EXT_disjoint_timer_query; WASM: synchronous wall time),
//                     input/output shapes and bytes added. TF.js ends the compute pass and submits
//                     after every timed kernel on WebGPU, so wall time in this phase is not
//                     latency (the counters recorded here quantify the perturbation).
//                     tf.profile also lists op-scope wrapper records ("<op>__op", e.g.
//                     fusedConv2d__op around FusedConv2D): on WASM they repeat the wrapped kernel's
//                     wall time, on GPU backends they are 0. Sums and counts exclude them.
//   profile_time      tf.time(): kernelMs and wallMs (execute-only models; tf.time needs a
//                     synchronous function).
//   profile_draws     WebGL only: the harness's own per-draw timer queries (instrument.js), to
//                     cross-check TF.js's WebGL kernel times.
//
// Memory snapshots (group 7) are taken after the first run, after warm-up and after the phases;
// runExperiment adds after_load and after_dispose.
import { tf } from "./tfjs-setup.js";
import {
  resetCounters, readCounters, readGroups, readMemory, resetPeak,
  startSubmitTracking, readSubmitTracking, startDrawTimers, collectDrawTimers,
} from "./instrument.js";
import { resetBackendHooks, readBackendHooks, startKernelLog, readKernelLog } from "./backend-hooks.js";
import { memorySnapshot } from "./memory.js";

const HOOK_SCALARS = ["kernels_run", "forwarded_kernels", "sync_reads", "sync_read_ms", "async_reads", "kernel_cpu_ms"];

function hookScalars() {
  const h = readBackendHooks();
  return Object.fromEntries(HOOK_SCALARS.map(k => [k, h[k]]));
}

/** Per-run summary of the tracked submissions (times relative to the run start). */
function submitSummary(log, t0) {
  if (!log.length) return { tracked_submits: 0 };
  const done = log.map(r => r.t_done).filter(t => t !== null);
  // Observed completion interval of group i: from when it could start on the queue (its submit,
  // or the previous group's completion if later) to its own completion.
  const groupMs = [];
  for (let i = 0; i < log.length; i++) {
    if (log[i].t_done === null) continue;
    const prevDone = i > 0 && log[i - 1].t_done !== null ? log[i - 1].t_done : -Infinity;
    groupMs.push(log[i].t_done - Math.max(log[i].t_submit, prevDone));
  }
  const gaps = log.slice(1).map((r, i) => r.t_submit - log[i].t_submit);
  const median = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  return {
    tracked_submits: log.length,
    // Spread of the completion callbacks: when they arrive together, per-group durations can't be
    // resolved from them (they are then coalesced by the browser, not ordered by GPU progress).
    done_spread_ms: done.length ? Math.max(...done) - Math.min(...done) : null,
    unfinished_at_end: log.length - done.length,
    max_in_flight: Math.max(...log.map(r => r.in_flight_before)) + 1,
    first_submit_ms: log[0].t_submit - t0,
    last_done_ms: done.length ? Math.max(...done) - t0 : null,
    submit_gap_median_ms: median(gaps),
    group_completion_median_ms: median(groupMs),
    group_completion_max_ms: groupMs.length ? Math.max(...groupMs) : null,
  };
}

async function countersPhase(runner, count, runs, profile) {
  for (let i = 0; i < count; i++) {
    resetCounters();
    resetBackendHooks();
    startSubmitTracking();
    const t0 = performance.now();
    const rec = { phase: "profile_counters", i, ...(await runner.run()) };
    const log = await readSubmitTracking();
    rec.counters = readCounters();
    rec.groups = readGroups();
    rec.hooks = hookScalars();
    rec.submits = submitSummary(log, t0);
    rec.memory = readMemory();
    runs.push(rec);
    if (i === count - 1) {
      // Per-kernel CPU time of one run (outermost engine.runKernel; see backend-hooks.js).
      for (const [name, ms] of Object.entries(readBackendHooks().kernel_cpu_ms_by_kernel)) {
        profile.kernels.push({ source: "kernel_cpu", phase: "profile_counters", run: i, name, kernel_ms: ms });
      }
    }
  }
}

async function kernelsPhase(runner, count, runs, profile) {
  for (let i = 0; i < count; i++) {
    resetCounters();
    resetBackendHooks();
    const t0 = performance.now();
    let held = null;
    startKernelLog();
    const info = await tf.profile(async () => {
      held = await runner.executeOnce();
      return held.out;
    });
    const t1 = performance.now();
    // Training phase of each kernel: tf.profile and the hooks' log both list engine.runKernel calls in
    // execution order; the join is used only when the two name sequences are identical.
    const klogs = readKernelLog();
    const realIdx = info.kernels.map((k, idx) => (k.name.endsWith("__op") ? -1 : idx)).filter(i => i >= 0);
    const joinOk = klogs.length === realIdx.length && realIdx.every((idx, j) => info.kernels[idx].name === klogs[j][0]);
    const phaseOfIdx = new Map(joinOk ? realIdx.map((idx, j) => [idx, klogs[j][1]]) : []);
    const mis = joinOk ? -1 : realIdx.findIndex((idx, j) => !klogs[j] || info.kernels[idx].name !== klogs[j][0]);
    const joinMismatch = mis < 0 ? null : { at: mis, profile: realIdx.slice(Math.max(0, mis - 3), mis + 4).map(i => info.kernels[i].name),
      log: klogs.slice(Math.max(0, mis - 3), mis + 4).map(x => x[0]) };
    await runner.readAndRelease(held);
    const t2 = performance.now();
    const real = info.kernels.filter(k => !k.name.endsWith("__op"));
    const times = real.map(k => (typeof k.kernelTimeMs === "number" ? k.kernelTimeMs : null));
    const valid = times.filter(t => t !== null);
    runs.push({
      phase: "profile_kernels", i,
      t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t2 - t0, exec_ms: t1 - t0, read_ms: t2 - t1,
      gpu_kernels: real.length, wrapper_records: info.kernels.length - real.length,
      gpu_busy_ms: valid.length === times.length ? valid.reduce((a, b) => a + b, 0) : null,
      kernel_time_invalid: times.length - valid.length,
      zero_time_kernels: valid.filter(t => t === 0).length,
      prof_peak_bytes: info.peakBytes, prof_new_bytes: info.newBytes, prof_new_tensors: info.newTensors,
      phase_join_ok: joinOk, kernel_log_len: klogs.length, phase_join_mismatch: joinMismatch ? JSON.stringify(joinMismatch) : null,
      counters: readCounters(),
    });
    info.kernels.forEach((k, idx) => profile.kernels.push({
      source: "tf.profile", phase: "profile_kernels", run: i, idx, name: k.name,
      kernel_ms: typeof k.kernelTimeMs === "number" ? k.kernelTimeMs : null,
      kernel_time_error: typeof k.kernelTimeMs === "number" ? null : JSON.stringify(k.kernelTimeMs),
      input_shapes: JSON.stringify(k.inputShapes), output_shapes: JSON.stringify(k.outputShapes),
      bytes_added: k.bytesAdded, train_phase: phaseOfIdx.get(idx) ?? null,
    }));
  }
}

async function timePhase(runner, count, runs) {
  for (let i = 0; i < count; i++) {
    let held = null;
    const t0 = performance.now();
    const tm = await tf.time(() => { held = runner.executeSyncOnce(); });
    const t1 = performance.now();
    await runner.readAndRelease(held);
    runs.push({
      phase: "profile_time", i, t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: performance.now() - t0,
      time_wall_ms: tm.wallMs, time_kernel_ms: typeof tm.kernelMs === "number" ? tm.kernelMs : null,
      time_kernel_error: typeof tm.kernelMs === "number" ? null : JSON.stringify(tm.kernelMs),
      time_upload_wait_ms: tm.uploadWaitMs ?? null, time_download_wait_ms: tm.downloadWaitMs ?? null,
      time_call_ms: t1 - t0,
    });
  }
}

async function drawsPhase(runner, count, runs, profile) {
  for (let i = 0; i < count; i++) {
    startDrawTimers();
    const r = await runner.run();
    const draws = await collectDrawTimers();
    const valid = draws.filter(d => d.gpu_ms !== null);
    runs.push({
      phase: "profile_draws", i, ...r,
      gpu_draws: draws.length, draw_timer_invalid: draws.length - valid.length,
      draw_gpu_ms: valid.length === draws.length ? valid.reduce((a, d) => a + d.gpu_ms, 0) : null,
      draw_disjoint: draws.filter(d => d.disjoint).length,
    });
    if (i === 0) {
      draws.forEach(d => profile.kernels.push({ source: "webgl_draw_timer", phase: "profile_draws", run: i,
        idx: d.drawIndex, name: `program:${d.program}`, kernel_ms: d.gpu_ms, disjoint: d.disjoint }));
    }
  }
}

/** cfg.warmup, cfg.measure; runner from runner.js. Counters and hooks are already installed. */
export async function profileMode(runner, cfg, result, backend) {
  const profile = { kernels: [], memory: result.profile_memory ?? {}, phases: [] };
  result.profile = profile;
  let count = cfg.measure;

  // Warm-up: first run separately so memory right after it is visible.
  if (cfg.warmup > 0) {
    result.runs.push({ phase: "warmup", i: 0, ...(await runner.run()) });
    profile.memory.after_first_run = memorySnapshot(true);
    for (let i = 1; i < cfg.warmup; i++) result.runs.push({ phase: "warmup", i, ...(await runner.run()) });
  }
  profile.memory.after_warmup = memorySnapshot(true);
  resetPeak();

  // Measurement cap (cfg.maxMeasureS): each phase must run the same count (the counter phase's
  // last run is special), so the count is fixed up front from the last warm-up run's duration,
  // bounded to [min(5, measure), measure]. A cut count is recorded in result.capped.
  if (cfg.maxMeasureS && cfg.warmup > 0) {
    const lastMs = result.runs[result.runs.length - 1].dur_ms;
    const fit = Math.floor((cfg.maxMeasureS * 1000) / Math.max(lastMs, 1e-3));
    count = Math.max(Math.min(5, cfg.measure), Math.min(cfg.measure, fit));
    if (count < cfg.measure) result.capped.push({ phase: "profile", runs: count, warmup_last_ms: lastMs });
  }
  profile.runs_per_phase = count;

  await countersPhase(runner, count, result.runs, profile);
  profile.phases.push("profile_counters");
  profile.memory.after_counter_runs = memorySnapshot(true);

  await kernelsPhase(runner, count, result.runs, profile);
  profile.phases.push("profile_kernels");

  if (runner.exec === "execute") {
    await timePhase(runner, count, result.runs);
    profile.phases.push("profile_time");
  } else {
    profile.profile_time_skipped = "tf.time() needs a synchronous function; this model needs executeAsync";
  }

  if (backend === "webgl") {
    await drawsPhase(runner, count, result.runs, profile);
    profile.phases.push("profile_draws");
  }
  profile.memory.after_runs = memorySnapshot(true);
}
