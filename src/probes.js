// Phase-0 feasibility probes that need code the harness modes don't have (probes/run_probes.py drives
// them through window.__bench.probes, one fresh browser per call). Their numbers decide design
// questions (study_report/FEASIBILITY.md); none is reported as a study result.
//
//   syncSemantics  P0.6  Does `await loss.data()` after applyGradients() wait for the whole step
//                        (forward, backward, optimizer update)? After each awaited loss read, a
//                        gpuFence() is timed; if the read waited for everything, the fence finds an
//                        empty queue and costs what a fence on an idle queue costs (baseline, measured
//                        first). Also checks that a weight changes between steps.
//   trainPrecompile P0.7 TF.js compile-only mode around one training step: are later steps free of
//                        pipeline / program creation, and do losses match a run without it?
//   fitYields      P0.14 model.fit() with yieldEvery 'never' and no callbacks: requestAnimationFrame
//                        calls and backend reads during fit.
import { tf, applyConfig, gpuFence } from "./tfjs-setup.js";
import { buildModel, weightSlots } from "./models/build.js";
import { loadData, imagesF32, oneHot } from "./data.js";
import { fetchWeights, assignWeights } from "./weights.js";
import { makeTrainRunner, makeOptimizer } from "./train-runner.js";
import { installCounters, uninstallCounters, resetCounters, readCounters } from "./instrument.js";

async function setup(cfg, kind = "init") {
  const info = await applyConfig(cfg);
  const data = await loadData();
  const w = await fetchWeights(cfg.model.name, kind);
  const model = buildModel(cfg.model);
  const slots = weightSlots(model, cfg.model);
  assignWeights(slots, w);
  return { info, data, w, model, slots };
}

const median = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

export async function syncSemantics(cfg) {
  const { info, data, model, slots } = await setup(cfg);
  // Baseline: fence cost on an idle queue (nothing enqueued since the last fence).
  const idle = [];
  for (let i = 0; i < 20; i++) {
    await gpuFence();
    const t0 = performance.now();
    await gpuFence();
    idle.push(performance.now() - t0);
  }
  const runner = makeTrainRunner(model, slots, data, cfg.model, { syncPolicy: "loss" }, cfg.model.batch);
  const residual = [];
  const steps = [];
  const w0 = (await slots[0].variable.read().data()).slice(0, 4);
  for (let i = 0; i < (cfg.steps ?? 20); i++) {
    const r = await runner.run();
    const t0 = performance.now();
    await gpuFence();
    residual.push(performance.now() - t0);
    steps.push(r.dur_ms);
  }
  const w1 = (await slots[0].variable.read().data()).slice(0, 4);
  runner.dispose();
  model.dispose();
  return {
    backend: info.backend, idle_fence_ms: idle, residual_fence_ms: residual, step_ms: steps,
    idle_fence_median_ms: median(idle), residual_fence_median_ms: median(residual.slice(5)),
    step_median_ms: median(steps.slice(5)), weight_changed: Array.from(w0).some((v, k) => v !== w1[k]),
  };
}

export async function trainPrecompile(cfg) {
  const backend = cfg.backend;
  const { info, data, w, model, slots } = await setup(cfg);
  const flag = backend === "webgpu" ? "WEBGPU_ENGINE_COMPILE_ONLY" : "ENGINE_COMPILE_ONLY";
  installCounters();
  const out = { backend: info.backend };
  try {
    // Compile-only pass of one full step on a throwaway optimizer.
    const pre = makeTrainRunner(model, slots, data, cfg.model, {}, cfg.model.batch);
    resetCounters();
    const p0 = performance.now();
    tf.env().set(flag, true);
    try {
      const held = pre.executeSyncOnce();
      await tf.backend().checkCompileCompletionAsync();
      if (backend === "webgl") tf.backend().getUniformLocations();
      held.out[0].dispose();
    } finally {
      tf.env().set(flag, false);
    }
    out.precompile_ms = performance.now() - p0;
    out.precompile_counters = readCounters();
    pre.dispose();
    // Compile-only mode ran the optimizer's assigns on undefined data: restore the weights.
    assignWeights(slots, w);
    const runner = makeTrainRunner(model, slots, data, cfg.model, {}, cfg.model.batch);
    out.steps = [];
    for (let i = 0; i < 5; i++) {
      resetCounters();
      const r = await runner.run();
      const c = readCounters();
      out.steps.push({ dur_ms: r.dur_ms, loss: r.loss, pipelines: c.webgpu_pipelines_created + c.webgpu_pipelines_created_async,
        programs: c.webgl_programs_linked });
    }
    runner.dispose();
  } catch (e) {
    out.error = String(e?.message ?? e);
  } finally {
    uninstallCounters();
    model.dispose();
  }
  return out;
}

export async function fitYields(cfg) {
  const { info, data, model } = await setup(cfg);
  model.compile({ optimizer: makeOptimizer(cfg.model), loss: "categoricalCrossentropy" });
  const n = cfg.model.batch * 8;
  const rows = data.order.subarray(0, n);
  const x = tf.tensor(imagesF32(data.trainX, rows, data.lut), [n, 32, 32, 3]);
  const y = tf.tensor(oneHot(data.trainY, rows), [n, 10]);
  let raf = 0;
  const origRaf = window.requestAnimationFrame;
  window.requestAnimationFrame = cb => { raf++; return origRaf.call(window, cb); };
  const b = tf.backend();
  let reads = 0, syncReads = 0;
  const origRead = b.read, origReadSync = b.readSync;
  b.read = function (...a) { reads++; return origRead.apply(this, a); };
  b.readSync = function (...a) { syncReads++; return origReadSync.apply(this, a); };
  const out = { backend: info.backend, steps: 8 };
  try {
    for (const yieldEvery of ["never", "auto"]) {
      raf = 0; reads = 0; syncReads = 0;
      const t0 = performance.now();
      await model.fit(x, y, { batchSize: cfg.model.batch, epochs: 1, shuffle: false, yieldEvery, verbose: 0 });
      out[yieldEvery] = { ms: performance.now() - t0, raf_calls: raf, async_reads: reads, sync_reads: syncReads };
    }
  } finally {
    window.requestAnimationFrame = origRaf;
    b.read = origRead;
    b.readSync = origReadSync;
    tf.dispose([x, y]);
    model.dispose();
  }
  return out;
}

export const PROBES = { syncSemantics, trainPrecompile, fitYields };
