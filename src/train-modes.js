// Pillar 3 (training) and the warmup why-axis (lifecycle) modes. Training launches start from the
// study's initial weights (identical bytes natively); the step is src/train-runner.js.
//
// Phase labels in runs.csv:
//   train_warmup / train_measure      the custom-loop steps (warm-up curve / steady state)
//   infer_b1_warmup / infer_b1        post-training inference at batch 1 (first call = first_seen)
//   infer_bB_warmup / infer_bB        post-training inference at the training batch size
//   train_pipelined                   window records (n unsynced steps + one fence)
//   train_batch_warmup / train_batch  batch sweep
//   train_long                        every step of an E-epoch run
//   fit_warmup / fit_measure          model.fit() calls (whole-call records)
//   phase_fwd / phase_fwdbwd / phase_step   ablation
//   correctness_step                  untimed-purpose steps of the correctness mode
//   lc_*                              lifecycle mode (cross-phase warm-up transfer)
import { tf } from "./tfjs-setup.js";
import { makeTrainRunner, makeOptimizer, cceLoss } from "./train-runner.js";
import { inferRunner } from "./infer-modes.js";
import { instrumented, runnerRuns, capOf, encodeTensor, mulberry32, fnv1a32, pageMemory, warmupCap } from "./common.js";
import { profileMode } from "./profile.js";
import { responsivenessMode } from "./responsiveness.js";
import { imagesF32, oneHot, valRows, PIXELS, CLASSES } from "./data.js";

export const DEFAULT_TRAIN_BATCH_VALUES = [16, 32, 64, 128, 256];

function trainRunner(model, ctx, cfg, B) {
  const r = makeTrainRunner(model, ctx.slots, ctx.data, ctx.spec, cfg, B ?? cfg.trainBatch ?? ctx.spec.batch);
  return cfg.instrumentRuns ? instrumented(r) : r;
}

/**
 * A train runner that has taken one (unrecorded) step, so its optimizer slots exist before the
 * verification step (which checks that a step leaves the tensor count unchanged).
 */
async function warmedTrainRunner(model, ctx, cfg) {
  const r = trainRunner(model, ctx, cfg);
  await r.run();
  return r;
}

const trainCap = cfg => capOf(cfg.trainMaxS, cfg.trainMinMeasure, cfg.trainMeasure, cfg.minMeasureS);
const inferCap = cfg => capOf(cfg.inferMaxS, cfg.inferMinMeasure, cfg.inferMeasure, cfg.minMeasureS);

/** Warm-up + capped measure of one runner, labelled with a phase prefix. */
// Warm-up rule (count, time floor, time cap): common.js warmupCap().
async function warmMeasure(runner, prefix, warmup, measure, cap, result, extra = {}) {
  const w = await runnerRuns(runner, `${prefix}_warmup`, warmup, result.runs, extra, warmupCap());
  if (w.capped) result.capped.push({ phase: `${prefix}_warmup`, runs: w.runs, ...extra });
  const m = await runnerRuns(runner, prefix, measure, result.runs, extra, cap);
  if (m.capped) result.capped.push({ phase: prefix, runs: m.runs, ...extra });
  return m;
}

/** Inference after training at batch 1 and at the training batch size. */
async function inferAfter(model, ctx, cfg, result, B, prefix = "infer") {
  for (const [tag, b] of [["b1", 1], ["bB", B]]) {
    const runner = inferRunner(model, ctx, cfg, b);
    await warmMeasure(runner, `${prefix}_${tag}`, cfg.inferWarmup, cfg.inferMeasure, inferCap(cfg), result, { batch: b });
    runner.dispose();
  }
}

/** Projections <g, r_k> (k = 0..3) of a flat array on +-1 vectors seeded by fnv1a32("<layer>/<name>/<k>"), float64. */
export function sketch(values, slotName, k = 4) {
  const out = [];
  for (let j = 0; j < k; j++) {
    const rand = mulberry32(fnv1a32(`${slotName}/${j}`));
    let s = 0;
    for (let i = 0; i < values.length; i++) s += rand() < 0.5 ? -values[i] : values[i];
    out.push(s);
  }
  return out;
}

function sumSq(values) {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i] * values[i];
  return s;
}

/** Mean loss / accuracy over the validation subset (forward only, untimed). */
async function evaluate(model, data, B) {
  let lossSum = 0, correct = 0;
  for (let start = 0; start < data.n_val; start += B) {
    const n = Math.min(B, data.n_val - start);
    const rows = Uint32Array.from({ length: n }, (_, r) => start + r);
    const [l, acc] = tf.tidy(() => {
      const x = tf.tensor(imagesF32(data.valX, rows, data.lut), [n, 32, 32, 3]);
      const y = tf.tensor(oneHot(data.valY, rows), [n, CLASSES]);
      const p = model.apply(x, { training: false });
      return [tf.mul(cceLoss(y, p), n), tf.sum(tf.cast(tf.equal(tf.argMax(p, -1), tf.argMax(y, -1)), "float32"))];
    });
    lossSum += (await l.data())[0];
    correct += (await acc.data())[0];
    tf.dispose([l, acc]);
  }
  return { val_loss: lossSum / data.n_val, val_acc: correct / data.n_val };
}

export const TRAIN_MODES = {
  /** Primary training mode: warm-up curve, steady steps, then post-training inference. */
  async train(model, ctx, cfg, result) {
    const runner = trainRunner(model, ctx, cfg);
    const B = runner.batch;
    await warmMeasure(runner, "train", cfg.trainWarmup, cfg.trainMeasure, trainCap(cfg), result);
    // Rename: warmMeasure labels "<prefix>_warmup" / "<prefix>"; the training phases are train_warmup / train_measure.
    for (const r of result.runs) if (r.phase === "train") r.phase = "train_measure";
    if (cfg.inferAfterTrain !== false) await inferAfter(model, ctx, cfg, result, B);
    return runner;
  },

  /** Sync-policy why-axis: n unsynced steps per window, one fence + loss read per window. */
  async train_pipelined(model, ctx, cfg, result) {
    const runner = trainRunner(model, ctx, cfg);
    await runnerRuns(runner, "train_warmup", cfg.trainWarmup, result.runs, {}, warmupCap());
    const n = cfg.pipelineWindow ?? 10;
    const cap = trainCap(cfg);
    const windows = Math.max(1, Math.ceil(cfg.trainMeasure / n));
    const t0 = performance.now();
    let i = 0;
    for (; i < windows; i++) {
      if (cap && i * n >= cap.min && performance.now() - t0 > cap.maxMs) break;
      result.runs.push({ phase: "train_pipelined", i, ...(await runner.pipelined(n)) });
    }
    if (i < windows) result.capped.push({ phase: "train_pipelined", runs: i });
    result.config.pipeline_window = n;
    return runner;
  },

  /** Instrumented training pass (profile.js on a train runner): counters, tf.profile by phase, tf.time, draws, memory. */
  async train_profile(model, ctx, cfg, result) {
    const runner = makeTrainRunner(model, ctx.slots, ctx.data, ctx.spec, cfg, cfg.trainBatch ?? ctx.spec.batch);
    await profileMode(runner, { ...cfg, warmup: cfg.trainWarmup, measure: cfg.profileSteps ?? 5, maxMeasureS: cfg.trainMaxS }, result, cfg.backend);
    result.page_memory.after_runs = await pageMemory();
    return runner;
  },

  /**
   * Numerical agreement: step-0 loss and gradients (sketch for every variable; full arrays for small
   * models), then K steps' losses, then the weights after K steps (sketch; full for small models).
   * Compared offline with native tf-cpu / tf-cuda-fp32 and the TF.js cpu backend on identical bytes.
   */
  async train_correctness(model, ctx, cfg, result) {
    const runner = makeTrainRunner(model, ctx.slots, ctx.data, ctx.spec, cfg, cfg.trainBatch ?? ctx.spec.batch);
    const full = ctx.spec.params <= (cfg.dumpParamsMax ?? 3_000_000);
    const g0 = await runner.gradientsAt(0);
    const corr = { step0_loss: g0.loss, full_dumps: full, grads: [], weights_after: [] };
    const io = { grads: [], weights_after: [] };
    ctx.slots.forEach((s, j) => {
      const name = `${s.layer}/${s.name}`;
      corr.grads.push({ name, sketch: sketch(g0.grads[j], name), sumsq: sumSq(g0.grads[j]) });
      if (full) io.grads.push(encodeTensor(name, "float32", s.shape, g0.grads[j]));
    });
    const K = cfg.correctnessSteps ?? 20;
    for (let i = 0; i < K; i++) result.runs.push({ phase: "correctness_step", i, ...(await runner.run()) });
    for (const s of ctx.slots) {
      const name = `${s.layer}/${s.name}`;
      const w = await s.variable.read().data();
      corr.weights_after.push({ name, sketch: sketch(w, name), sumsq: sumSq(w) });
      if (full) io.weights_after.push(encodeTensor(name, "float32", s.shape, w));
    }
    corr.steps = K;
    result.train_correctness = corr;
    if (full) result.io = io;
    return runner;
  },

  /** Training batch sweep (ascending), each batch with its own optimizer state. */
  async train_batch(model, ctx, cfg, result) {
    const values = [...(cfg.trainBatchValues ?? DEFAULT_TRAIN_BATCH_VALUES)].sort((a, b) => a - b);
    result.config.train_batch_values = values;
    for (const b of values) {
      const runner = trainRunner(model, ctx, cfg, b);
      await warmMeasure(runner, "train_batch", cfg.batchWarmup ?? 5, cfg.trainMeasure, capOf(cfg.batchMaxS ?? 10, cfg.batchMinMeasure ?? 10, cfg.trainMeasure), result, { batch: b });
      runner.dispose();
    }
    return warmedTrainRunner(model, ctx, cfg);
  },

  /** E full epochs, every step recorded; validation loss/accuracy after each epoch (untimed). */
  async train_long(model, ctx, cfg, result) {
    const runner = trainRunner(model, ctx, cfg);
    const B = runner.batch;
    const E = cfg.longEpochs ?? 3;
    const perEpoch = ctx.data.n_train / B;
    result.long = { epochs: E, steps_per_epoch: perEpoch, val: [] };
    for (let e = 0; e < E; e++) {
      for (let i = 0; i < perEpoch; i++) result.runs.push({ phase: "train_long", i: e * perEpoch + i, ...(await runner.run()) });
      result.long.val.push({ epoch: e, ...(await evaluate(model, ctx.data, B)) });
    }
    return runner;
  },

  /**
   * model.fit() (the Layers API's own training loop): one 1-epoch call (warm-up), then one call of
   * cfg.fitEpochs epochs (measured). yieldEvery 'never', no callbacks, no metrics, shuffle false; the
   * training set is passed in epoch-0 order. Records are whole fit() calls.
   */
  async train_fit(model, ctx, cfg, result) {
    const B = cfg.trainBatch ?? ctx.spec.batch;
    const data = ctx.data;
    const rows = data.order.subarray(0, data.n_train);
    const c0 = performance.now();
    model.compile({ optimizer: makeOptimizer(ctx.spec), loss: "categoricalCrossentropy" });
    result.compile_ms = performance.now() - c0;
    const x = tf.tensor(imagesF32(data.trainX, rows, data.lut), [data.n_train, 32, 32, 3]);
    const y = tf.tensor(oneHot(data.trainY, rows), [data.n_train, CLASSES]);
    const E = cfg.fitEpochs ?? 2;
    const steps = data.n_train / B;
    try {
      for (const [phase, epochs] of [["fit_warmup", 1], ["fit_measure", E]]) {
        const t0 = performance.now();
        const h = await model.fit(x, y, { batchSize: B, epochs, shuffle: false, yieldEvery: "never", verbose: 0 });
        const dur = performance.now() - t0;
        result.runs.push({ phase, i: 0, t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: dur, epochs, steps: steps * epochs,
          per_step_ms: dur / (steps * epochs), batch: B, loss: h.history.loss[h.history.loss.length - 1],
          loss_history: JSON.stringify(h.history.loss), num_tensors: tf.memory().numTensors });
      }
    } finally {
      tf.dispose([x, y]);
    }
    // Verification runs one custom-loop step on the same model.
    return warmedTrainRunner(model, ctx, cfg);
  },

  /** Ablation: forward-only, forward+backward, full step; each warmed up and measured separately. */
  async train_phases(model, ctx, cfg, result) {
    const runner = trainRunner(model, ctx, cfg);
    for (const part of ["fwd", "fwdbwd", "step"]) {
      const r = { run: () => runner.runPart(part) };
      await warmMeasure(r, `phase_${part}`, cfg.trainWarmup, cfg.trainMeasure, capOf(cfg.phaseMaxS ?? 10, cfg.trainMinMeasure, cfg.trainMeasure), result);
    }
    return runner;
  },

  /** Main-thread availability while training runs (responsiveness.js with a train runner). */
  async train_responsiveness(model, ctx, cfg, result) {
    const runner = trainRunner(model, ctx, cfg);
    await responsivenessMode(runner, { ...cfg, warmup: cfg.trainWarmup }, result);
    return runner;
  },

  /**
   * Warm-up transfer between pillars in one cold launch (cfg.lifecycleOrder):
   *   infer_first  inference (lc_infer1) -> training (lc_train) -> inference again (lc_infer2)
   *   train_first  training (lc_train1) -> inference (lc_infer) -> training again (lc_train2)
   * Compared offline with the cold first-run / first-step costs of the latency / train modes.
   */
  async lifecycle(model, ctx, cfg, result) {
    const order = cfg.lifecycleOrder ?? "infer_first";
    result.config.lifecycle_order = order;
    const B = cfg.trainBatch ?? ctx.spec.batch;
    const train = trainRunner(model, ctx, cfg, B);
    const infer = inferRunner(model, ctx, cfg, cfg.inferBatch ?? 1);
    const doInfer = p => warmMeasure(infer, p, cfg.inferWarmup, cfg.inferMeasure, inferCap(cfg), result);
    const doTrain = p => warmMeasure(train, p, cfg.trainWarmup, cfg.trainMeasure, trainCap(cfg), result);
    if (order === "infer_first") {
      await doInfer("lc_infer1");
      await doTrain("lc_train");
      await doInfer("lc_infer2");
    } else {
      await doTrain("lc_train1");
      await doInfer("lc_infer");
      await doTrain("lc_train2");
    }
    infer.dispose();
    return train;
  },
};

export { PIXELS };
