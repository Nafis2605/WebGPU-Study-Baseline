// One timed training step of a code-defined model (custom loop; the study's primary training
// measurement). model.fit() is measured separately (train-modes.js, mode "train_fit").
//
// step():
//   before the timed region  the batch's rows (order.u32) are gathered and normalised into CPU
//                            Float32Arrays (host input pipeline; prep_ms, reported separately)
//   timed                    t0 -> tensors created from those arrays (uploaded on first use)
//                               -> tf.variableGrads(): forward pass + loss (engine scope "forward"),
//                                  backprop (scope "backward")
//                               -> optimizer.applyGradients() inside tf.tidy("optimizer")
//                            t1 (exec_ms: everything enqueued; GPU work may still be running)
//                               -> sync policy:
//                                  "loss"  (primary) await loss.data(): the loss value on the CPU.
//                                          Queued after the optimizer update on the same queue, so it
//                                          also waits for the update (checked by probe P0.6)
//                                  "fence" gpuFence() first (gpu_done_ms), then loss.data()
//                            t2 = dur_ms
//
// pipelined(n): n steps enqueued back to back without reading anything, then one fence and one loss
// read: a window record whose dur_ms covers all n steps (how fast training runs when the CPU never
// waits for the GPU).
//
// Data I/O "resident": the batch tensors are created and uploaded (identity kernel + fence) before
// t0, so the step excludes the host->device copy.
//
// The model's own layers run in training mode (model.apply(x, {training: true})); the suite has no
// dropout or batch norm, so this changes no math, only TF.js's code path.
import { tf, gpuFence } from "./tfjs-setup.js";
import { trainableVariables } from "./models/build.js";
import { imagesF32, oneHot, trainRows, PIXELS, CLASSES } from "./data.js";

/**
 * TF.js tfjs-layers categoricalCrossentropy (fromLogits=false), mean over the batch, with the clip epsilon
 * pinned by the spec (1e-7). tfjs-layers itself uses tf.backend().epsilon(), which is 1e-4 on WebGL
 * (backend_webgl.js floatPrecision()); model.fit() (mode train_fit) keeps that library default.
 */
export const LOSS_EPSILON = 1e-7;
export function cceLoss(y, p, eps = LOSS_EPSILON) {
  const q = tf.clipByValue(tf.div(p, tf.sum(p, -1, true)), eps, 1 - eps);
  return tf.mean(tf.neg(tf.sum(tf.mul(y, tf.log(q)), -1)));
}

export function makeOptimizer(spec) {
  const o = spec.optimizer;
  if (o.type !== "adam") throw new Error(`optimizer ${o.type} not supported`);
  // epsilon pinned (spec): tf.train.adam's default would be tf.backend().epsilon() (1e-4 on WebGL).
  return tf.train.adam(o.learning_rate, o.beta1, o.beta2, o.epsilon);
}

export function makeTrainRunner(model, slots, data, spec, cfg, batch) {
  const B = batch;
  const vars = trainableVariables(slots);
  const optimizer = makeOptimizer(spec);
  const policy = cfg.syncPolicy ?? "loss";
  const resident = cfg.dataIo === "resident";
  const xDims = [B, 32, 32, 3];
  const yDims = [B, CLASSES];
  const xBuf = new Float32Array(B * PIXELS);
  const yBuf = new Float32Array(B * CLASSES);
  let k = 0;   // global step index (determines the batch)

  function prepare() {
    const p0 = performance.now();
    const { rows, epoch } = trainRows(data, k, B);
    imagesF32(data.trainX, rows, data.lut, xBuf);
    oneHot(data.trainY, rows, yBuf);
    return { epoch, prep_ms: performance.now() - p0, step: k++ };
  }

  async function residentBatch() {
    const t = tf.tidy(() => [tf.add(tf.tensor(xBuf, xDims, "float32"), 0), tf.add(tf.tensor(yBuf, yDims, "float32"), 0)]);
    await gpuFence();
    return t;
  }

  /** Forward + backward + update, enqueued; returns the (unread) loss tensor. Disposes x, y. */
  function enqueueStep(x, y) {
    const { value, grads } = tf.variableGrads(() => cceLoss(y, model.apply(x, { training: true })), vars);
    tf.tidy("optimizer", () => optimizer.applyGradients(grads));
    tf.dispose(grads);
    x.dispose();
    y.dispose();
    return value;
  }

  return {
    batch: B,
    exec: "execute",
    spec: [{ name: "images", dtype: "float32", dims: xDims }, { name: "labels", dtype: "float32", dims: yDims }],
    optimizer,
    stepIndex: () => k,

    async run() {
      const info = prepare();
      const pre = resident ? await residentBatch() : null;
      const t0 = performance.now();
      const [x, y] = pre ?? [tf.tensor(xBuf, xDims, "float32"), tf.tensor(yBuf, yDims, "float32")];
      const loss = enqueueStep(x, y);
      const t1 = performance.now();
      let tf_ = t1;
      if (policy === "fence") { await gpuFence(); tf_ = performance.now(); }
      const lossVal = (await loss.data())[0];
      const t2 = performance.now();
      loss.dispose();
      const rec = { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t2 - t0, exec_ms: t1 - t0, read_ms: t2 - tf_,
        step: info.step, epoch: info.epoch, batch: B, prep_ms: info.prep_ms, loss: lossVal,
        nonfinite: !Number.isFinite(lossVal), num_tensors: tf.memory().numTensors };
      if (policy === "fence") rec.gpu_done_ms = tf_ - t1;
      return rec;
    },

    /**
     * Phase ablation (mode train_phases), same timing as run() but doing only part of a step:
     *   "fwd"     forward pass + loss, no gradient tape
     *   "fwdbwd"  forward + backward (variableGrads), gradients discarded, no update
     *   "step"    the full step (= run())
     */
    async runPart(part) {
      if (part === "step") return this.run();
      const info = prepare();
      const t0 = performance.now();
      const x = tf.tensor(xBuf, xDims, "float32");
      const y = tf.tensor(yBuf, yDims, "float32");
      let loss;
      if (part === "fwd") {
        loss = tf.tidy("forward", () => cceLoss(y, model.apply(x, { training: true })));
      } else {
        const { value, grads } = tf.variableGrads(() => cceLoss(y, model.apply(x, { training: true })), vars);
        tf.dispose(grads);
        loss = value;
      }
      x.dispose();
      y.dispose();
      const t1 = performance.now();
      const lossVal = (await loss.data())[0];
      const t2 = performance.now();
      loss.dispose();
      return { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t2 - t0, exec_ms: t1 - t0, read_ms: t2 - t1,
        step: info.step, epoch: info.epoch, batch: B, prep_ms: info.prep_ms, loss: lossVal,
        nonfinite: !Number.isFinite(lossVal), num_tensors: tf.memory().numTensors };
    },

    /** n unsynced steps then one fence + loss read; host batch prep stays outside the window. */
    async pipelined(n) {
      const batches = [];
      for (let i = 0; i < n; i++) {
        const info = prepare();
        batches.push({ info, x: new Float32Array(xBuf), y: new Float32Array(yBuf) });
      }
      const t0 = performance.now();
      let loss = null;
      for (const b of batches) {
        loss?.dispose();
        loss = enqueueStep(tf.tensor(b.x, xDims, "float32"), tf.tensor(b.y, yDims, "float32"));
      }
      const t1 = performance.now();
      await gpuFence();
      const tf_ = performance.now();
      const lossVal = (await loss.data())[0];
      const t2 = performance.now();
      loss.dispose();
      return { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t2 - t0, exec_ms: t1 - t0, gpu_done_ms: tf_ - t1,
        read_ms: t2 - tf_, window_steps: n, step: batches[0].info.step, epoch: batches[n - 1].info.epoch, batch: B,
        loss: lossVal, nonfinite: !Number.isFinite(lossVal), num_tensors: tf.memory().numTensors };
    },

    // Profile-pass interface (src/profile.js): one step enqueued without readback, then read + release.
    async executeOnce() {
      prepare();
      return { feeds: {}, out: [enqueueStep(tf.tensor(xBuf, xDims, "float32"), tf.tensor(yBuf, yDims, "float32"))] };
    },
    executeSyncOnce() {
      prepare();
      return { feeds: {}, out: [enqueueStep(tf.tensor(xBuf, xDims, "float32"), tf.tensor(yBuf, yDims, "float32"))] };
    },
    async readAndRelease({ out }) {
      await out[0].data();
      out[0].dispose();
    },

    /** Untimed: step 0's gradients (correctness mode). Doesn't advance the step or touch the optimizer. */
    async gradientsAt(stepIndex) {
      const { rows } = trainRows(data, stepIndex, B);
      const x = tf.tensor(imagesF32(data.trainX, rows, data.lut), xDims, "float32");
      const y = tf.tensor(oneHot(data.trainY, rows), yDims, "float32");
      const { value, grads } = tf.variableGrads(() => cceLoss(y, model.apply(x, { training: true })), vars);
      const out = { loss: (await value.data())[0], grads: [] };
      for (const v of vars) out.grads.push(await grads[v.name].data());
      tf.dispose([value, grads, x, y]);
      return out;
    },

    dispose() {
      optimizer.dispose();
    },
  };
}
