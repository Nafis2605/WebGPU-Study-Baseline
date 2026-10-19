// One timed inference of a code-defined model (the counterpart of the sibling study's runner.js,
// with the same interface, so its latency / shapes / batch / profile / correctness / load /
// responsiveness modes run unchanged).
//
// run(): the batch starts as a CPU Float32Array (validation images, prepared before the timed
// region); inside the timed region a tensor is created from it (TF.js uploads it on first use),
// model.predict() runs (the standard Layers API inference call), and the [B, 10] output is read
// back with data(). dur_ms = request start -> output on the CPU; exec_ms = until predict() returned
// (GPU work may still be queued); read_ms = data().
//
// Inputs rotate over POOL distinct batches of real validation images (fixed order), so consecutive
// runs don't reuse one array; the correctness reference uses pool[0].
//
// Variants (experiment config):
//   io "gpu"         the input is uploaded once before the runs and stays GPU-resident
//   fence            gpuFence() between predict() and the readback (gpu_done_ms; read_ms = transfer)
//   readback "sync"  dataSync() instead of data()
import { tf, gpuFence } from "./tfjs-setup.js";
import { imagesF32, valRows, PIXELS, CLASSES } from "./data.js";

const POOL = 8;

export function makeLayersRunner(model, data, cfg, batch) {
  const B = batch;
  const dims = [B, 32, 32, 3];
  const pool = Array.from({ length: POOL }, (_, i) => imagesF32(data.valX, valRows(data, i, B), data.lut));
  const gpuIo = cfg.io === "gpu";
  const fence = !!cfg.fence;
  const syncRead = cfg.readback === "sync";
  let k = 0;
  let resident = null;
  if (gpuIo) {
    // Upload once (an identity kernel makes the tensor GPU-resident), outside every timed region.
    resident = tf.tidy(() => tf.add(tf.tensor(pool[0], dims, "float32"), 0));
  }
  const predict = x => model.predict(x, { batchSize: B });

  return {
    exec: "execute",
    batch: B,
    spec: [{ name: "images", dtype: "float32", dims }],
    output_names: ["probs"],
    async run() {
      const arr = pool[k++ % POOL];
      const t0 = performance.now();
      const x = gpuIo ? resident : tf.tensor(arr, dims, "float32");
      const out = predict(x);
      const t1 = performance.now();
      let tf_ = t1;
      if (fence) { await gpuFence(); tf_ = performance.now(); }
      if (syncRead) out.dataSync();
      else await out.data();
      const t2 = performance.now();
      if (!gpuIo) x.dispose();
      out.dispose();
      const rec = { t_start_epoch_ms: performance.timeOrigin + t0, dur_ms: t2 - t0, exec_ms: t1 - t0, read_ms: t2 - tf_, batch: B };
      if (fence) rec.gpu_done_ms = tf_ - t1;
      return rec;
    },
    async outputs() {
      const x = tf.tensor(pool[0], dims, "float32");
      const out = predict(x);
      const res = [{ name: "probs", dtype: out.dtype, shape: out.shape, data: await out.data() }];
      x.dispose();
      out.dispose();
      return res;
    },
    async executeOnce() {
      const x = tf.tensor(pool[k++ % POOL], dims, "float32");
      return { feeds: { images: x }, out: [predict(x)] };
    },
    executeSyncOnce() {
      const x = tf.tensor(pool[k++ % POOL], dims, "float32");
      return { feeds: { images: x }, out: [predict(x)] };
    },
    async readAndRelease({ feeds, out }) {
      await out[0].data();
      for (const t of Object.values(feeds)) t.dispose();
      for (const t of out) t.dispose();
    },
    async compileOnce() {
      const x = tf.tensor(pool[0], dims, "float32");
      const out = predict(x);
      x.dispose();
      out.dispose();
    },
    inputData() {
      return [{ name: "images", dtype: "float32", dims, data: pool[0] }];
    },
    input_bytes: B * PIXELS * 4,
    output_bytes: B * CLASSES * 4,
    dispose() {
      resident?.dispose();
    },
  };
}
