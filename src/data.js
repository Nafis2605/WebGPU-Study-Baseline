// The study's fixed CIFAR-10 subsets (scripts/prepare_cifar10.py), loaded from public/data/ and
// verified byte-for-byte against manifest.json (sha256 via WebCrypto). The native worker reads the
// same files (native/data.py), so both runtimes see identical inputs.
//
// Normalisation: x_f32 = LUT[x_u8], LUT[i] = Math.fround(i / 255) (float64 division, one rounding),
// the same LUT numpy builds; its sha256 must equal the manifest's lut_sha256.

export const PIXELS = 32 * 32 * 3;
export const CLASSES = 10;

export async function sha256Hex(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, "0")).join("");
}

export function makeLut() {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.fround(i / 255);
  return lut;
}

async function fetchBytes(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`data_mismatch: ${url} -> HTTP ${r.status}`);
  return r.arrayBuffer();
}

/**
 * Fetch and verify the data subsets. Returns typed-array views plus timing:
 * {trainX, trainY, valX, valY, order, lut, manifest, fetch_ms, verify_ms}
 */
export async function loadData(base = "/data/") {
  const t0 = performance.now();
  const manifest = await (await fetch(`${base}manifest.json`, { cache: "no-store" })).json();
  const names = ["train_x.u8", "train_y.u8", "val_x.u8", "val_y.u8", "order.u32"];
  const bufs = await Promise.all(names.map(n => fetchBytes(base + n)));
  const t1 = performance.now();
  for (const [k, n] of names.entries()) {
    const got = await sha256Hex(bufs[k]);
    if (got !== manifest.files[n].sha256) throw new Error(`data_mismatch: ${n} sha256 ${got} != manifest`);
  }
  const lut = makeLut();
  const lutSha = await sha256Hex(lut.buffer);
  if (lutSha !== manifest.lut_sha256) throw new Error(`data_mismatch: normalisation LUT sha256 ${lutSha} != manifest`);
  const t2 = performance.now();
  const [trainX, trainY, valX, valY] = bufs.slice(0, 4).map(b => new Uint8Array(b));
  const order = new Uint32Array(bufs[4]);
  return { trainX, trainY, valX, valY, order, lut, manifest, n_train: trainY.length, n_val: valY.length,
    epochs_in_order: manifest.epochs_in_order, fetch_ms: t1 - t0, verify_ms: t2 - t1 };
}

/** Float32 images [B * 3072] for the given row indices of a uint8 NHWC array. */
export function imagesF32(xU8, rows, lut, out = null) {
  const a = out ?? new Float32Array(rows.length * PIXELS);
  for (let r = 0; r < rows.length; r++) {
    const src = rows[r] * PIXELS, dst = r * PIXELS;
    for (let i = 0; i < PIXELS; i++) a[dst + i] = lut[xU8[src + i]];
  }
  return a;
}

/** One-hot float32 labels [B * 10]. */
export function oneHot(yU8, rows, out = null) {
  const a = out ?? new Float32Array(rows.length * CLASSES);
  a.fill(0);
  for (let r = 0; r < rows.length; r++) a[r * CLASSES + yU8[rows[r]]] = 1;
  return a;
}

/**
 * Training batch k at batch size B: rows order[epoch mod E][(k*B) mod N ... +B), epoch = floor(k*B / N),
 * E = epochs in order.u32 (10). N is divisible by every batch size in the study, so a batch never straddles
 * two epochs. Steps past E epochs reuse the orders cyclically (fast pairs with the 1 s warm-up floor and the
 * 2 s minimum window run > 800 steps at B = 64); native/data.py does the same, so batches stay identical.
 */
export function trainRows(data, k, B) {
  const N = data.n_train;
  if (N % B !== 0) throw new Error(`batch ${B} does not divide the training subset (${N})`);
  const start = k * B;
  const epoch = Math.floor(start / N);
  const off = (epoch % data.epochs_in_order) * N + (start % N);
  return { rows: data.order.subarray(off, off + B), epoch };
}

/** Inference request i at batch B: B consecutive validation images starting at (i * B) mod N_val. */
export function valRows(data, i, B) {
  const N = data.n_val;
  const rows = new Uint32Array(B);
  for (let r = 0; r < B; r++) rows[r] = (i * B + r) % N;
  return rows;
}
