// Weight files (public/weights/<model>.<kind>.bin, manifest.json): written by
// scripts/make_init_weights.py ("init") and scripts/make_trained_weights.py ("trained"). The same
// bytes are loaded natively (native/weights.py).
import { tf } from "./models/build.js";
import { sha256Hex } from "./data.js";

let manifestPromise = null;
function weightManifest() {
  manifestPromise ??= fetch("/weights/manifest.json", { cache: "no-store" }).then(r => {
    if (!r.ok) throw new Error(`weights_mismatch: manifest.json HTTP ${r.status}`);
    return r.json();
  });
  return manifestPromise;
}

/** Fetch + sha256-check one weight file. Returns {entry, buffer, fetch_ms, verify_ms}. */
export async function fetchWeights(model, kind) {
  const t0 = performance.now();
  const entry = (await weightManifest())[model]?.[kind];
  if (!entry) throw new Error(`weights_mismatch: no ${kind} weights for ${model} in manifest.json`);
  const r = await fetch(`/weights/${entry.file}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`weights_mismatch: ${entry.file} HTTP ${r.status}`);
  const buffer = await r.arrayBuffer();
  const t1 = performance.now();
  const got = await sha256Hex(buffer);
  if (got !== entry.sha256) throw new Error(`weights_mismatch: ${entry.file} sha256 ${got} != manifest`);
  return { entry, buffer, fetch_ms: t1 - t0, verify_ms: performance.now() - t1 };
}

/**
 * Write the fetched weights into the model's variables (slots from weightSlots(), spec order).
 * On WebGL/WebGPU the tensors stay CPU-side until first use: the GPU upload lands in the first
 * inference / training step, as in the sibling study.
 */
export function assignWeights(slots, { entry, buffer }) {
  if (entry.tensors.length !== slots.length) throw new Error(`weights_mismatch: ${entry.tensors.length} tensors for ${slots.length} slots`);
  slots.forEach((s, k) => {
    const t = entry.tensors[k];
    if (t.layer !== s.layer || t.name !== s.name || JSON.stringify(t.shape) !== JSON.stringify(s.shape)) {
      throw new Error(`weights_mismatch: file tensor ${t.layer}/${t.name} [${t.shape}] vs slot ${s.layer}/${s.name} [${s.shape}]`);
    }
    const val = tf.tensor(new Float32Array(buffer, t.offset, t.bytes / 4), t.shape, "float32");
    s.variable.write(val);
    val.dispose();
  });
}

/**
 * Read every weight back from the backend and compare its bytes with the manifest (gate P0.4).
 * Returns {ok, mismatches: [names]}.
 */
export async function verifyAssigned(slots, entry) {
  const mismatches = [];
  for (const [k, s] of slots.entries()) {
    const data = await s.variable.read().data();
    const got = await sha256Hex(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
    if (got !== entry.tensors[k].sha256) mismatches.push(`${s.layer}/${s.name}`);
  }
  return { ok: mismatches.length === 0, mismatches };
}
