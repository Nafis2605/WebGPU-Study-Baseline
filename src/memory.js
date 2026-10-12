// Memory snapshots (group 7). Every field is read from TF.js or the browser; nothing is derived.
//
//   tf_memory      tf.memory(): numTensors, numDataBuffers, numBytes (tensor bytes), plus the
//                  backend's own fields (WebGL numBytesInGPU/Allocated/Free, WebGPU
//                  numBytesInGPU/numBytesAllocatedInGPU; WASM has none)
//   pool           the backend's resource pool: WebGPU bufferManager (used/free buffers, bytes
//                  used/allocated), WebGL textureManager (used/free textures, bytes allocated/free),
//                  WASM linear-memory (heap) size. The pool keeps released buffers/textures for
//                  reuse, so allocated >= used.
//   logical        instrument.js live/peak bytes of GPU buffers/textures created by the page
//                  (only while counters are installed)
import { tf } from "./tfjs-setup.js";
import { readMemory } from "./instrument.js";

function pick(obj, keys) {
  const out = {};
  for (const k of keys) {
    const v = typeof obj?.[k] === "function" ? obj[k]() : obj?.[k];
    if (v !== undefined) out[k] = v;
  }
  return out;
}

export function poolStats() {
  const backend = tf.getBackend();
  const b = tf.backend();
  if (backend === "webgpu" && b.bufferManager) {
    const m = b.bufferManager;
    return { kind: "webgpu_buffer_pool", used_buffers: m.getNumUsedBuffers(), free_buffers: m.getNumFreeBuffers(),
      bytes_used: m.numBytesUsed, bytes_allocated: m.numBytesAllocated };
  }
  if (backend === "webgl" && b.textureManager) {
    const m = b.textureManager;
    const p = pick(m, ["numUsedTextures", "numFreeTextures", "numBytesAllocated", "numBytesFree"]);
    return { kind: "webgl_texture_pool", used_textures: p.numUsedTextures ?? null, free_textures: p.numFreeTextures ?? null,
      bytes_allocated: p.numBytesAllocated ?? m._numBytesAllocated ?? null, bytes_free: p.numBytesFree ?? m._numBytesFree ?? null };
  }
  if (backend === "wasm") {
    const w = b.wasm;
    const buf = w?.wasmMemory?.buffer ?? w?.HEAPU8?.buffer;
    return { kind: "wasm_heap", heap_bytes: buf ? buf.byteLength : null };
  }
  return { kind: backend, note: "no pool statistics for this backend" };
}

/** One snapshot. logical=true only when instrument.js counters are installed. */
export function memorySnapshot(logical = false) {
  return {
    tf_memory: tf.memory(),
    pool: poolStats(),
    logical: logical ? readMemory() : null,
  };
}
