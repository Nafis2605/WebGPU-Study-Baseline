// Graphics-API instrumentation, installed by wrapping WebGPU / WebGL prototype methods.
//
// Ported from the ONNX Runtime Web study (same file name) so the two studies count identical
// API events. Wrappers are installed only for instrumented passes and removed afterwards, so
// timing passes run against the unmodified browser APIs. TF.js looks methods up on the prototype
// at call time, so installing/removing takes effect immediately for existing objects.
//
// Everything counted here is read from the API call arguments or the objects themselves
// (buffer.size, texture dimensions); nothing is inferred.

const originals = [];
let c = null;            // counters for the current window (reset per run)
let groups = null;       // dispatches per queue.submit, in order
let pendingDispatches = 0;

// Submission tracking (profile phases only): per queue.submit, the CPU time of the call, its
// dispatch count, how many earlier submissions were still unfinished, and when the queue reported
// it done (onSubmittedWorkDone, awaited without blocking). Completion times include queue
// scheduling, so group durations derived from them are bounded, not GPU-exact.
let submitLog = null;
let inFlight = 0;

// Logical GPU memory owned by the page, tracked from creation to destroy/delete.
const liveBuffers = new Map();   // GPUBuffer -> bytes
const liveTextures = new Map();  // WebGLTexture -> bytes
let liveBytes = 0;
let peakLiveBytes = 0;

function emptyCounters() {
  return {
    webgpu_submits: 0,
    webgpu_command_buffers: 0,
    webgpu_dispatches: 0,
    webgpu_compute_passes: 0,
    webgpu_write_buffer_calls: 0,
    webgpu_write_buffer_bytes: 0,
    webgpu_copy_b2b_calls: 0,
    webgpu_copy_b2b_bytes: 0,
    webgpu_map_read_calls: 0,
    webgpu_map_read_bytes: 0,
    webgpu_map_read_wait_ms: 0,
    webgpu_buffers_created: 0,
    webgpu_buffer_bytes_created: 0,
    webgpu_buffers_destroyed: 0,
    webgpu_uniform_buffers_created: 0,
    webgpu_mapped_at_creation_bytes: 0,
    webgpu_bind_groups_created: 0,
    webgpu_pipelines_created: 0,
    webgpu_pipelines_created_async: 0,
    webgpu_query_resolves: 0,
    webgl_draws: 0,
    webgl_tex_uploads: 0,
    webgl_tex_upload_bytes: 0,
    webgl_read_pixels_calls: 0,
    webgl_read_pixels_bytes: 0,
    webgl_get_buffer_sub_data_calls: 0,
    webgl_get_buffer_sub_data_bytes: 0,
    webgl_programs_linked: 0,
    webgl_fences: 0,
  };
}

function wrap(proto, name, before, after) {
  if (!proto || typeof proto[name] !== "function") return;
  const original = proto[name];
  proto[name] = function (...args) {
    const ctx = before ? before(this, args) : undefined;
    const ret = original.apply(this, args);
    return after ? after(this, args, ret, ctx) : ret;
  };
  originals.push([proto, name, original]);
}

function trackAlloc(map, key, bytes) {
  map.set(key, bytes);
  liveBytes += bytes;
  if (liveBytes > peakLiveBytes) peakLiveBytes = liveBytes;
}

function trackFree(map, key) {
  const bytes = map.get(key);
  if (bytes === undefined) return;
  map.delete(key);
  liveBytes -= bytes;
}

// Bytes per texel for the (internalformat) values TF.js's WebGL backend uses.
function bytesPerTexel(gl, internalFormat) {
  switch (internalFormat) {
    case gl.RGBA32F: return 16;
    case gl.RGBA16F: return 8;
    case gl.R32F: return 4;
    case gl.R16F: return 2;
    case gl.RGBA8:
    case gl.RGBA: return 4;
    case gl.RG32F: return 8;
    default: return null;
  }
}

function installWebGPU() {
  if (typeof GPUQueue === "undefined") return;
  wrap(GPUQueue.prototype, "submit", null, (q, [buffers], ret) => {
    c.webgpu_submits++;
    c.webgpu_command_buffers += buffers?.length ?? 0;
    groups.push(pendingDispatches);
    if (submitLog) {
      const rec = { t_submit: performance.now(), dispatches: pendingDispatches, in_flight_before: inFlight, t_done: null };
      submitLog.push(rec);
      inFlight++;
      q.onSubmittedWorkDone().then(() => { rec.t_done = performance.now(); inFlight--; });
    }
    pendingDispatches = 0;
    return ret;
  });
  wrap(GPUQueue.prototype, "writeBuffer", (_q, [, , data, dataOffset = 0, size]) => {
    c.webgpu_write_buffer_calls++;
    const elemBytes = data?.BYTES_PER_ELEMENT ?? 1;
    const total = data?.byteLength ?? 0;
    c.webgpu_write_buffer_bytes += size !== undefined ? size * elemBytes : total - dataOffset * elemBytes;
  });
  wrap(GPUCommandEncoder.prototype, "copyBufferToBuffer", (_e, args) => {
    c.webgpu_copy_b2b_calls++;
    // (src, srcOffset, dst, dstOffset, size) or the newer (src, dst, size?) overload.
    c.webgpu_copy_b2b_bytes += typeof args[1] === "number" ? args[4] ?? 0 : args[2] ?? args[0].size;
  });
  wrap(GPUCommandEncoder.prototype, "beginComputePass", () => { c.webgpu_compute_passes++; });
  wrap(GPUComputePassEncoder.prototype, "dispatchWorkgroups", () => { c.webgpu_dispatches++; pendingDispatches++; });
  wrap(GPUComputePassEncoder.prototype, "dispatchWorkgroupsIndirect", () => { c.webgpu_dispatches++; pendingDispatches++; });
  wrap(GPUBuffer.prototype, "mapAsync", (buffer, [mode, offset = 0, size]) => {
    if (!(mode & GPUMapMode.READ)) return null;
    c.webgpu_map_read_calls++;
    c.webgpu_map_read_bytes += size ?? buffer.size - offset;
    return performance.now();
  }, (_b, _a, promise, t0) => {
    if (t0 === null) return promise;
    const counters = c;
    return promise.then(v => { counters.webgpu_map_read_wait_ms += performance.now() - t0; return v; });
  });
  wrap(GPUDevice.prototype, "createBuffer", null, (_d, [desc], buffer) => {
    c.webgpu_buffers_created++;
    c.webgpu_buffer_bytes_created += desc.size;
    if (desc.usage & GPUBufferUsage.UNIFORM) c.webgpu_uniform_buffers_created++;
    // TF.js uploads tensor data by creating buffers mapped at creation (or MAP_WRITE staging
    // buffers that are mapped at creation), not with writeBuffer.
    if (desc.mappedAtCreation) c.webgpu_mapped_at_creation_bytes += desc.size;
    trackAlloc(liveBuffers, buffer, desc.size);
    return buffer;
  });
  wrap(GPUDevice.prototype, "createBindGroup", () => { c.webgpu_bind_groups_created++; });
  wrap(GPUDevice.prototype, "createComputePipeline", () => { c.webgpu_pipelines_created++; });
  wrap(GPUDevice.prototype, "createComputePipelineAsync", () => { c.webgpu_pipelines_created_async++; });
  wrap(GPUCommandEncoder.prototype, "resolveQuerySet", () => { c.webgpu_query_resolves++; });
  wrap(GPUBuffer.prototype, "destroy", buffer => {
    if (liveBuffers.has(buffer)) c.webgpu_buffers_destroyed++;
    trackFree(liveBuffers, buffer);
  });
}

// WebGL per-draw GPU timing (EXT_disjoint_timer_query_webgl2), independent of TF.js's own
// tf.profile timers so the two sources can be cross-checked.
let drawTimers = null;               // null = off; array of pending {gl, ext, query, program, drawIndex}
let drawIndex = 0;
const timerExt = new WeakMap();      // gl -> extension or null
const programIds = new WeakMap();    // WebGLProgram -> small integer id
let nextProgramId = 0;

function timerExtension(gl) {
  if (!timerExt.has(gl)) timerExt.set(gl, gl.getExtension("EXT_disjoint_timer_query_webgl2"));
  return timerExt.get(gl);
}

function timedDraw(gl, original, args) {
  c.webgl_draws++;
  const ext = drawTimers && gl instanceof WebGL2RenderingContext ? timerExtension(gl) : null;
  if (!ext) return original.apply(gl, args);
  const program = gl.getParameter(gl.CURRENT_PROGRAM);
  if (program && !programIds.has(program)) programIds.set(program, nextProgramId++);
  const query = gl.createQuery();
  gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
  const ret = original.apply(gl, args);
  gl.endQuery(ext.TIME_ELAPSED_EXT);
  drawTimers.push({ gl, ext, query, program: program ? programIds.get(program) : null, drawIndex: drawIndex++ });
  return ret;
}

export function startDrawTimers() {
  drawTimers = [];
  drawIndex = 0;
}

/** Wait for every pending draw query and return [{program, drawIndex, gpu_ms | null, disjoint}]. */
export async function collectDrawTimers(timeoutMs = 10000) {
  const pending = drawTimers ?? [];
  drawTimers = null;
  const deadline = performance.now() + timeoutMs;
  const out = [];
  for (const d of pending) {
    const { gl, ext, query } = d;
    while (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE) && performance.now() < deadline) {
      await new Promise(r => setTimeout(r, 1));
    }
    const available = gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE);
    const disjoint = !!gl.getParameter(ext.GPU_DISJOINT_EXT);
    out.push({
      program: d.program, drawIndex: d.drawIndex, disjoint,
      gpu_ms: available && !disjoint ? gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6 : null,
    });
    gl.deleteQuery(query);
  }
  return out;
}

function installWebGL() {
  for (const ctx of [globalThis.WebGL2RenderingContext, globalThis.WebGLRenderingContext]) {
    if (!ctx) continue;
    const p = ctx.prototype;
    for (const name of ["drawArrays", "drawElements"]) {
      const original = p[name];
      p[name] = function (...args) { return timedDraw(this, original, args); };
      originals.push([p, name, original]);
    }
    wrap(p, "readPixels", (gl, [, , w, h, format, type]) => {
      c.webgl_read_pixels_calls++;
      const channels = format === gl.RGBA ? 4 : format === gl.RED ? 1 : format === gl.RG ? 2 : 4;
      const bytes = type === gl.FLOAT ? 4 : type === gl.HALF_FLOAT ? 2 : 1;
      c.webgl_read_pixels_bytes += w * h * channels * bytes;
    });
    // texImage2D(target, level, internalformat, width, height, border, format, type, pixels)
    wrap(p, "texImage2D", (gl, args) => {
      if (args.length < 9) return;
      const [, level, internalFormat, w, h, , , , pixels] = args;
      const bpt = bytesPerTexel(gl, internalFormat);
      if (pixels) {
        c.webgl_tex_uploads++;
        c.webgl_tex_upload_bytes += pixels.byteLength ?? 0;
      }
      const tex = gl.getParameter(gl.TEXTURE_BINDING_2D);
      if (level === 0 && tex && bpt !== null) {
        trackFree(liveTextures, tex);
        trackAlloc(liveTextures, tex, w * h * bpt);
      }
    });
    wrap(p, "texSubImage2D", (_gl, args) => {
      const pixels = args[args.length - 1];
      c.webgl_tex_uploads++;
      c.webgl_tex_upload_bytes += pixels?.byteLength ?? 0;
    });
    wrap(p, "deleteTexture", (_gl, [tex]) => { trackFree(liveTextures, tex); });
    wrap(p, "linkProgram", () => { c.webgl_programs_linked++; });
    wrap(p, "fenceSync", () => { c.webgl_fences++; });
    if (typeof p.getBufferSubData === "function") {
      // TF.js WebGL2 async readback: readPixels into a PIXEL_PACK_BUFFER, then getBufferSubData.
      wrap(p, "getBufferSubData", (_gl, [, , dst, , length]) => {
        c.webgl_get_buffer_sub_data_calls++;
        const bpe = dst?.BYTES_PER_ELEMENT ?? 1;
        c.webgl_get_buffer_sub_data_bytes += length ? length * bpe : dst?.byteLength ?? 0;
      });
    }
  }
}

export function installCounters() {
  if (originals.length) throw new Error("counters already installed");
  submitLog = null;
  inFlight = 0;
  drawIndex = 0;
  c = emptyCounters();
  groups = [];
  pendingDispatches = 0;
  installWebGPU();
  installWebGL();
}

// CPU time spent inside WebGL calls, per method (profile mode, WebGL phase "profile_gltime").
// Every method on the WebGL prototypes is wrapped with performance.now(); only the outermost
// call is timed, so GL calls made by the counter wrappers above aren't counted twice. The time
// between calls (wall - total) is JavaScript: TF.js's own work.
const glTimingOriginals = [];
let glTime = null;   // method name -> {calls, ms}
let glDepth = 0;

function restore(list) {
  while (list.length) {
    const [proto, name, original] = list.pop();
    proto[name] = original;
  }
}

export function startGlTiming() {
  glTime = new Map();
  if (glTimingOriginals.length) return;
  for (const ctx of [globalThis.WebGL2RenderingContext, globalThis.WebGLRenderingContext]) {
    if (!ctx) continue;
    const p = ctx.prototype;
    for (const name of Object.getOwnPropertyNames(p)) {
      // Accessors (canvas, drawingBufferWidth, ...) throw when read on the prototype.
      const desc = Object.getOwnPropertyDescriptor(p, name);
      if (name === "constructor" || typeof desc.value !== "function") continue;
      const original = desc.value;
      p[name] = function (...args) {
        if (glDepth > 0 || !glTime) return original.apply(this, args);
        glDepth++;
        const t0 = performance.now();
        try {
          return original.apply(this, args);
        } finally {
          const dt = performance.now() - t0;
          glDepth--;
          const e = glTime.get(name);
          if (e) { e.calls++; e.ms += dt; } else glTime.set(name, { calls: 1, ms: dt });
        }
      };
      glTimingOriginals.push([p, name, original]);
    }
  }
}

/** Per-method {calls, ms} since the last read (or start). */
export function readGlTiming() {
  const out = Object.fromEntries([...(glTime ?? new Map())].map(([k, v]) => [k, { ...v }]));
  glTime = new Map();
  return out;
}

export function stopGlTiming() {
  restore(glTimingOriginals);
  glTime = null;
}

// readPixels split (profile mode, WebGL phase "profile_readsplit"): before each readPixels into
// client memory, a 1x1 readPixels of the same framebuffer/format/type forces every queued draw
// to finish without moving the payload ("drain"); the real call that follows then measures the
// transfer. Reads into a PIXEL_PACK_BUFFER (offset argument) aren't split.
const readSplitOriginals = [];
let readSplit = null;

function scratchFor(gl, type) {
  if (type === gl.FLOAT) return new Float32Array(4);
  if (type === gl.HALF_FLOAT) return new Uint16Array(4);
  if (type === gl.UNSIGNED_INT || type === gl.INT) return new Uint32Array(4);
  return new Uint8Array(4);
}

export function startReadSplit() {
  readSplit = { calls: 0, split_calls: 0, drain_ms: 0, transfer_ms: 0, unsplit_ms: 0 };
  if (readSplitOriginals.length) return;
  for (const ctx of [globalThis.WebGL2RenderingContext, globalThis.WebGLRenderingContext]) {
    if (!ctx) continue;
    const p = ctx.prototype;
    const original = p.readPixels;
    p.readPixels = function (...args) {
      if (!readSplit) return original.apply(this, args);
      readSplit.calls++;
      const [x, y, w, h, format, type, dst] = args;
      if (typeof dst !== "object" || dst === null || w * h <= 1) {
        const t0 = performance.now();
        const ret = original.apply(this, args);
        readSplit.unsplit_ms += performance.now() - t0;
        return ret;
      }
      const t0 = performance.now();
      original.call(this, x, y, 1, 1, format, type, scratchFor(this, type));
      const t1 = performance.now();
      const ret = original.apply(this, args);
      const t2 = performance.now();
      readSplit.split_calls++;
      readSplit.drain_ms += t1 - t0;
      readSplit.transfer_ms += t2 - t1;
      return ret;
    };
    readSplitOriginals.push([p, "readPixels", original]);
  }
}

export function readReadSplit() {
  const out = { ...readSplit };
  readSplit = { calls: 0, split_calls: 0, drain_ms: 0, transfer_ms: 0, unsplit_ms: 0 };
  return out;
}

export function stopReadSplit() {
  restore(readSplitOriginals);
  readSplit = null;
}

export function uninstallCounters() {
  while (originals.length) {
    const [proto, name, original] = originals.pop();
    proto[name] = original;
  }
}

export function resetCounters() {
  c = emptyCounters();
  groups = [];
  pendingDispatches = 0;
}

export function readCounters() {
  return { ...c };
}

export function startSubmitTracking() {
  submitLog = [];
}

/**
 * Stop tracking and return the submissions since startSubmitTracking(). Waits (bounded) until every
 * tracked submission has reported completion.
 */
export async function readSubmitTracking(timeoutMs = 10000) {
  const log = submitLog ?? [];
  submitLog = null;
  const deadline = performance.now() + timeoutMs;
  while (log.some(r => r.t_done === null) && performance.now() < deadline) {
    await new Promise(r => setTimeout(r, 1));
  }
  return log;
}

/** Dispatches per queue.submit since the last reset (execution-group sizes). */
export function readGroups() {
  return [...groups];
}

export function readMemory() {
  return {
    live_bytes: liveBytes,
    peak_live_bytes: peakLiveBytes,
    live_buffers: liveBuffers.size,
    live_textures: liveTextures.size,
  };
}

/** Start a new peak-tracking interval from the current live level. */
export function resetPeak() {
  peakLiveBytes = liveBytes;
}

/**
 * Decide whether the counted API activity proves the requested backend executed the run.
 * WebGPU must submit GPU work; WebGL must issue draws; WASM must do neither.
 */
export function verifyBackend(backend, counters) {
  const gpuWork = counters.webgpu_submits > 0 || counters.webgpu_dispatches > 0;
  const glWork = counters.webgl_draws > 0;
  const summary = `submits=${counters.webgpu_submits} dispatches=${counters.webgpu_dispatches} draws=${counters.webgl_draws}`;
  switch (backend) {
    case "webgpu":
      return gpuWork && !glWork
        ? { verified: true, detail: `${counters.webgpu_dispatches} dispatches in ${counters.webgpu_submits} submits` }
        : { verified: false, detail: `expected WebGPU work only; ${summary}` };
    case "webgl":
      return glWork && !gpuWork
        ? { verified: true, detail: `${counters.webgl_draws} draws` }
        : { verified: false, detail: `expected WebGL draws only; ${summary}` };
    case "wasm":
    case "cpu":
      return !gpuWork && !glWork
        ? { verified: true, detail: "no GPU API activity" }
        : { verified: false, detail: `expected no GPU activity; ${summary}` };
    default:
      return { verified: false, detail: `unknown backend ${backend}` };
  }
}
