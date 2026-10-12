// Pillar 2 (inference) modes: the sibling study's MODES (bench.js), unchanged in what they measure,
// driving the code-defined models through src/layers-runner.js. Inference launches load the
// study's trained weights. The batch dimension is the models' only free input dimension, so the
// shapes mode sweeps it.
import { makeLayersRunner } from "./layers-runner.js";
import { instrumented, runnerRuns, capOf, shapeSequence, encodeTensor, pageMemory } from "./common.js";
import { profileMode } from "./profile.js";
import { loadMode } from "./load.js";
import { responsivenessMode } from "./responsiveness.js";
import { memorySnapshot } from "./memory.js";

export const DEFAULT_SHAPE_VALUES = [1, 2, 4, 8, 16, 32, 64];
export const DEFAULT_BATCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

export function inferRunner(model, ctx, cfg, B) {
  const r = makeLayersRunner(model, ctx.data, cfg, B);
  return cfg.instrumentRuns ? instrumented(r) : r;
}

const measureCap = cfg => capOf(cfg.maxMeasureS, cfg.minMeasure, cfg.measure, cfg.minMeasureS);

export const INFER_MODES = {
  /** Closed-loop latency at the base batch (cfg.inferBatch, default 1). */
  async latency(model, ctx, cfg, result) {
    const runner = inferRunner(model, ctx, cfg, cfg.inferBatch ?? 1);
    await runnerRuns(runner, "warmup", cfg.warmup, result.runs);
    const m = await runnerRuns(runner, "measure", cfg.measure, result.runs, {}, measureCap(cfg));
    if (m.capped) result.capped.push({ phase: "measure", runs: m.runs });
    return runner;
  },

  /**
   * Shape reuse: after warm-up at the base batch, replay a seeded sequence of batch sizes, each
   * `shapeRepeats` times in a row; every call records whether its shape was seen before.
   */
  async shapes(model, ctx, cfg, result) {
    const baseB = cfg.inferBatch ?? 1;
    const base = inferRunner(model, ctx, cfg, baseB);
    await runnerRuns(base, "warmup", cfg.warmup, result.runs);
    const values = cfg.shapeValues ?? DEFAULT_SHAPE_VALUES;
    const shapeSeed = cfg.seed + (cfg.trial ?? 0);
    const sequence = shapeSequence(values, cfg.shapeRounds ?? 3, shapeSeed);
    const repeats = cfg.shapeRepeats ?? 3;
    Object.assign(result.config, { shape_seed: shapeSeed, shape_sequence: sequence, shape_repeats: repeats, shape_values: values });
    result.shape_memory = [{ block: -1, shape_value: baseB, first_seen: false, ...memorySnapshot(false) }];
    const seen = new Set([baseB]);
    let i = 0;
    for (const [block, value] of sequence.entries()) {
      const runner = inferRunner(model, ctx, cfg, value);
      const firstSeen = !seen.has(value);
      seen.add(value);
      for (let pos = 0; pos < repeats; pos++) {
        result.runs.push({ phase: "shape", i: i++, block, block_pos: pos, shape_value: value,
          shape: `${value}x32x32x3`, first_seen: firstSeen && pos === 0, ...(await runner.run()) });
      }
      runner.dispose();
      result.shape_memory.push({ block, shape_value: value, first_seen: firstSeen, ...memorySnapshot(false) });
    }
    return base;
  },

  /** Batch scaling: per batch value (ascending) warm up, then measure. */
  async batch(model, ctx, cfg, result) {
    const values = [...(cfg.batchValues ?? DEFAULT_BATCH_VALUES)].sort((a, b) => a - b);
    result.config.batch_values = values;
    let base = null;
    for (const value of values) {
      const runner = inferRunner(model, ctx, cfg, value);
      await runnerRuns(runner, "batch_warmup", cfg.warmup, result.runs, { batch: value });
      const m = await runnerRuns(runner, "batch", cfg.measure, result.runs, { batch: value }, measureCap(cfg));
      if (m.capped) result.capped.push({ phase: "batch", batch: value, runs: m.runs });
      if (value === (cfg.inferBatch ?? 1)) base = runner;
      else runner.dispose();
    }
    return base ?? inferRunner(model, ctx, cfg, cfg.inferBatch ?? 1);
  },

  /** Instrumented pass (profile.js): counters, tf.profile kernels, tf.time, WebGL draw timers, memory. */
  async profile(model, ctx, cfg, result) {
    const runner = makeLayersRunner(model, ctx.data, cfg, cfg.inferBatch ?? 1);
    await profileMode(runner, cfg, result, cfg.backend);
    result.page_memory.after_runs = await pageMemory();
    return runner;
  },

  /** One untimed inference after warm-up; exact input/output bytes go to io/ for comparison. */
  async correctness(model, ctx, cfg, result) {
    const runner = makeLayersRunner(model, ctx.data, cfg, cfg.correctnessBatch ?? 32);
    await runnerRuns(runner, "warmup", cfg.warmup, result.runs);
    const outputs = await runner.outputs();
    result.io = {
      inputs: runner.inputData().map(i => encodeTensor(i.name, i.dtype, i.dims, i.data)),
      outputs: outputs.map(o => encodeTensor(o.name, o.dtype, o.shape, o.data)),
    };
    return runner;
  },

  /** Open-loop Poisson load with deadlines and a bounded queue (load.js). */
  async load(model, ctx, cfg, result) {
    const runner = inferRunner(model, ctx, cfg, cfg.inferBatch ?? 1);
    await loadMode(runner, cfg, result);
    return runner;
  },

  /** Main-thread availability while inference runs (responsiveness.js). */
  async responsiveness(model, ctx, cfg, result) {
    const runner = inferRunner(model, ctx, cfg, cfg.inferBatch ?? 1);
    await responsivenessMode(runner, cfg, result);
    return runner;
  },
};
