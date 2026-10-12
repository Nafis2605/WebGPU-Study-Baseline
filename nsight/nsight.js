// Nsight capture target (port of the sibling study's nsight.js). Loads one spec model on one config and
// runs inference or training steps back to back (closed loop, as the study's timing stages), so a trace
// collected after warm-up sees steady-state work.
// Query parameters:
//   model=<spec name>&config=<label>&cfg=<config JSON: backend, flags, threads, ...>
//   task=infer   inference at batch `batch` (default 1), trained weights (the inference stages)
//   task=train   custom-loop training steps at the spec batch, initial weights (the train stage)
//   wait=1       after loading, wait for window.__nsight.go() before the first run, so collection can
//                start before warm-up (the study's phase)
// window.__nsight.stats(from) returns per-run epoch timestamps (performance.timeOrigin + now), so the
// trace's wall-clock axis can be segmented into inferences / steps.
import { applyConfig } from "../src/tfjs-setup.js";
import { buildModel, weightSlots } from "../src/models/build.js";
import { loadData } from "../src/data.js";
import { fetchWeights, assignWeights } from "../src/weights.js";
import { makeLayersRunner } from "../src/layers-runner.js";
import { makeTrainRunner } from "../src/train-runner.js";

const q = new URLSearchParams(location.search);
const status = document.getElementById("status");
const runs = [];
const epoch = () => performance.timeOrigin + performance.now();

async function main() {
  const cfg = JSON.parse(q.get("cfg") ?? '{"backend":"webgpu"}');
  const task = q.get("task") ?? "infer";
  const wait = q.get("wait") === "1";
  const spec = await (await fetch("/spec/models.json")).json();
  const m = { ...spec.models.find(x => x.name === q.get("model")), optimizer: spec.optimizer };
  await applyConfig(cfg);
  const data = await loadData();
  const w = await fetchWeights(m.name, task === "train" ? "init" : "trained");
  const model = buildModel(m);
  const slots = weightSlots(model, m);
  assignWeights(slots, w);
  const runner = task === "train"
    ? makeTrainRunner(model, slots, data, m, cfg, Number(q.get("batch") ?? m.batch))
    : makeLayersRunner(model, data, cfg, Number(q.get("batch") ?? 1));
  const t0 = performance.now();
  let go;
  const started = new Promise(r => { go = r; });
  window.__nsight = {
    ready: true, mode: "closed", task, go: () => go(),
    runs: () => runs.length,
    stats: (from = 0) => ({ mode: "closed", task, time_origin_ms: performance.timeOrigin, from, total: runs.length,
                            runs: runs.slice(from) }),
  };
  if (wait) await started;
  window.__nsight.first_run_epoch_ms = epoch();
  for (;;) {
    const s = epoch();
    await runner.run();
    runs.push({ start_epoch_ms: s, end_epoch_ms: epoch() });
    if (runs.length % 20 === 0) {
      status.textContent = `${m.name} / ${q.get("config")} / ${task}: ${runs.length} runs, ` +
        `${((performance.now() - t0) / runs.length).toFixed(2)} ms/run`;
      await new Promise(r => setTimeout(r, 0));   // let CDP read stats
    }
  }
}

main().catch(e => { status.textContent = `error: ${e?.message ?? e}`; window.__nsight = { ready: false, error: String(e) }; });
