// Page entry point: manual UI plus the `window.__bench` API that run_benchmark.py drives.
import { runExperiment, probeEnv, setLogger, INFER_MODE_NAMES, TRAIN_MODE_NAMES } from "./bench.js";
import { tf, applyConfig } from "./tfjs-setup.js";
import { readbackMicro } from "./readback_micro.js";
import { PROBES } from "./probes.js";

const $ = id => document.getElementById(id);

// Manual-UI configs only; run_benchmark.py's CONFIGS is the full registry.
const UI_CONFIGS = {
  "wasm-8t": { backend: "wasm", threads: 8 },
  "webgl": { backend: "webgl" },
  "webgpu": { backend: "webgpu" },
  "cpu": { backend: "cpu" },
};

function logStatus(msg) {
  $("status").textContent += msg + "\n";
  console.log(msg);
}
setLogger(logStatus);

async function loadSpec() {
  const res = await fetch("/spec/models.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load spec/models.json (${res.status})`);
  return res.json();
}

function median(a) {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

async function main() {
  const spec = await loadSpec();
  const models = spec.models.map(m => ({ ...m, optimizer: spec.optimizer }));
  for (const m of models) $("modelSelect").add(new Option(`${m.name} (${m.family}, ${m.params.toLocaleString()} params)`, m.name));
  for (const c of Object.keys(UI_CONFIGS)) $("configSelect").add(new Option(c, c));
  for (const m of [...INFER_MODE_NAMES, ...TRAIN_MODE_NAMES]) $("modeSelect").add(new Option(m, m));

  // TF.js env flags and the WASM thread count are fixed at backend init: reload between configs.
  $("runBtn").onclick = async () => {
    $("runBtn").disabled = true;
    const label = $("configSelect").value;
    const r = await runExperiment({
      ...UI_CONFIGS[label], configLabel: label, mode: $("modeSelect").value,
      model: models.find(m => m.name === $("modelSelect").value),
      warmup: 5, measure: 30, trainWarmup: 5, trainMeasure: 20, inferWarmup: 3, inferMeasure: 10, seed: 1,
    });
    const timed = r.runs.filter(x => ["measure", "train_measure"].includes(x.phase)).map(x => x.dur_ms);
    const tr = document.createElement("tr");
    for (const c of [r.model, r.config_label, r.mode, r.status, r.verification?.detail ?? r.failure?.message ?? "",
      r.runs[0]?.dur_ms?.toFixed(2) ?? "—", median(timed).toFixed(2), String(timed.length)]) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    $("resultsBody").appendChild(tr);
    window.__lastResult = r;
    $("runBtn").disabled = false;
  };

  const micro = async cfg => {
    if (cfg.backend) await applyConfig({ backend: cfg.backend });
    return readbackMicro(cfg);
  };
  // `tf` is exposed for diagnostics scripts only.
  window.__bench = { ready: true, error: null, probeEnv, runExperiment, spec, models, tf, readbackMicro: micro, probes: PROBES };
}

main().catch(e => {
  logStatus(`Initialisation failed: ${e?.message ?? e}`);
  window.__bench = { ready: false, error: String(e?.message ?? e) };
});
