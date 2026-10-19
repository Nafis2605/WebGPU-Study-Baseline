"""
Preflight: will the full study complete on the locked environment, and are its inputs what they claim?
(Structure ported from the sibling TF.js study's preflight.py.)

1. Static gates (no browser): environment lock (full: incl. every data / weight file hash, clean committed
   code); both model builders reproduce the spec (scripts/check_spec.py); the page's and the native worker's
   sketch PRNG agree bit for bit (scripts/check_prng.py); exactly one @tensorflow/tfjs-core is installed;
   the data manifest is the CIFAR-10 release (not a placeholder); every model has init and trained weights.
2. The real pipeline (run_study.py: lock checks, GPU idle + boost guards, every stage, every model x config
   pair, native stages, GPU traces, all analyses) with minimal run counts into results/<device>/chrome-<ver>/_dev/preflight/.
   The infer_latency / infer_native / train / native_train smoke keep the full warm-up rule and measure 20
   records where they fit in 10 s, so warm-up sufficiency is checked (analysis/warmup.py).
3. Report (preflight_report.json / .md): per stage launches, failures by category; failures that are not
   known TF.js coverage gaps are BLOCKERS (incl. status leak_detected / rng_used / nonfinite_loss /
   backend_not_verified / strict_violation); low-clock re-runs; warm-up sufficiency; analysis tracebacks;
   duration projection for the full study from the measured smoke timings. Verdict: GO only without blockers.

Usage (after `python -m benchkit.envlock freeze` on a committed tree, with `npm run bench:serve` running):
    python preflight.py
    python preflight.py --report-only    # rebuild the report from a finished preflight
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchkit import envlock, layout  # noqa: E402
from benchkit.paths import DATA_DIR, SPEC, WEIGHTS_DIR  # noqa: E402
import run_study  # noqa: E402

OUT = layout.run_dir("chrome", "_dev/preflight")
# Known, reproducible TF.js coverage gaps (not harness failures): (family, backend, pillar).
from run_benchmark import KNOWN_GAPS  # noqa: E402  (family or model name, backend, pillar)
TRAIN_MODES = {"train", "train_pipelined", "train_profile", "train_correctness", "train_batch", "train_long",
               "train_fit", "train_phases", "train_responsiveness", "lifecycle"}

S_INFER = ["--trials", "1", "--measure", "5", "--min-measure", "5", "--max-measure-s", "10", "--min-measure-s", "0"]
# warm-up gate stages: full warm-up rule and 20 measured records (analysis/warmup.py MIN_MEASURED) where they
# fit in the 10 s cap; slower pairs stop at 5 and are reported "not assessable" (seconds-long steps)
S_INFER_GATE = ["--trials", "1", "--measure", "20", "--min-measure", "5", "--max-measure-s", "10", "--min-measure-s", "0"]
S_TRAIN = ["--trials", "1", "--train-measure", "10", "--train-min-measure", "5", "--train-max-s", "10",
           "--min-measure-s", "0", "--infer-warmup", "2", "--infer-measure", "3", "--infer-max-s", "5"]
S_TRAIN_GATE = ["--trials", "1", "--train-measure", "20", "--train-min-measure", "5", "--train-max-s", "10",
                "--min-measure-s", "0", "--infer-warmup", "2", "--infer-measure", "3", "--infer-max-s", "5"]
S_NATIVE = ["--trials", "1"]
SMOKE = {
    # full warm-up counts kept (warm-up sufficiency check), fewer measured records
    "infer_latency": S_INFER_GATE, "train": S_TRAIN_GATE,
    "infer_native": S_INFER_GATE, "infer_native_why": [*S_INFER, "--warmup", "2"],
    "infer_correctness": ["--warmup", "0"], "infer_native_correctness": ["--warmup", "0"],
    "infer_batch": [*S_INFER, "--warmup", "1", "--measure", "2", "--min-measure", "2", "--batch-values", "1", "32", "256"],
    "infer_native_batch": [*S_INFER, "--warmup", "1", "--measure", "2", "--min-measure", "2", "--batch-values", "1", "32", "256"],
    "infer_shapes": ["--trials", "1", "--warmup", "1", "--shape-rounds", "1", "--shape-repeats", "1"],
    "infer_why": [*S_INFER, "--warmup", "2"], "infer_instrumented": [*S_INFER, "--warmup", "2"],
    "infer_profile": ["--trials", "1", "--warmup", "1", "--measure", "2", "--max-measure-s", "10"],
    "infer_responsiveness": ["--trials", "1", "--warmup", "1", "--page-baseline-s", "1", "--load-s", "2"],
    "infer_load": ["--trials", "1", "--warmup", "1", "--rates", "5", "50", "--load-s", "2", "--drain-s", "2"],
    "readback_micro": ["--trials", "1", "--reps", "2", "--sizes-mib", "1", "64"],
    "warmup_precompile": [*S_INFER, "--warmup", "2"], "warmup_diskcache": [*S_INFER, "--warmup", "2"],
    "warmup_precompile_train": [*S_TRAIN, "--train-warmup", "3"],
    "warmup_lifecycle_if": [*S_TRAIN, "--train-warmup", "2"], "warmup_lifecycle_tf": [*S_TRAIN, "--train-warmup", "2"],
    "native_train": S_TRAIN_GATE, "native_train_why": [*S_TRAIN, "--train-warmup", "3"],
    "train_instrumented": [*S_TRAIN, "--train-warmup", "2", "--train-measure", "2"],
    "train_profile": ["--trials", "1", "--train-warmup", "1", "--profile-steps", "1", "--train-max-s", "10"],
    "train_coverage": [],
    "train_correctness": ["--correctness-steps", "3"], "train_correctness_cpu": ["--correctness-steps", "3"],
    "native_train_correctness_ref": ["--correctness-steps", "3"], "native_train_correctness_keras": ["--correctness-steps", "3"],
    "native_train_correctness_gpu": ["--correctness-steps", "3"],
    "train_long": ["--trials", "1", "--long-epochs", "1"], "native_train_long": ["--trials", "1", "--long-epochs", "1"],
    "train_batch": [*S_TRAIN, "--batch-warmup", "1", "--train-measure", "2", "--train-batch-values", "16", "256"],
    "native_train_batch": [*S_TRAIN, "--batch-warmup", "1", "--train-measure", "2", "--train-batch-values", "16", "256"],
    "train_why": [*S_TRAIN, "--train-warmup", "3"], "train_sync_io": [*S_TRAIN, "--train-warmup", "3"],
    "train_pipelined": [*S_TRAIN, "--train-warmup", "3"],
    "train_phases": [*S_TRAIN, "--train-warmup", "2", "--train-measure", "3"],
    "native_train_phases": [*S_TRAIN, "--train-warmup", "2", "--train-measure", "3"],
    "train_fit": ["--trials", "1", "--fit-epochs", "1"], "native_train_fit": ["--trials", "1", "--fit-epochs", "1"],
    "train_responsiveness": ["--trials", "1", "--train-warmup", "1", "--page-baseline-s", "1", "--load-s", "2"],
    "gpu_trace": ["--repeats", "1", "--models", "D3", "L3", "T3"],
}
BAD_STATUS_MEANS = {"leak_detected": "a step / run leaked tensors", "rng_used": "Math.random called during measured work",
                    "nonfinite_loss": "non-finite loss", "backend_not_verified": "API counters don't match the backend",
                    "strict_violation": "strict config forwarded kernels to the CPU"}


def _arg(spec, flag, default):
    idx = [i for i, a in enumerate(spec) if a == flag]
    return float(spec[idx[-1] + 1]) if idx else default


def static_gates():
    gates = []

    def gate(name, ok, detail=""):
        gates.append({"gate": name, "ok": bool(ok), "detail": detail})

    for script in ("scripts/check_spec.py", "scripts/check_prng.py"):
        p = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True, errors="replace")
        gate(script, p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:])
    p = subprocess.run(["npm.cmd", "ls", "@tensorflow/tfjs-core", "--all", "--parseable"], cwd=ROOT,
                       capture_output=True, text=True, errors="replace")
    copies = {l.strip() for l in p.stdout.splitlines() if l.strip().lower().endswith("tfjs-core")}
    gate("single @tensorflow/tfjs-core", len(copies) == 1, f"{len(copies)} install path(s)")
    man = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    gate("data is the CIFAR-10 release", man.get("dataset", "").startswith("CIFAR-10") and
         man.get("archive_md5") == "c32a1d4ab5d03f1284b67883e8d87530", man.get("dataset"))
    wman = json.loads((WEIGHTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in json.loads(SPEC.read_text(encoding="utf-8"))["models"]]
    missing = [f"{n}:{k}" for n in names for k in ("init", "trained") if k not in wman.get(n, {})]
    gate("init + trained weights for every model", not missing, ", ".join(missing[:10]))
    return gates


def stage_dir(stage):
    f = OUT / stage / "stage_done.json"
    return Path(json.loads(f.read_text(encoding="utf-8"))["result_dir"]) if f.exists() else None


def known_gap(row):
    pillar = "train" if row.get("mode") in TRAIN_MODES else "infer"
    return ((row.get("family"), row.get("ep"), pillar) in KNOWN_GAPS
            or (row.get("model"), row.get("ep"), pillar) in KNOWN_GAPS)


def stage_report(stage):
    d = stage_dir(stage)
    rep = {"done": d is not None}
    if d is None or not (d / "sessions.csv").exists():
        return rep
    s = pd.read_csv(d / "sessions.csv")
    rep["launches"] = int(len(s))
    fails = []
    for r in s[s["status"] != "ok"].to_dict("records"):
        kg = known_gap(r) and r.get("failure_category") in ("kernel_unregistered_on_backend", "browser_failure", "timeout")
        fails.append({"model": r["model"], "config": r["backend"], "status": r["status"],
                      "category": r.get("failure_category"), "known_gap": kg,
                      "message": str(r.get("failure_message") or BAD_STATUS_MEANS.get(r["status"], ""))[:200]})
    rep["failures"] = fails
    rep["blockers"] = [f for f in fails if not f["known_gap"]]
    if "low_clock_reruns" in s:
        rep["low_clock_reruns"] = int(pd.to_numeric(s["low_clock_reruns"], errors="coerce").fillna(0).sum())
    if "gpu_sm_clock_max_mhz" in s:
        g = pd.to_numeric(s["gpu_sm_clock_max_mhz"], errors="coerce").dropna()
        rep["gpu_launch_sm_max_mhz_min"] = float(g.min()) if len(g) else None
    exp = [json.loads(l) for l in (d / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rep.update(_projection(stage, d, exp))
    return rep


# measured phase -> (count flag, warm-up flag, time-cap flag, min-count flag) of the full stage
PHASE_FLAGS = {
    "measure": ("--measure", "--warmup", "--max-measure-s", "--min-measure"),
    "batch": ("--measure", "--warmup", "--max-measure-s", "--min-measure"),
    "train_measure": ("--train-measure", "--train-warmup", "--train-max-s", "--train-min-measure"),
    "train_batch": ("--train-measure", "--batch-warmup", "--batch-max-s", "--batch-min-measure"),
    "train_pipelined": ("--train-measure", "--train-warmup", "--train-max-s", "--train-min-measure"),
    "infer_b1": ("--infer-measure", "--infer-warmup", "--infer-max-s", "--infer-min-measure"),
    "infer_bB": ("--infer-measure", "--infer-warmup", "--infer-max-s", "--infer-min-measure"),
}
for p in ("phase_fwd", "phase_fwdbwd", "phase_step"):
    PHASE_FLAGS[p] = ("--train-measure", "--train-warmup", "--phase-max-s", "--train-min-measure")
for p in ("lc_train", "lc_train1", "lc_train2"):
    PHASE_FLAGS[p] = ("--train-measure", "--train-warmup", "--train-max-s", "--train-min-measure")
for p in ("lc_infer", "lc_infer1", "lc_infer2"):
    PHASE_FLAGS[p] = ("--infer-measure", "--infer-warmup", "--infer-max-s", "--infer-min-measure")
DEFAULTS = {"--measure": 200, "--warmup": 10, "--train-measure": 100, "--train-warmup": 15, "--infer-measure": 50,
            "--infer-warmup": 5, "--batch-warmup": 5, "--min-measure-s": 2.0, "--train-min-measure": 20,
            "--min-measure": 30, "--infer-min-measure": 20, "--batch-min-measure": 10}
TRIALS = {st: int(_arg(v[1], "--trials", 1)) for st, v in run_study.STAGES.items()}


def _projection(stage, d, exp):
    """Full-protocol duration of a launch-based stage from its smoke launches (sibling method): per smoke
    launch, wall time to the next launch minus its cooling wait, plus the extra timed records of the full
    protocol priced at the launch's own median per (phase, batch), with the full run's caps and minimum
    duration; epoch-based modes scale with the epoch count. Smoke = 1 trial; x the stage's trials."""
    if len(exp) < 2:
        return {}
    full, smoke = run_study.STAGES[stage][1], SMOKE.get(stage, [])
    t = pd.to_datetime([e.get("started_at") for e in exp]).values
    cool = np.array([e.get("cooled_s") or 0.0 for e in exp])
    wall = np.diff(t).astype("timedelta64[s]").astype(float) - cool[:-1]
    wall = np.append(wall, np.median(wall))
    extra = np.zeros(len(exp))
    rf = d / "runs.csv"
    r = pd.read_csv(rf, low_memory=False) if rf.exists() and rf.stat().st_size > 10 else pd.DataFrame()
    if len(r):
        min_s = _arg(full, "--min-measure-s", DEFAULTS["--min-measure-s"])
        for i, e in enumerate(exp):
            g = r[(r["trial"] == e["trial"]) & (r["model"] == e["model"]) & (r["backend"] == e.get("config_label"))]
            for (phase, *_), u in g.groupby(["phase"] + (["batch"] if "batch" in g and phase_has_batch(g) else [])):
                v = float(u["dur_ms"].median()) / 1e3
                if phase in ("train_long", "fit_measure"):
                    k = _arg(full, "--long-epochs" if phase == "train_long" else "--fit-epochs", 1) / \
                        _arg(smoke, "--long-epochs" if phase == "train_long" else "--fit-epochs", 1)
                    extra[i] += (k - 1) * u["dur_ms"].sum() / 1e3
                    continue
                if phase not in PHASE_FLAGS:
                    continue
                cf, wf, capf, minf = PHASE_FLAGS[phase]
                n_full = _arg(full, cf, DEFAULTS.get(cf, 100))
                cap = _arg(full, capf, 1e9)
                t_full = max(min(n_full * v, max(cap, _arg(full, minf, DEFAULTS.get(minf, 0)) * v)), min_s)
                w_extra = max(0.0, _arg(full, wf, DEFAULTS.get(wf, 0)) - _arg(smoke, wf, _arg(full, wf, DEFAULTS.get(wf, 0))))
                extra[i] += max(0.0, t_full - u["dur_ms"].sum() / 1e3) + w_extra * v
    if stage in ("infer_responsiveness", "train_responsiveness"):
        extra += 8.0
    if stage == "infer_load":   # 5 rates x (6 s + 4 s) vs 2 x (2 s + 2 s)
        extra += 5 * 10 - 2 * 4
    per_launch = wall + extra
    return {"smoke_launches": len(exp), "smoke_cooling_min": round(cool.sum() / 60, 1),
            "launch_wall_s_median_excl_cooling": round(float(np.median(wall)), 1),
            "projected_full_minutes": round(per_launch.sum() * TRIALS.get(stage, 1) / 60, 1)}


def phase_has_batch(g):
    return g["phase"].isin(["batch", "train_batch"]).any()


def warmup_check():
    """Warm-up sufficiency on the smoke launches of infer_latency, infer_native, train, native_train (full warm-up rule)."""
    dirs = [str(stage_dir(s)) for s in ("infer_latency", "infer_native", "train", "native_train") if stage_dir(s)]
    if not dirs:
        return None, []
    out = OUT / "analysis" / "warmup_smoke"
    p = subprocess.run([sys.executable, "analysis/warmup.py", *dirs, "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True, errors="replace")
    f = out / "warmup_by_launch.csv"
    if p.returncode != 0 or not f.exists():
        return {"error": p.stderr[-500:]}, [{"stage": "warmup", "problem": "warmup analysis failed"}]
    t = pd.read_csv(f)
    bad = t[t["warmup_sufficient"] == False]  # noqa: E712 (None = not assessable: < MIN_MEASURED records)
    rep = {"launches": len(t), "assessable": int(t["warmup_sufficient"].notna().sum()), "insufficient": len(bad),
           "drift_flags": int((t["drift_flag"] == True).sum()),  # noqa: E712
           "examples": bad[["model", "backend", "mode", "steps_to_steady", "warmup_configured"]].head(20).to_dict("records")}
    return rep, [{"stage": "warmup", "model": r["model"], "config": r["backend"], "mode": r["mode"],
                  "problem": f"warmup_insufficient: steps_to_steady {r['steps_to_steady']} > warm-up {r['warmup_configured']}"}
                 for _, r in bad.iterrows()]


def main():
    report_only = "--report-only" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    blockers = []
    gates = static_gates()
    blockers += [{"stage": "static", "problem": f"{g['gate']}: {g['detail']}"} for g in gates if not g["ok"]]
    if report_only:
        lock = {"sha256": json.loads(next((OUT / s / "stage_done.json").read_text(encoding="utf-8")
                                          for s in run_study.STAGES if (OUT / s / "stage_done.json").exists()))["env_lock_sha256"]}
        rc = None
    else:
        lock = envlock.verify(scope="full")
        print(f"lock verified ({lock['sha256'][:16]})", flush=True)
        if blockers:
            print("static gates failed:\n" + "\n".join(json.dumps(b) for b in blockers))
        cmd = [sys.executable, "run_study.py", "--output-dir", str(OUT), "--resume", "--stage-args", json.dumps(SMOKE)]
        rc = subprocess.call(cmd, cwd=ROOT)
    report = {"env_lock_sha256": lock["sha256"], "static_gates": gates, "run_study_exit": rc, "stages": {}}
    for stage in run_study.STAGES:
        rep = stage_report(stage)
        report["stages"][stage] = rep
        if not rep.get("done"):
            blockers.append({"stage": stage, "problem": "stage did not complete"})
        blockers += [{"stage": stage, **b} for b in rep.get("blockers", [])]
    report["warmup"], wb = warmup_check()
    blockers += wb
    nc = stage_dir("infer_native_correctness")
    if nc and (nc / "correctness.csv").exists():
        c = pd.read_csv(nc / "correctness.csv")
        ok = bool(c.get("inputs_identical", pd.Series([False])).fillna(False).astype(bool).all())
        report["native_correctness_inputs_identical"] = ok
        if not ok:
            blockers.append({"stage": "infer_native_correctness", "problem": "native inputs differ from the browser reference"})
    alog = (OUT / "analysis.log").read_text(encoding="utf-8", errors="replace") if (OUT / "analysis.log").exists() else ""
    report["analysis_tracebacks"] = alog.count("Traceback")
    if report["analysis_tracebacks"]:
        blockers.append({"stage": "analyses", "problem": f"{report['analysis_tracebacks']} analysis tracebacks (analysis.log)"})
    stage_min = {s: json.loads((OUT / s / "stage_done.json").read_text(encoding="utf-8")).get("minutes")
                 for s in run_study.STAGES if (OUT / s / "stage_done.json").exists()}
    if "gpu_trace" in stage_min:   # smoke: 3 models x 1 repeat; full: 30 models x 2 repeats
        report["stages"]["gpu_trace"]["projected_full_minutes"] = round(stage_min["gpu_trace"] * 20, 1)
    if "readback_micro" in stage_min:
        report["stages"]["readback_micro"]["projected_full_minutes"] = round(stage_min["readback_micro"] * 3 * 10 * 85 / 65, 1)
    guards = []
    for st, v in report["stages"].items():
        mins, n, w = stage_min.get(st), v.get("smoke_launches"), v.get("launch_wall_s_median_excl_cooling")
        if mins and n and w:
            guards.append(max(0.0, mins - (n * w + 60 * v.get("smoke_cooling_min", 0)) / 60))
    report["guards_per_stage_min"] = round(float(np.median(guards)), 1) if guards else None
    proj = sum(v.get("projected_full_minutes") or 0 for v in report["stages"].values())
    proj += (report["guards_per_stage_min"] or 0) * len(run_study.STAGES)
    report["projected_full_hours_excl_cooling"] = round(proj / 60, 1)
    report["preflight_stage_hours"] = round(sum(m or 0 for m in stage_min.values()) / 60, 1)
    report["blockers"] = blockers
    report["verdict"] = "GO" if not blockers else "NO-GO"
    report["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "preflight_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = [f"# Preflight: {report['verdict']}", "", f"- environment lock sha256 `{lock['sha256'][:16]}` verified",
             *[f"- static gate {'OK  ' if g['ok'] else 'FAIL'} {g['gate']}: {g['detail']}" for g in gates],
             f"- preflight stages {report['preflight_stage_hours']} h",
             f"- projected full study {report['projected_full_hours_excl_cooling']} h excluding cooling waits "
             f"(guards {report['guards_per_stage_min']} min per stage)",
             f"- warm-up sufficiency: {json.dumps({k: v for k, v in (report['warmup'] or {}).items() if k != 'examples'})}", "",
             "| stage | launches | failures (known gaps) | blockers | low-clock re-runs | smoke cooling (min) | projected full (min) |",
             "|---|---|---|---|---|---|---|"]
    for st, v in report["stages"].items():
        f = v.get("failures", [])
        lines.append(f"| {st} | {v.get('launches', '-' if v.get('done') else 'NOT DONE')} | {len(f)} "
                     f"({sum(x['known_gap'] for x in f)}) | {len(v.get('blockers', []))} | {v.get('low_clock_reruns', '-')} "
                     f"| {v.get('smoke_cooling_min', '-')} | {v.get('projected_full_minutes', '-')} |")
    lines += ["", f"Blockers: {len(blockers)}"] + [f"- {json.dumps(b)}" for b in blockers]
    (OUT / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
