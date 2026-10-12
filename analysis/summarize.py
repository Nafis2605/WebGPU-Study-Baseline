"""
Latency / step-time statistics: distribution, closed-loop throughput, speedups, batch scaling.
Port of the sibling TF.js study's analysis/summarize.py (itself a port of the ONNX study's), parametrised
by series so the same statistics cover every pillar:
    --series infer        inference runs        (mode latency, phase measure)        [default]
    --series train        training steps        (mode train, phase train_measure)
    --series infer_b1     post-training inference at batch 1   (mode train, phase infer_b1)
    --series infer_bB     post-training inference at the training batch (mode train, phase infer_bB)
Per-sample columns (per_sample_p50_ms, samples_per_s) divide by each record's batch.
Native launches (native/run_native.py results dirs) use the same schema and are summarised alongside.

Only timing launches are used: status == "ok" (backend verified by API counters) and not
instrumented (--instrument-runs). Quantiles use numpy's default linear interpolation
(Hyndman & Fan type 7).

Writes, into --out (default: the first input dir):
    latency_by_trial.csv   one row per (model, config, trial): distribution, throughput, the split
                           of dur_ms into exec_ms (until execute() returned) and read_ms (data()),
                           p99/p50 tail ratio, and stationarity (Spearman rho of dur_ms vs run
                           index; drift = p < 0.01 and |rho| > 0.3, the ONNX study's rule;
                           drift_ratio = median of the last quarter of runs / first quarter, so a
                           flagged drift comes with its size)
    latency_summary.csv    one row per (model, config), pooled over trials: distribution with
                           bootstrap CIs for p50 and p99, ICC(1) across trials, CoV of trial medians
    speedup_summary.csv    ratio of medians (baseline / target) with a bootstrap CI:
                             webgpu vs wasm-1t, wasm-4t, wasm-8t, webgl;  webgl vs wasm-8t;
                             every why-axis config vs its parent (webgpu-strict vs webgpu, ...)
                           faster = CI entirely above 1, slower = CI entirely below 1
    batch_summary.csv      batch mode: per (model, config, batch) median whole-batch latency and
                           per-sample latency (dur_ms / batch); amortisation = per-sample at the
                           smallest batch / per-sample at the largest

Usage:
    python analysis/summarize.py <results dir> [<results dir> ...] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats import icc_oneway, ratio_of_medians_ci, spearman

QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
# With fewer samples than this, the sample p99 is set by one or two runs; it's still reported but flagged.
MIN_RUNS_FOR_P99 = 100
KEYS = ["source", "trial", "model", "backend"]
# Why-axis configs compared against the config they modify.
PARENT = {"webgl-strict": "webgl", "webgpu-strict": "webgpu", "webgl-precompile": "webgl", "webgpu-precompile": "webgpu",
          "webgl-diskcache": "webgl", "webgpu-diskcache": "webgpu",
          "webgpu-sb1": "webgpu", "webgpu-sb4": "webgpu", "webgpu-sb64": "webgpu", "webgpu-sb512": "webgpu",
          "webgl-gpuio": "webgl", "webgpu-gpuio": "webgpu", "webgl-fence": "webgl", "webgpu-fence": "webgpu",
          "webgl-sync": "webgl", "webgpu-sync": "webgpu", "webgl-tfence": "webgl", "webgpu-tfence": "webgpu",
          "webgl-resident": "webgl", "webgpu-resident": "webgpu", "wasm-1t": "wasm-8t", "wasm-4t": "wasm-8t",
          "tf-cuda": "tf-cuda-fp32", "tf-cuda-resident": "tf-cuda"}
BASELINE_PAIRS = [("wasm-1t", "webgpu"), ("wasm-4t", "webgpu"), ("wasm-8t", "webgpu"), ("webgl", "webgpu"),
                  ("wasm-8t", "webgl"),
                  # browser vs native (ratio > 1: the browser config is slower than native)
                  ("webgpu", "tf-cuda-fp32"), ("webgpu", "tf-cuda"), ("webgpu", "tf-cuda-eager"),
                  ("webgl", "tf-cuda-fp32"), ("wasm-8t", "tf-cpu"), ("tf-cuda-eager", "tf-cuda-fp32")]
SERIES = {"infer": ("latency", "measure"), "train": ("train", "train_measure"),
          "infer_b1": ("train", "infer_b1"), "infer_bB": ("train", "infer_bB")}


def _bool(col):
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(dirs):
    runs, sessions, envs = [], [], {}
    for d in map(Path, dirs):
        r = pd.read_csv(d / "runs.csv", low_memory=False)
        s = pd.read_csv(d / "sessions.csv")
        runs.append(r.assign(source=d.name))
        sessions.append(s.assign(source=d.name))
        if (d / "env.json").exists():
            envs[d.name] = json.loads((d / "env.json").read_text(encoding="utf-8"))
    runs = pd.concat(runs, ignore_index=True)
    sessions = pd.concat(sessions, ignore_index=True)
    sessions["instrumented"] = _bool(sessions["instrumented"]) if "instrumented" in sessions else False
    return runs, sessions, envs


def describe(durations):
    x = np.asarray(durations, dtype=float)
    out = {"n": len(x), "mean_ms": x.mean(), "std_ms": x.std(ddof=1) if len(x) > 1 else np.nan,
           "min_ms": x.min(), "max_ms": x.max()}
    for name, q in QUANTILES.items():
        out[f"{name}_ms"] = np.quantile(x, q)
    out["iqr_ms"] = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    out["cov"] = out["std_ms"] / out["mean_ms"]
    out["tail_ratio_p99_p50"] = out["p99_ms"] / out["p50_ms"]
    out["p99_reliable"] = len(x) >= MIN_RUNS_FOR_P99
    return out


def throughput(group):
    """Completed requests per second over the measured window, including gaps between runs."""
    start = group["t_start_epoch_ms"].min()
    end = (group["t_start_epoch_ms"] + group["dur_ms"]).max()
    return len(group) / ((end - start) / 1000.0)


def bootstrap_ci(x, stat_q, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    stats = np.quantile(rng.choice(x, size=(n_boot, len(x)), replace=True), stat_q, axis=1)
    return np.quantile(stats, 0.025), np.quantile(stats, 0.975)


def timing_ok(sessions, mode):
    return sessions[(sessions["status"] == "ok") & (sessions["mode"] == mode) & ~sessions["instrumented"]][KEYS]


def summarize(runs, sessions, series="infer"):
    mode, phase = SERIES[series]
    measured = runs[runs["phase"] == phase].merge(timing_ok(sessions, mode), on=KEYS)
    if "batch" not in measured:
        measured["batch"] = 1
    measured["batch"] = measured["batch"].fillna(1)

    by_trial = []
    for (src, model, backend, trial), g in measured.groupby(["source", "model", "backend", "trial"]):
        g = g.sort_values("i")
        rho, p = spearman(g["i"], g["dur_ms"]) if len(g) > 2 else (np.nan, np.nan)
        q = max(len(g) // 4, 1)
        drift_ratio = g["dur_ms"].iloc[-q:].median() / g["dur_ms"].iloc[:q].median()
        by_trial.append({"source": src, "model": model, "backend": backend, "trial": trial,
                         **describe(g["dur_ms"]), "throughput_rps": throughput(g),
                         "exec_ms_p50": g["exec_ms"].median(), "read_ms_p50": g["read_ms"].median(),
                         "per_sample_p50_ms": (g["dur_ms"] / g["batch"]).median(),
                         "rho": rho, "rho_p": p, "drift": bool(p < 0.01 and abs(rho) > 0.3),
                         "drift_ratio": drift_ratio})
    by_trial = pd.DataFrame(by_trial)

    summary = []
    for (model, backend), g in measured.groupby(["model", "backend"]):
        t = by_trial[(by_trial["model"] == model) & (by_trial["backend"] == backend)]
        row = {"model": model, "backend": backend, "trials": len(t), **describe(g["dur_ms"]),
               "throughput_rps_mean": t["throughput_rps"].mean(),
               "per_sample_p50_ms": (g["dur_ms"] / g["batch"]).median(),
               "samples_per_s": 1000.0 / (g["dur_ms"] / g["batch"]).median(),
               "exec_ms_p50": g["exec_ms"].median(), "read_ms_p50": g["read_ms"].median(),
               "trial_median_min_ms": t["p50_ms"].min(), "trial_median_max_ms": t["p50_ms"].max(),
               "between_trial_cov": t["p50_ms"].std(ddof=1) / t["p50_ms"].mean() if len(t) > 1 else np.nan,
               "icc_trials": icc_oneway([gg["dur_ms"].to_numpy() for _, gg in g.groupby(["source", "trial"])]),
               "drifting_trials": int(t["drift"].sum())}
        row["p50_ci95_lo_ms"], row["p50_ci95_hi_ms"] = bootstrap_ci(g["dur_ms"], 0.50)
        row["p99_ci95_lo_ms"], row["p99_ci95_hi_ms"] = bootstrap_ci(g["dur_ms"], 0.99)
        summary.append(row)
    summary = pd.DataFrame(summary)
    return measured, by_trial, summary


def speedups(measured):
    lat = {k: g["dur_ms"].to_numpy() for k, g in measured.groupby(["model", "backend"])}
    pairs = BASELINE_PAIRS + [(parent, child) for child, parent in PARENT.items()]
    rows = []
    for model in sorted({m for m, _ in lat}):
        for base, target in pairs:
            if (model, base) not in lat or (model, target) not in lat:
                continue
            r, lo, hi = ratio_of_medians_ci(lat[(model, base)], lat[(model, target)])
            rows.append({"model": model, "baseline": base, "target": target,
                         "kind": "why_axis" if target in PARENT else "backend",
                         "baseline_p50_ms": float(np.median(lat[(model, base)])),
                         "target_p50_ms": float(np.median(lat[(model, target)])),
                         "speedup": r, "ci_lo": lo, "ci_hi": hi,
                         "faster": bool(lo > 1), "slower": bool(hi < 1)})
    return pd.DataFrame(rows)


def batch_table(runs, sessions, series="infer"):
    mode, phase = ("batch", "batch") if series.startswith("infer") else ("train_batch", "train_batch")
    b = runs[runs["phase"] == phase].merge(timing_ok(sessions, mode), on=KEYS)
    if b.empty:
        return pd.DataFrame()
    b = b.assign(per_sample_ms=b["dur_ms"] / b["batch"])
    t = (b.groupby(["model", "backend", "batch"])
         .agg(n=("dur_ms", "size"), trials=("trial", "nunique"), p50_ms=("dur_ms", "median"),
              p95_ms=("dur_ms", lambda x: np.quantile(x, 0.95)), per_sample_p50_ms=("per_sample_ms", "median"),
              throughput_samples_per_s=("per_sample_ms", lambda x: 1000.0 / np.median(x)))
         .reset_index())
    amort = []
    for (model, backend), g in t.groupby(["model", "backend"]):
        g = g.sort_values("batch")
        amort.append({"model": model, "backend": backend,
                      "amortisation": g["per_sample_p50_ms"].iloc[0] / g["per_sample_p50_ms"].iloc[-1],
                      "batch_min": int(g["batch"].iloc[0]), "batch_max": int(g["batch"].iloc[-1])})
    return t.merge(pd.DataFrame(amort), on=["model", "backend"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="output directory (default: the first input dir)")
    ap.add_argument("--series", default="infer", choices=list(SERIES))
    args = ap.parse_args()

    runs, sessions, envs = load(args.dirs)
    out = Path(args.out or args.dirs[0])
    out.mkdir(parents=True, exist_ok=True)
    measured, by_trial, summary = summarize(runs, sessions, args.series)
    sp = speedups(measured)
    batch = batch_table(runs, sessions, args.series)
    tag = args.series
    by_trial.to_csv(out / f"{tag}_by_trial.csv", index=False)
    summary.to_csv(out / f"{tag}_summary.csv", index=False)
    if not sp.empty:
        sp.to_csv(out / f"{tag}_speedup_summary.csv", index=False)
    if not batch.empty:
        batch.to_csv(out / f"{tag}_batch_summary.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        gran = {src: (e.get("page_env") or {}).get("timer_granularity_ms") for src, e in envs.items()}
        print("Timer granularity (ms) per results dir:", gran)
        if len(summary):
            cols = ["model", "backend", "trials", "n", "p50_ms", "p50_ci95_lo_ms", "p50_ci95_hi_ms", "per_sample_p50_ms", "p95_ms",
                    "p99_ms", "tail_ratio_p99_p50", "cov", "exec_ms_p50", "read_ms_p50", "throughput_rps_mean",
                    "between_trial_cov", "icc_trials", "drifting_trials", "p99_reliable"]
            print("\nLatency (pooled over trials):")
            print(summary[cols].round(3).to_string(index=False))
        else:
            print("No verified latency experiments.")
        if len(by_trial):
            d = by_trial[by_trial["drift"]]
            span = (f"; last/first-quarter median ratio of those: {d['drift_ratio'].min():.3f}-"
                    f"{d['drift_ratio'].max():.3f}") if len(d) else ""
            print(f"\nStationarity: {len(d)}/{len(by_trial)} launches drift (|rho| > 0.3, p < 0.01){span}")
        if len(sp):
            print("\nSpeedups (baseline p50 / target p50):")
            print(sp.round(3).to_string(index=False))
        if len(batch):
            print("\nBatch scaling:")
            print(batch.round(3).to_string(index=False))
        not_ok = sessions[sessions["status"] != "ok"][
            ["source", "trial", "model", "backend", "status", "failure_category", "failure_message"]]
        if len(not_ok):
            print("\nExcluded experiments:")
            print(not_ok.to_string(index=False, max_colwidth=90))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
