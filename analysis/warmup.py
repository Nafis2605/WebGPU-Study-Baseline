"""
Pillar 1 (warm-up / cold start) for both inference and training launches. Definitions fixed before any
study data was collected (study_report/VERIFICATION.md):

  steady median s       median of the launch's measured records (inference: phase "measure";
                        training: "train_measure")
  band b                P90 of the launch's measured records (the steady state's own spread, so noisy
                        backends aren't held to a fixed tolerance)
  steps_to_steady k     the first index k (0-based, over warm-up + measured records in order) such that
                        t_k <= b and, for every position i >= k, the forward median of the next WINDOW
                        records is <= b (positions in the last WINDOW-1 records use the last full window). The per-record condition catches a single cold
                        step; the window condition requires the series to stay in the band.
  cold_overhead_ms      sum over records before k of (t_i - s)
  first_overhead_ms     t_0 - s (inference: first_run_overhead_ms, the sibling study's definition)
  time_to_first_result  backend_init + fetch + weights_verify + session_create (+ precompile) + first run / step
                        (data_load is the harness's input pipeline and is reported separately)
  warmup_sufficient     k <= the launch's configured warm-up count (else the measured window starts
                        before steady state; flagged, and preflight treats it as a blocker)

Timing launches only (status ok, not instrumented). Writes warmup_by_launch.csv and warmup_summary.csv
(median over trials with a bootstrap CI when >= 3 trials).

Usage:  python analysis/warmup.py <results dir> [...] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats import median_ci

WINDOW = 5
BAND_QUANTILE = 0.90
KEYS = ["source", "trial", "model", "backend"]
SERIES = {
    # mode -> (warm-up phase, measured phase, configured warm-up column in sessions/env)
    "latency": ("warmup", "measure"),
    "train": ("train_warmup", "train_measure"),
}


def _bool(col):
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(dirs):
    runs, sessions = [], []
    for d in map(Path, dirs):
        runs.append(pd.read_csv(d / "runs.csv", low_memory=False).assign(source=d.name))
        sessions.append(pd.read_csv(d / "sessions.csv").assign(source=d.name))
    s = pd.concat(sessions, ignore_index=True)
    s["instrumented"] = _bool(s["instrumented"]) if "instrumented" in s else False
    return pd.concat(runs, ignore_index=True), s


def steps_to_steady(t, band, window=WINDOW):
    t = np.asarray(t, float)
    n = len(t)
    # full windows only: positions in the last window-1 records are covered by the last full window
    last = max(n - window, 0)
    stays = np.array([np.median(t[min(i, last):min(i, last) + window]) <= band for i in range(n)])
    # suffix_ok[i]: every window from i on stays in the band
    suffix_ok = np.flip(np.cumprod(np.flip(stays)).astype(bool))
    for k in range(n):
        if t[k] <= band and suffix_ok[k]:
            return k
    return None


def per_launch(runs, sessions):
    rows = []
    for mode, (wph, mph) in SERIES.items():
        ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == mode) & ~sessions["instrumented"]]
        for _, sess in ok.iterrows():
            key = {k: sess[k] for k in KEYS}
            r = runs.merge(pd.DataFrame([key]), on=KEYS)
            w = r[r["phase"] == wph].sort_values("i")
            m = r[r["phase"] == mph].sort_values("i")
            if m.empty:
                continue
            series = np.concatenate([w["dur_ms"].to_numpy(), m["dur_ms"].to_numpy()])
            s = float(np.median(m["dur_ms"]))
            band = float(np.quantile(m["dur_ms"], BAND_QUANTILE))
            k = steps_to_steady(series, band)
            first = float(series[0])
            init = sum(float(sess.get(c) or 0) for c in ("backend_init_ms", "fetch_ms", "weights_verify_ms",
                                                         "session_create_ms", "precompile_ms") if pd.notna(sess.get(c)))
            rows.append({**key, "mode": mode, "pillar": sess.get("pillar"), "family": sess.get("family"),
                         "size": sess.get("size"), "warmup_configured": len(w), "steady_median_ms": s, "steady_band_ms": band,
                         "first_ms": first, "second_ms": float(series[1]) if len(series) > 1 else np.nan,
                         "first_overhead_ms": first - s, "first_over_steady": first / s,
                         "steps_to_steady": k,
                         "cold_overhead_ms": float(np.sum(series[:k] - s)) if k is not None else np.nan,
                         "warmup_sufficient": (k is not None and k <= len(w)),
                         "backend_init_ms": sess.get("backend_init_ms"), "fetch_ms": sess.get("fetch_ms"),
                         "session_create_ms": sess.get("session_create_ms"), "build_ms": sess.get("build_ms"),
                         "weights_assign_ms": sess.get("weights_assign_ms"), "precompile_ms": sess.get("precompile_ms"),
                         "data_load_ms": (sess.get("data_fetch_ms") or 0) + (sess.get("data_verify_ms") or 0),
                         "time_to_first_result_ms": init + first})
    return pd.DataFrame(rows)


def summary(t):
    rows = []
    cols = ["backend_init_ms", "fetch_ms", "session_create_ms", "precompile_ms", "first_ms", "second_ms",
            "steady_median_ms", "first_overhead_ms", "first_over_steady", "steps_to_steady", "cold_overhead_ms",
            "time_to_first_result_ms"]
    for (mode, model, backend), g in t.groupby(["mode", "model", "backend"]):
        row = {"mode": mode, "model": model, "backend": backend, "trials": len(g),
               "warmup_insufficient_trials": int((~g["warmup_sufficient"].astype(bool)).sum())}
        for c in cols:
            v = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy()
            if len(v) >= 3:
                row[f"{c}_median"], row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = median_ci(v)
            else:
                row[f"{c}_median"] = float(np.median(v)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs, sessions = load(args.dirs)
    out = Path(args.out or args.dirs[0])
    out.mkdir(parents=True, exist_ok=True)
    t = per_launch(runs, sessions)
    if t.empty:
        print("no ok timing launches")
        return
    t.to_csv(out / "warmup_by_launch.csv", index=False)
    s = summary(t)
    s.to_csv(out / "warmup_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(s[["mode", "model", "backend", "trials", "first_ms_median", "steady_median_ms_median",
                 "first_over_steady_median", "steps_to_steady_median", "cold_overhead_ms_median",
                 "warmup_insufficient_trials"]].round(2).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
