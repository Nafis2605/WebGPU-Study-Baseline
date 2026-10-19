"""
Is the short training window representative of training deep into a run? (Decided before any study data:
reports/methodology/VERIFICATION.md, "shortened training".)

For every (model, config) with train_long launches (E epochs, every step recorded):
  long        steps of epochs 2..E (epoch index >= 1) of the train_long launches, pooled
  short_same  steps W..W+M-1 of the same train_long launches (the train stage's window position:
              W = 15 warm-up steps, M = 100 measured steps)
  short_stage the train stage's pooled train_measure steps of the same (model, config), if given
Test: two one-sided tests on the ratio of medians long / short, via its 90 % percentile-bootstrap CI
(stats.ratio_of_medians_ci, level 0.90): equivalent iff the CI lies inside [1 - MARGIN, 1 + MARGIN],
MARGIN = 0.05. Also: Spearman rho of step time vs step index over the whole long run (drift).
A pair that is not equivalent is reported as window_not_representative; nothing is redefined afterwards.

Usage:  python analysis/equivalence.py --long <train_long dir> [...] [--stage <train dir> ...] --out DIR
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats import ratio_of_medians_ci, spearman

MARGIN = 0.05
W, M = 15, 100
KEYS = ["source", "trial", "model", "backend"]


def load(dirs, mode, phase):
    out = []
    for d in map(Path, dirs):
        s = pd.read_csv(d / "sessions.csv")
        r = pd.read_csv(d / "runs.csv", low_memory=False)
        ok = s[(s["status"] == "ok") & (s["mode"] == mode)][["trial", "model", "backend"]]
        out.append(r[r["phase"] == phase].merge(ok, on=["trial", "model", "backend"]).assign(source=d.name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", nargs="+", required=True)
    ap.add_argument("--stage", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lg = load(args.long, "train_long", "train_long")
    st = load(args.stage, "train", "train_measure") if args.stage else pd.DataFrame()
    rows = []
    for (model, backend), g in lg.groupby(["model", "backend"]):
        long = g[g["epoch"] >= 1]["dur_ms"].to_numpy()
        same = g[(g["i"] >= W) & (g["i"] < W + M)]["dur_ms"].to_numpy()
        rho, p = spearman(g["i"], g["dur_ms"]) if len(g) > 2 else (np.nan, np.nan)
        comps = [("short_same_launch", same)]
        if len(st):
            comps.append(("train_stage", st[(st["model"] == model) & (st["backend"] == backend)]["dur_ms"].to_numpy()))
        for name, short in comps:
            if len(long) < 10 or len(short) < 10:
                rows.append({"model": model, "backend": backend, "comparison": name, "error": "too few steps"})
                continue
            r, lo, hi = ratio_of_medians_ci(long, short, level=0.90)
            rows.append({"model": model, "backend": backend, "comparison": name, "long_steps": len(long),
                         "short_steps": len(short), "long_median_ms": float(np.median(long)),
                         "short_median_ms": float(np.median(short)), "ratio": r, "ci90_lo": lo, "ci90_hi": hi,
                         "equivalent": bool(lo >= 1 - MARGIN and hi <= 1 + MARGIN),
                         "long_run_rho": rho, "long_run_rho_p": p, "launches": g["source"].nunique() * g["trial"].nunique()})
    t = pd.DataFrame(rows)
    if "equivalent" in t:
        t["window_not_representative"] = ~t["equivalent"].fillna(False).astype(bool)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "equivalence.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(t.round(4).to_string(index=False))
    print(f"\nWritten to {out / 'equivalence.csv'}")


if __name__ == "__main__":
    main()
