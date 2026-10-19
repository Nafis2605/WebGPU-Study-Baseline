"""
Firefox vs Chrome comparison for the cross-browser verification flight. The rules are fixed in
reports/<device>/compare/VERIFICATION_PLAN.md before any Firefox flight data exists; this script applies them.

  margins   reads only the Chrome flight and writes the equivalence margins (reports/<device>/compare/margins.json)
  compare   reads both flights and classifies every (stage, model, config, metric) cell

Unit of comparison: the per-launch median (one fresh browser launch = one observation; records inside a launch are
not independent). Chrome has 3 launches per cell in the main stages and 2 in the why-axis stages, Firefox the same
counts.

Classification of a cell, with m = the pre-registered margin of its metric group and backend family:
  ratio  = median(Firefox launch medians) / median(Chrome launch medians)
  equivalent    |ratio - 1| <= m
  different     |ratio - 1| >  m  and every Firefox launch median lies beyond the Chrome launch range on the same side
  inconclusive  |ratio - 1| >  m  otherwise
  below_floor   Chrome median < FLOOR_MS (timer resolution is 5 us on Chrome, 20 us on Firefox): reported, not classified
A bootstrap 95% CI of the ratio of medians (analysis/stats.py) is reported next to the class, not used for it: with 3
launches per side it takes only a few distinct values.

Usage:
    python analysis/browser_compare.py margins
    python analysis/browser_compare.py compare [--device-id D] [--chrome DIR] [--firefox DIR] \
        [--out DIR]   (defaults: results/<device>/chrome-<ver>/full and results/<device>/firefox-<ver>/full)
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stats import ratio_of_medians_ci  # noqa: E402
from benchkit import layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def margins_path():
    return layout.reports_dir() / "compare" / "margins.json"
FLOOR_MS = 0.2          # below this Chrome median the 20 us Firefox timer step is >= 10% of the value
MARGIN_FLOOR = 0.10     # no margin is narrower than 10%
CORR_ABS = 5e-6         # inference max_abs_err bound: the Chrome flight's maximum over browser backends is 4.23e-6
GRAD_FACTOR, GRAD_ABS = 2.0, 1e-5   # training: Firefox grad_rel_l2 <= max(GRAD_FACTOR x Chrome value, GRAD_ABS)

# metric group -> (stages that hold it, source table, phase or column, value column)
# runs.csv groups: records of a phase; sessions.csv groups: one value per launch.
GROUPS = {
    "infer_total_ms": (("infer_latency", "infer_why"), "runs", "measure", "dur_ms"),
    "infer_exec_ms": (("infer_latency", "infer_why"), "runs", "measure", "exec_ms"),
    "infer_read_ms": (("infer_latency", "infer_why"), "runs", "measure", "read_ms"),
    "train_step_ms": (("train", "train_why"), "runs", "train_measure", "dur_ms"),
    "infer_first_run_ms": (("infer_latency", "infer_why"), "sessions", None, "first_run_ms"),
    "train_first_step_ms": (("train", "train_why"), "sessions", None, "first_step_ms"),
}
MARGIN_STAGES = {"infer": ("infer_latency",), "train": ("train",)}   # primary stages set the margins
COUNTS = ["webgpu_submits", "webgpu_dispatches", "webgl_draws", "kernels_run", "forwarded_kernels"]


def family(config):
    return "webgpu" if config.startswith("webgpu") else "webgl" if config.startswith("webgl") else \
        "wasm" if config.startswith("wasm") else config


def stage_dirs(study, stage):
    return sorted(glob.glob(str(Path(study) / stage / "2*")))


def load(study, stage, table):
    frames = [pd.read_csv(Path(d) / f"{table}.csv", low_memory=False) for d in stage_dirs(study, stage)
              if (Path(d) / f"{table}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def launch_medians(study, group):
    """DataFrame [stage, model, backend(config), trial, value] with one row per launch."""
    stages, table, phase, col = GROUPS[group]
    out = []
    for stage in stages:
        df = load(study, stage, table)
        if df.empty or col not in df:
            continue
        if table == "runs":
            df = df[df.phase == phase]
            if "config_label" in df:
                key = "config_label"
            else:
                key = "backend"
        else:
            df = df[df.status == "ok"] if "status" in df else df
            key = "config_label" if "config_label" in df else "backend"
        df = df.assign(config=df[key])
        g = df.groupby(["model", "config", "trial"])[col].median().dropna().reset_index(name="value")
        g.insert(0, "stage", stage)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _config_labels(study, stage):
    """runs.csv has only 'backend' (= the config label for why-axis configs); sessions.csv is authoritative."""
    s = load(study, stage, "sessions")
    return s


def spread(vals):
    v = np.asarray(vals, dtype=float)
    return (v.max() - v.min()) / np.median(v) if len(v) > 1 and np.median(v) > 0 else np.nan


def compute_margins(chrome):
    res = {"basis": "Chrome flight primary stages: per cell (model x backend) the relative range (max-min)/median of the "
                    "launch medians; margin = max(MARGIN_FLOOR, 90th percentile over models) per metric group and backend "
                    "family", "margin_floor": MARGIN_FLOOR, "floor_ms": FLOOR_MS, "groups": {}}
    for group in GROUPS:
        lm = launch_medians(chrome, group)
        lm = lm[lm.stage.isin(MARGIN_STAGES["train" if group.startswith("train") else "infer"])]
        res["groups"][group] = {}
        for fam, g in lm.assign(fam=lm.config.map(family)).groupby("fam"):
            cells = g.groupby(["model", "config"]).value.apply(spread).dropna()
            med = g.groupby(["model", "config"]).value.median()
            cells = cells[med.reindex(cells.index) >= (FLOOR_MS if group.endswith("_ms") and "first" not in group else 0)]
            if len(cells):
                res["groups"][group][fam] = {"margin": round(float(max(MARGIN_FLOOR, np.quantile(cells, 0.9))), 4),
                                             "p90_range": round(float(np.quantile(cells, 0.9)), 4),
                                             "median_range": round(float(np.median(cells)), 4), "cells": int(len(cells))}
    corr = load(chrome, "infer_correctness", "correctness")
    if not corr.empty:
        b = corr[corr.backend.isin(["wasm-8t", "webgl", "webgpu"])]
        res["inference_correctness"] = {"chrome_max_abs_err_max": float(b.max_abs_err.max()), "bound_used": CORR_ABS,
                                        "chrome_within_bound": bool((b.max_abs_err <= CORR_ABS).all())}
    return res


def classify(ch, ff, margin):
    c_med, f_med = float(np.median(ch)), float(np.median(ff))
    if c_med < FLOOR_MS and margin.get("floor_applies", True):
        return "below_floor", f_med / c_med if c_med else np.nan
    r = f_med / c_med
    if abs(r - 1) <= margin["margin"]:
        return "equivalent", r
    lo, hi = min(ch), max(ch)
    beyond = all(x > hi for x in ff) if r > 1 else all(x < lo for x in ff)
    return ("different" if beyond else "inconclusive"), r


def compare(chrome, firefox, out, correctness_from=None):
    corr_dir = correctness_from or firefox   # where the Firefox correctness stages and train_correctness analysis live
    margins = json.loads(margins_path().read_text(encoding="utf-8"))
    rows = []
    for group in GROUPS:
        c, f = launch_medians(chrome, group), launch_medians(firefox, group)
        if c.empty or f.empty:
            continue
        for (stage, model, config), fg in f.groupby(["stage", "model", "config"]):
            cg = c[(c.stage == stage) & (c.model == model) & (c.config == config)]
            if cg.empty:
                rows.append({"group": group, "stage": stage, "model": model, "config": config, "cls": "no_chrome_cell"})
                continue
            m = margins["groups"][group].get(family(config))
            if m is None:
                continue
            floor_applies = not group.startswith(("infer_first", "train_first"))
            cls, r = classify(cg.value.values, fg.value.values, {**m, "floor_applies": floor_applies})
            ci = ratio_of_medians_ci(fg.value.values, cg.value.values) if len(cg) > 1 and len(fg) > 1 else (r, np.nan, np.nan)
            rows.append({"group": group, "stage": stage, "model": model, "config": config, "family": family(config),
                         "margin": m["margin"], "chrome_launches": len(cg), "firefox_launches": len(fg),
                         "chrome_median": float(cg.value.median()), "firefox_median": float(fg.value.median()),
                         "ratio": r, "ci_lo": ci[1], "ci_hi": ci[2], "cls": cls})
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "cells.csv", index=False)
    summ = res.groupby(["group", "family", "cls"]).size().unstack(fill_value=0)
    summ.to_csv(out / "summary_by_group_family.csv")
    print(summ.to_string())

    # Same GPU work: counts per launch must be identical between browsers (TF.js issues the same dispatches).
    crow = []
    for stage in ("infer_latency", "infer_why", "infer_instrumented", "train", "train_why", "train_instrumented"):
        cs, fs = load(chrome, stage, "sessions"), load(firefox, stage, "sessions")
        if cs.empty or fs.empty:
            continue
        key = "config_label" if "config_label" in cs else "backend"
        for col in [x for x in COUNTS if x in cs and x in fs]:
            cm = cs.groupby(["model", key])[col].median()
            fm = fs.groupby(["model", key])[col].median()
            j = pd.concat([cm.rename("chrome"), fm.rename("firefox")], axis=1).dropna()
            for (model, cfg), r in j.iterrows():
                crow.append({"stage": stage, "model": model, "config": cfg, "count": col, "chrome": r.chrome,
                             "firefox": r.firefox, "equal": bool(r.chrome == r.firefox)})
    cnt = pd.DataFrame(crow)
    cnt.to_csv(out / "counts.csv", index=False)
    if len(cnt):
        print(f"\nGPU-work counts: {int(cnt.equal.sum())} of {len(cnt)} cells equal")
        print(cnt[~cnt.equal].head(30).to_string(index=False))

    # Inference numerics: Firefox error bound and top-1 agreement.
    fc = load(corr_dir, "infer_correctness", "correctness")
    cc = load(chrome, "infer_correctness", "correctness")
    if not fc.empty:
        b = fc[fc.backend.isin(["wasm-8t", "webgl", "webgpu"])]
        top1_ff = b.groupby(["model", "backend"]).top1_agreement.min()
        top1_ch = cc.groupby(["model", "backend"]).top1_agreement.min()
        bad_top1 = [k for k in top1_ff.index if k in top1_ch.index and top1_ch[k] >= 1.0 and top1_ff[k] < 1.0]
        print(f"\ninference correctness: Firefox max_abs_err max {b.max_abs_err.max():.3g} (bound {CORR_ABS:g}), "
              f"cells above bound {int((b.max_abs_err > CORR_ABS).sum())} of {len(b)}, top-1 regressions {bad_top1}")
        pd.DataFrame({"firefox_max_abs_err": b.groupby(["model", "backend"]).max_abs_err.max()}).to_csv(
            out / "infer_correctness.csv")
    # Training numerics against the shared tf-cpu + TF.js-Adam reference (analysis/train_correctness.py output).
    name = "train_correctness/train_correctness_vs_tf-cpu_tfjsadam.csv"
    cf, cch = Path(corr_dir) / "analysis" / name, Path(chrome) / "analysis" / name
    if cf.exists() and cch.exists():
        f, c = pd.read_csv(cf), pd.read_csv(cch)
        f, c = (x[x.config.isin(["wasm-8t", "webgl", "webgpu"]) & (x.status == "ok")] for x in (f, c))
        j = f.merge(c, on=["model", "config"], suffixes=("_ff", "_ch"))
        j["bound"] = np.maximum(GRAD_FACTOR * j.grad_rel_l2_ch, GRAD_ABS)
        # Cells where the Chrome flight itself has no gradient metric (NaN) cannot be compared.
        j["comparable"] = j.grad_rel_l2_ch.notna()
        j["pass"] = j.comparable & (j.grad_rel_l2_ff <= j.bound)
        j.to_csv(out / "train_numerics.csv", index=False)
        cmp_ = j[j.comparable]
        print(f"\ntraining gradients: {int(cmp_['pass'].sum())} of {len(cmp_)} comparable cells within "
              f"max({GRAD_FACTOR}x Chrome, {GRAD_ABS:g}); {int((~j.comparable).sum())} cells not comparable "
              f"(no Chrome gradient metric)")
        print(cmp_[~cmp_["pass"]][["model", "config", "grad_rel_l2_ff", "grad_rel_l2_ch"]].to_string(index=False))
    return res, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("margins", "compare"))
    ap.add_argument("--device-id", default=None)
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--firefox", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--correctness-from", default=None,
                    help="study folder holding the Firefox correctness stages (default: --firefox); used when they were not re-run")
    a = ap.parse_args()
    if a.device_id:
        os.environ["TRAINBENCH_DEVICE"] = a.device_id
    a.chrome = a.chrome or str(layout.run_dir("chrome", "full"))
    a.firefox = a.firefox or str(layout.run_dir("firefox", "full"))
    if a.cmd == "margins":
        res = compute_margins(a.chrome)
        margins_path().parent.mkdir(parents=True, exist_ok=True)
        margins_path().write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({g: {f: v["margin"] for f, v in d.items()} for g, d in res["groups"].items()}, indent=1))
        print(res.get("inference_correctness"))
    else:
        compare(a.chrome, a.firefox, Path(a.out or Path(a.firefox) / "analysis" / "browser_compare"), a.correctness_from)


if __name__ == "__main__":
    main()
