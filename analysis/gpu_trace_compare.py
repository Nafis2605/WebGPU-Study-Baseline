"""
Compare the Nsight Systems GPU-metrics summaries (analysis/nsys_summary.py) of two browsers, per (model, config, task).

Both summaries give, inside the measured inferences or steps of each capture: the page's median time, the mean GPU clock
(GPC), the share of metric samples at boost clock (>= 1,700 MHz), and the mean GR Active and SMs Active percentages. These
are device-wide hardware counters sampled at 10 kHz, so they do not depend on whether the browser's GPU process could be
instrumented. Direct3D 12 workload columns exist only for captures that traced the API (Chrome); they are left out here.

Each capture is repeated; the table uses the median over repeats. Differences are candidate minus baseline for clock and
activity, candidate / baseline for the page time.

Usage:
    python analysis/gpu_trace_compare.py [--baseline CSV] [--candidate CSV] [--out DIR] [--models D3 C3 ...]
(defaults: the Chrome flight's analysis/gpu_trace/nsys_summary.csv and the Firefox gpu_trace/analysis/nsys_summary.csv)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit import layout  # noqa: E402

METRICS = ["page_p50_ms", "gpc_clock_mhz_in_inference_mean", "boost_share_in_inference", "gr_active_pct_in_inference_mean",
           "sms_active_pct_in_inference_mean"]
KEY = ["model", "config", "mode"]


def medians(path, models=None):
    t = pd.read_csv(path)
    t = t[t["config"].isin(["webgpu", "webgl"])]
    if models:
        t = t[t["model"].isin(models)]
    g = t.groupby(KEY)
    out = g[METRICS].median()
    out["captures"] = g.size()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=str(layout.run_dir("chrome", "full") / "analysis" / "gpu_trace" / "nsys_summary.csv"))
    ap.add_argument("--candidate", default=str(layout.run_dir("firefox", "gpu_trace") / "analysis" / "nsys_summary.csv"))
    ap.add_argument("--out", default=None, help="default: <candidate folder>")
    ap.add_argument("--models", nargs="+", default=None, help="default: the models present in the candidate summary")
    args = ap.parse_args()
    cand_all = pd.read_csv(args.candidate)
    models = args.models or sorted(cand_all["model"].unique())
    b, c = medians(args.baseline, models), medians(args.candidate, models)
    j = b.join(c, lsuffix="_base", rsuffix="_cand", how="inner")
    res = pd.DataFrame(index=j.index)
    res["page_ms_base"], res["page_ms_cand"] = j["page_p50_ms_base"], j["page_p50_ms_cand"]
    res["page_ratio"] = j["page_p50_ms_cand"] / j["page_p50_ms_base"]
    for m, short in (("gpc_clock_mhz_in_inference_mean", "clock_mhz"), ("boost_share_in_inference", "boost_share"),
                     ("gr_active_pct_in_inference_mean", "gr_active_pct"), ("sms_active_pct_in_inference_mean", "sms_active_pct")):
        res[f"{short}_base"], res[f"{short}_cand"] = j[f"{m}_base"], j[f"{m}_cand"]
        res[f"{short}_diff"] = j[f"{m}_cand"] - j[f"{m}_base"]
    res["captures_base"], res["captures_cand"] = j["captures_base"], j["captures_cand"]
    out = Path(args.out) if args.out else Path(args.candidate).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    res.round(3).to_csv(out / "gpu_trace_compare.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res[["page_ms_base", "page_ms_cand", "page_ratio", "clock_mhz_base", "clock_mhz_cand", "boost_share_base",
                   "boost_share_cand", "gr_active_pct_base", "gr_active_pct_cand"]].round(2).to_string())
    print(f"\n{len(res)} (model, config, task) cells -> {out / 'gpu_trace_compare.csv'}")


if __name__ == "__main__":
    main()
