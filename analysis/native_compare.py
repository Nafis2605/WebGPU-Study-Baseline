"""
Native TensorFlow (CUDA / CPU) vs the browser backends, on the same models, weights and input bytes.
(Port of the sibling study's file; stage names mapped: infer_native(+_why), infer_latency, infer_why,
infer_native_correctness, infer_native_batch. Training-step comparisons are in summarize.py --series train.)

Writes <out>/:
  native_latency_summary.csv  per (model, native config): n, p50, p95, p99 (all trials pooled)
  native_vs_browser.csv       per model and comparison: browser (or variant) p50 / native p50, with a
                              bootstrap 95% CI (ratio of medians). Comparisons:
                                webgpu / tf-cuda           WebGPU vs native CUDA, TensorFlow defaults (TF32)
                                webgpu / tf-cuda-fp32      precision-matched (both full FP32)
                                webgl / tf-cuda, webgl / tf-cuda-fp32
                                wasm-8t / tf-cpu           browser CPU (WASM 8 threads) vs native CPU (8 threads)
                                webgpu-gpuio / tf-cuda-resident   GPU-resident inputs on both sides
                                tf-cuda-fp32 / tf-cuda     what TF32 tensor-core math buys natively
  cold_start.csv              per model: browser time to first result (init + fetch + build + first run)
                              and first-run ms (WebGPU, WebGL, WASM 8T) vs native (TF import/init +
                              graph load + build + first run, tf-cuda)
  native_correctness.csv      per (model, native config): worst normalised max error and min cosine vs the
                              TF.js cpu reference on identical input bytes
  native_batch_summary.csv    per (model, native config, batch): per-sample p50, amortisation
  trace_compare.csv           traced (Nsight Systems) WebGPU vs native CUDA per model: GPU busy ms and share
                              per inference, dispatches vs kernels per inference, traced p50
  memory_compare.csv          native peak GPU memory (TF allocator) vs WebGPU pool after runs

Usage:
    python analysis/native_compare.py --study benchmark_results/final --out benchmark_results/final/analysis/native
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats import ratio_of_medians_ci

KEY = ["trial", "model", "backend"]
COMPARISONS = [("webgpu", "tf-cuda"), ("webgpu", "tf-cuda-fp32"), ("webgl", "tf-cuda"), ("webgl", "tf-cuda-fp32"),
               ("wasm-8t", "tf-cpu"), ("webgpu-gpuio", "tf-cuda-resident"), ("tf-cuda-fp32", "tf-cuda")]


def stage_dir(study, stage):
    f = Path(study) / stage / "stage_done.json"
    return Path(json.loads(f.read_text(encoding="utf-8"))["result_dir"]) if f.exists() else None


def measured(d, phase="measure"):
    if d is None or not (d / "runs.csv").exists():
        return pd.DataFrame()
    s = pd.read_csv(d / "sessions.csv")
    ok = s[s["status"] == "ok"]
    if "instrumented" in ok:
        ok = ok[~ok["instrumented"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    r = pd.read_csv(d / "runs.csv", low_memory=False).merge(ok[KEY], on=KEY)
    return r[r["phase"] == phase]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    study, out = Path(args.study), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nat = pd.concat([measured(stage_dir(study, "infer_native")), measured(stage_dir(study, "infer_native_why"))],
                    ignore_index=True)
    lat = measured(stage_dir(study, "infer_latency"))
    io = measured(stage_dir(study, "infer_why"))
    runs = pd.concat([nat, lat, io[io["backend"].isin(["webgpu-gpuio"])] if len(io) else io], ignore_index=True)
    samples = runs.groupby(["model", "backend"])["dur_ms"].apply(np.asarray)

    t = nat.groupby(["model", "backend"])["dur_ms"].agg(
        n="size", p50_ms="median", p95_ms=lambda x: x.quantile(0.95), p99_ms=lambda x: x.quantile(0.99)).reset_index()
    t.to_csv(out / "native_latency_summary.csv", index=False)

    rows = []
    for m in sorted({k[0] for k in samples.index}):
        for num, den in COMPARISONS:
            if (m, num) in samples.index and (m, den) in samples.index:
                a, b = samples[(m, num)], samples[(m, den)]
                r, lo, hi = ratio_of_medians_ci(a, b)
                rows.append({"model": m, "numerator": num, "denominator": den, "numerator_p50_ms": float(np.median(a)),
                             "denominator_p50_ms": float(np.median(b)), "ratio": r, "ci_lo": lo, "ci_hi": hi})
    vs = pd.DataFrame(rows)
    vs.to_csv(out / "native_vs_browser.csv", index=False)

    # Cold start
    cold = []
    init = study / "analysis" / "init_reuse" / "init_summary.csv"
    bi = pd.read_csv(init) if init.exists() else pd.DataFrame()
    nd = stage_dir(study, "infer_native")
    ns = pd.read_csv(nd / "sessions.csv") if nd and (nd / "sessions.csv").exists() else pd.DataFrame()
    if len(ns):
        ns = ns[ns["status"] == "ok"]
        ns["ttfr_ms"] = ns[["backend_init_ms", "fetch_ms", "session_create_ms", "first_run_ms"]].sum(axis=1, min_count=4)
        nmed = ns.groupby(["model", "backend"])[["backend_init_ms", "fetch_ms", "session_create_ms", "first_run_ms",
                                                 "ttfr_ms"]].median()
        for m in sorted(ns["model"].unique()):
            row = {"model": m}
            if (m, "tf-cuda") in nmed.index:
                v = nmed.loc[(m, "tf-cuda")]
                row.update({"native_tf_cuda_init_ms": v["backend_init_ms"], "native_tf_cuda_load_build_ms":
                            v["fetch_ms"] + v["session_create_ms"], "native_tf_cuda_first_run_ms": v["first_run_ms"],
                            "native_tf_cuda_ttfr_ms": v["ttfr_ms"]})
            for c in ("webgpu", "webgl", "wasm-8t"):
                g = bi[(bi["model"] == m) & (bi["backend"] == c)] if len(bi) else bi
                if len(g):
                    row[f"{c}_ttfr_ms"] = float(g["time_to_first_result_ms_median"].iloc[0])
                    row[f"{c}_first_run_ms"] = float(g["first_run_ms_median"].iloc[0])
            cold.append(row)
    pd.DataFrame(cold).to_csv(out / "cold_start.csv", index=False)

    # Correctness
    cd = stage_dir(study, "infer_native_correctness")
    if cd and (cd / "correctness.csv").exists():
        c = pd.read_csv(cd / "correctness.csv")
        c = c[c["status"] == "ok"] if "status" in c else c
        agg = c.groupby(["model", "backend"]).agg(norm_max_err=("norm_max_err", "max"), cosine_min=("cosine", "min"),
                                                  inputs_identical=("inputs_identical", "all"),
                                                  top1=("top1_agreement", "min")).reset_index()
        agg.to_csv(out / "native_correctness.csv", index=False)

    # Batch
    bd = stage_dir(study, "infer_native_batch")
    br = measured(bd, "batch")
    if len(br):
        br = br.assign(per_sample_ms=br["dur_ms"] / br["batch"])
        bt = br.groupby(["model", "backend", "batch"]).agg(n=("dur_ms", "size"), p50_ms=("dur_ms", "median"),
                                                            per_sample_p50_ms=("per_sample_ms", "median")).reset_index()
        first = bt.sort_values("batch").groupby(["model", "backend"])["per_sample_p50_ms"].transform("first")
        bt["amortisation_vs_batch1"] = first / bt["per_sample_p50_ms"]
        bt.to_csv(out / "native_batch_summary.csv", index=False)

    # Traces
    ts = study / "analysis" / "gpu_trace" / "nsys_summary.csv"
    if ts.exists():
        tr = pd.read_csv(ts)
        tr = tr[tr["mode"] == "closed"]
        cols = ["page_p50_ms", "gpu_busy_ms_per_inference", "gpu_busy_frac_in_inference", "dispatch_per_inference",
                "kernels_per_inference", "memcpy_per_inference", "lead_ms_p50", "tail_ms_p50",
                "gpc_clock_mhz_in_inference_mean", "sms_active_pct_in_inference_mean"]
        g = tr.groupby(["model", "config"])[[x for x in cols if x in tr]].median().reset_index()
        g = g[g["config"].isin(["webgpu", "native-tf-cuda", "native-tf-cuda-fp32"])]
        g.to_csv(out / "trace_compare.csv", index=False)

    # Memory
    ms = study / "analysis" / "memory" / "memory_summary.csv"
    if len(ns) and "native_gpu_memory_peak_bytes" in ns:
        nm = ns[ns["backend"] == "tf-cuda"].groupby("model")["native_gpu_memory_peak_bytes"].median() / 2**20
        mem = nm.rename("native_tf_cuda_peak_mb").reset_index()
        if ms.exists():
            bm = pd.read_csv(ms)
            bm = bm[bm["backend"] == "webgpu"][["model", "pool_allocated_after_runs_mb", "weights_mb"]]
            mem = mem.merge(bm, on="model", how="left")
        mem.to_csv(out / "memory_compare.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        if len(vs):
            print(vs.pivot_table(index="model", columns=["numerator", "denominator"], values="ratio").round(2).to_string())
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
