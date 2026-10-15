"""
Preliminary analysis tables and figures (study_report/PRELIMINARY_ANALYSIS.md).

Inputs are development data, not the locked study:
  P  = benchmark_results/preflight           (every stage, 1 trial, smoke record counts; pre warm-up floor)
  DS = benchmark_results/dev_warmfloor/smoke (17 warm-up stages, 1 trial, 20 measured records; warm-up floor)
  DS2= benchmark_results/dev_warmfloor/smoke2 (native training stages re-run after the batch-order fix)
  DF = benchmark_results/dev_warmfloor/full  (+ full2): 13 models, 2 trials, full measurement settings
Profile summaries were re-run with the fixed analysis into scratch dirs given by --prof-i / --prof-t.

Usage: python study_report/prelim/make_prelim.py --prof-i DIR --prof-t DIR --energy CSV
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BR = ROOT / "benchmark_results"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "tables"
ORDER = [f + str(i) for f in "DCRLGT" for i in range(1, 6)]
FAM = {"D": "MLP", "C": "CNN", "R": "SimpleRNN", "L": "LSTM", "G": "GRU", "T": "Transformer"}
COL = {"webgpu": "#1f77b4", "webgl": "#ff7f0e", "wasm-8t": "#2ca02c", "tf-cuda-fp32": "#d62728", "tf-cpu": "#9467bd"}


def first(pattern):
    g = sorted(glob.glob(str(pattern)))
    return Path(g[0]) if g else None


def runs(path):
    return pd.read_csv(path / "runs.csv", low_memory=False)


def save(df, name):
    df.to_csv(TAB / f"{name}.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof-i", required=True)
    ap.add_argument("--prof-t", required=True)
    ap.add_argument("--energy", required=True)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    P, DS, DS2, DF = BR / "preflight", BR / "dev_warmfloor" / "smoke", BR / "dev_warmfloor" / "smoke2", BR / "dev_warmfloor" / "full"

    # ---------------- inference B=1, all 30 models (DS) ----------------
    inf = pd.read_csv(DS / "analysis" / "latency" / "infer_summary.csv")
    p50 = inf.pivot(index="model", columns="backend", values="p50_ms").reindex(ORDER)
    ex = inf.pivot(index="model", columns="backend", values="exec_ms_p50").reindex(ORDER)
    rd = inf.pivot(index="model", columns="backend", values="read_ms_p50").reindex(ORDER)
    t1 = p50[["wasm-8t", "webgl", "webgpu", "tf-cuda-fp32", "tf-cpu"]].copy()
    t1["webgpu/webgl"] = t1.webgpu / t1.webgl
    t1["webgpu/wasm"] = t1.webgpu / t1["wasm-8t"]
    t1["webgpu/cuda"] = t1.webgpu / t1["tf-cuda-fp32"]
    t1["webgpu_read_share"] = rd.webgpu / p50.webgpu
    save(t1.round(3), "t1_infer_b1")

    # profile counters (fixed profile analysis)
    pi = pd.read_csv(Path(a.prof_i) / "profile_summary.csv")
    disp = pi[pi.backend == "webgpu"].set_index("model").dispatches.reindex(ORDER)
    draws = pi[pi.backend == "webgl"].set_index("model").draws.reindex(ORDER)
    kern = pi[pi.backend == "webgpu"].set_index("model").kernels_run.reindex(ORDER)
    feats = pd.read_csv(P / "analysis" / "features" / "model_features.csv")
    feats = feats[feats.batch == 1].drop_duplicates("model").set_index("model").reindex(ORDER)
    macs = feats.macs / 1e6

    def ols(X, y):
        X = np.column_stack([np.ones(len(y))] + X)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return b, 1 - r.var() / y.var()

    reg = []
    for be, cnt in (("webgpu", disp), ("webgl", draws), ("wasm-8t", None)):
        y = p50[be].to_numpy(float)
        for name, X in (("kernels", [kern.to_numpy(float)]), ("MMACs", [macs.to_numpy(float)]),
                        ("kernels+MMACs", [kern.to_numpy(float), macs.to_numpy(float)])):
            b, r2 = ols(X, y)
            reg.append({"backend": be, "predictors": name, "R2": r2, "intercept_ms": b[0],
                        "per_kernel_us": b[1] * 1000 if "kernels" in name else np.nan,
                        "per_MMAC_us": (b[-1] * 1000) if "MMACs" in name else np.nan})
        if cnt is not None:
            b, r2 = ols([cnt.to_numpy(float)], y)
            reg.append({"backend": be, "predictors": "dispatches" if be == "webgpu" else "draws", "R2": r2,
                        "intercept_ms": b[0], "per_kernel_us": b[1] * 1000, "per_MMAC_us": np.nan})
    save(pd.DataFrame(reg).round(4), "t2_latency_regression")

    # Fig 1: WebGPU latency vs dispatches and vs MACs
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for fam, mk in zip("DCRLGT", "osv^Dx"):
        idx = [m for m in ORDER if m[0] == fam]
        ax[0].scatter(disp[idx], p50.webgpu[idx], marker=mk, label=FAM[fam])
        ax[1].scatter(macs[idx], p50.webgpu[idx], marker=mk, label=FAM[fam])
    b, r2 = ols([disp.to_numpy(float)], p50.webgpu.to_numpy(float))
    xs = np.linspace(0, disp.max() * 1.05, 50)
    ax[0].plot(xs, b[0] + b[1] * xs, "k--", lw=1, label=f"{b[0]:.2f} ms + {b[1]*1000:.1f} us/dispatch, R2={r2:.3f}")
    ax[0].set_xlabel("WebGPU dispatches per inference")
    ax[0].set_ylabel("WebGPU p50 latency, batch 1 (ms)")
    ax[0].legend(fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("MACs per inference (millions, log)")
    ax[1].set_title("same latency vs compute: R2 = %.2f" % [r["R2"] for r in reg if r["backend"] == "webgpu" and r["predictors"] == "MMACs"][0], fontsize=9)
    fig.suptitle("Fig 1. WebGPU batch-1 inference latency is set by the dispatch count, not by compute")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_dispatch_bound.png", dpi=150)
    plt.close(fig)

    # Fig 2: exec vs read decomposition, WebGPU and WebGL
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(ORDER))
    w = 0.4
    ax.bar(x - w / 2, ex.webgpu, w, color="#1f77b4", label="WebGPU exec (JS encode + submit)")
    ax.bar(x - w / 2, rd.webgpu, w, bottom=ex.webgpu, color="#aec7e8", label="WebGPU read (wait for result)")
    ax.bar(x + w / 2, ex.webgl, w, color="#ff7f0e", label="WebGL exec")
    ax.bar(x + w / 2, rd.webgl, w, bottom=ex.webgl, color="#ffbb78", label="WebGL read")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("ms (log)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Fig 2. Batch-1 inference split into exec and read (p50 of each part)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_exec_read.png", dpi=150)
    plt.close(fig)

    # per-kernel CPU cost by op (WebGPU vs WebGL)
    tk = pd.read_csv(Path(a.prof_i) / "top_kernels.csv")
    g = tk[tk.backend.isin(["webgpu", "webgl"])].groupby(["name", "backend"]).apply(
        lambda d: pd.Series({"calls": d.calls_per_run.sum(), "cpu_us_per_call": d.cpu_ms_per_run.sum() / d.calls_per_run.sum() * 1000}),
        include_groups=False).unstack()
    g = g[g[("calls", "webgpu")] > 50].sort_values(("calls", "webgpu"), ascending=False)
    g[("ratio", "webgpu/webgl")] = g[("cpu_us_per_call", "webgpu")] / g[("cpu_us_per_call", "webgl")]
    save(g.round(2), "t3_cpu_cost_per_kernel")

    # ---------------- training, all 30 (DS + DS2 native) ----------------
    ta = pd.read_csv(DS / "analysis" / "latency" / "train_summary.csv")
    tb = pd.read_csv(DS2 / "analysis" / "latency" / "train_summary.csv")
    tr = pd.concat([ta[~ta.backend.isin(["tf-cpu", "tf-cuda-fp32"])], tb])
    tp50 = tr.pivot(index="model", columns="backend", values="p50_ms").reindex(ORDER)
    trd = tr.pivot(index="model", columns="backend", values="read_ms_p50").reindex(ORDER)
    t4 = tp50[["wasm-8t", "webgl", "webgpu", "tf-cuda-fp32", "tf-cpu"]].copy()
    t4["webgpu/webgl"] = t4.webgpu / t4.webgl
    t4["webgpu/wasm"] = t4.webgpu / t4["wasm-8t"]
    t4["webgpu/cuda"] = t4.webgpu / t4["tf-cuda-fp32"]
    t4["webgpu_read_share"] = trd.webgpu / tp50.webgpu
    save(t4.round(3), "t4_train_step")

    # family geomeans
    def gm(s):
        s = s.dropna()
        return float(np.exp(np.log(s).mean())) if len(s) else np.nan
    fam_rows = []
    for fam in "DCRLGT":
        idx = [m for m in ORDER if m[0] == fam]
        fam_rows.append({"family": FAM[fam],
                         "infer webgpu/webgl": gm(t1.loc[idx, "webgpu/webgl"]), "infer webgpu/wasm": gm(t1.loc[idx, "webgpu/wasm"]),
                         "infer webgpu/cuda": gm(t1.loc[idx, "webgpu/cuda"]),
                         "train webgpu/webgl": gm(t4.loc[idx, "webgpu/webgl"]), "train webgpu/wasm": gm(t4.loc[idx, "webgpu/wasm"]),
                         "train webgpu/cuda": gm(t4.loc[idx, "webgpu/cuda"])})
    save(pd.DataFrame(fam_rows).set_index("family").round(3), "t5_family_geomean_ratios")

    # Fig 3: ratio heatmap
    hm = pd.DataFrame({"inf vs WebGL": t1["webgpu/webgl"], "inf vs WASM": t1["webgpu/wasm"], "inf vs CUDA": t1["webgpu/cuda"],
                       "train vs WebGL": t4["webgpu/webgl"], "train vs WASM": t4["webgpu/wasm"], "train vs CUDA": t4["webgpu/cuda"]})
    fig, ax = plt.subplots(figsize=(7, 10))
    v = np.log2(hm.to_numpy(float))
    im = ax.imshow(v, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            val = hm.iat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}" if val < 10 else f"{val:.0f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels(hm.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER, fontsize=7)
    fig.colorbar(im, ax=ax, label="log2(WebGPU time / other time); blue = WebGPU faster")
    ax.set_title("Fig 3. WebGPU time relative to each runtime (p50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ratio_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------------- training phase split + CNN backprop filter ----------------
    kp = pd.read_csv(P / "analysis" / "train_phase" / "kernel_phase.csv")
    hp = pd.read_csv(P / "analysis" / "train_phase" / "hook_phase.csv")
    ph = kp[kp.backend == "webgpu"].set_index("model").reindex(ORDER)
    hh = hp[hp.backend == "webgpu"].set_index("model").reindex(ORDER)
    t6 = pd.DataFrame({"fwd_gpu_ms": ph.kernel_ms_forward, "bwd_gpu_ms": ph.kernel_ms_backward, "opt_gpu_ms": ph.kernel_ms_optimizer,
                       "opt_share_gpu": ph.kernel_share_optimizer, "kernels_fwd": ph.kernels_forward, "kernels_bwd": ph.kernels_backward,
                       "kernels_opt": ph.kernels_optimizer, "cpu_fwd_ms": hh.hooks_kernel_cpu_ms_forward, "cpu_bwd_ms": hh.hooks_kernel_cpu_ms_backward,
                       "cpu_opt_ms": hh.hooks_kernel_cpu_ms_optimizer, "dispatches": hh.webgpu_dispatches, "bind_groups": hh.webgpu_bind_groups_created,
                       "forwarded_to_cpu": hh.hooks_forwarded_kernels})
    save(t6.round(3), "t6_webgpu_train_phases")

    kc = pd.read_csv(sorted(glob.glob(str(P / "train_profile" / "*" / "kernels.csv")))[0])
    kc = kc[(kc.source == "tf.profile") & kc.name.isin(["Conv2DBackpropFilter", "Conv2D", "Conv2DBackpropInput"]) & kc.model.str.startswith("C")]
    kc = kc[kc.backend.isin(["webgpu", "webgl"])]
    kc["layer"] = np.where(kc.input_shapes.str.contains(",3]") | kc.input_shapes.str.contains("32,32,3"), "conv_0", "conv_1")
    t7 = kc.groupby(["model", "backend", "name", "layer"]).kernel_ms.median().unstack(["backend"])
    save(t7.round(3), "t7_cnn_conv_kernels")

    tkt = pd.read_csv(Path(a.prof_t) / "top_kernels.csv")
    cc = tkt[tkt.model.str.startswith("C") & tkt.backend.isin(["webgpu", "webgl"])]
    tot = cc.groupby(["model", "backend"]).ms_per_run.sum()
    bf = cc[cc.name == "Conv2DBackpropFilter"].set_index(["model", "backend"]).ms_per_run
    fig, ax = plt.subplots(figsize=(8, 3.8))
    models = [f"C{i}" for i in range(1, 6)]
    x = np.arange(5)
    for k, (be, off) in enumerate((("webgpu", -0.2), ("webgl", 0.2))):
        b_ = np.array([bf.get((m, be), np.nan) for m in models])
        t_ = np.array([tot.get((m, be), np.nan) for m in models])
        ax.bar(x + off, b_, 0.4, color=COL[be], label=f"{be}: Conv2DBackpropFilter")
        ax.bar(x + off, t_ - b_, 0.4, bottom=b_, color=COL[be], alpha=0.35, label=f"{be}: all other kernels")
    nat = t4.loc[models, "tf-cuda-fp32"].to_numpy(float)
    ax.plot(x, nat, "k_", ms=25, mew=2, label="native tf-cuda-fp32 whole step (wall)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("GPU kernel ms per training step")
    ax.legend(fontsize=7)
    ax.set_title("Fig 4. CNN training: one kernel is 74-85 % of WebGPU's GPU time", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cnn_backprop_filter.png", dpi=150)
    plt.close(fig)

    # ---------------- cold start ----------------
    wb = pd.read_csv(DS / "analysis" / "warmup" / "warmup_by_launch.csv")
    cs = {}
    for mode in ("latency", "train"):
        x_ = wb[(wb["mode"] == mode) & wb.backend.isin(["wasm-8t", "webgl", "webgpu", "tf-cuda-fp32", "tf-cpu"])]
        for c in ("first_ms", "time_to_first_result_ms", "first_over_steady", "backend_init_ms", "steps_to_steady", "cold_overhead_ms"):
            cs[(mode, c)] = x_.pivot_table(index="model", columns="backend", values=c, aggfunc="median").reindex(ORDER)
    t8 = pd.concat({f"{m}:{c}": d for (m, c), d in cs.items()}, axis=1)
    save(t8.round(2), "t8_cold_start")
    fc = pd.read_csv(P / "analysis" / "init_reuse" / "first_run_counters.csv")
    fc = fc[fc.kind == "first_run"].pivot_table(index="model", columns="backend", values="pipelines_or_programs", aggfunc="median").reindex(ORDER)
    save(fc, "t9_first_run_programs")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for be in ("wasm-8t", "webgl", "webgpu", "tf-cuda-fp32"):
        ax[0].plot(ORDER, cs[("latency", "time_to_first_result_ms")][be], "o-", ms=3, color=COL[be], label=be)
        ax[1].plot(ORDER, cs[("train", "first_ms")][be], "o-", ms=3, color=COL[be], label=be)
    for k_, t_ in enumerate(("inference: time to first result (ms)", "training: first step (ms)")):
        ax[k_].set_yscale("log")
        ax[k_].set_title(t_, fontsize=9)
        ax[k_].tick_params(axis="x", labelsize=6, rotation=90)
        ax[k_].legend(fontsize=7)
    fig.suptitle("Fig 5. Cold start per model (fresh browser / process per launch)")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_cold_start.png", dpi=150)
    plt.close(fig)

    # shapes
    sr = pd.read_csv(DS / "analysis" / "init_reuse" / "shape_reuse_summary.csv")
    sr = sr.pivot_table(index=["model", "shape_value"], columns="backend", values="first_seen_median")
    save(sr.round(2), "t10_shape_first_seen")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for be in ("webgl", "webgpu"):
        for m, mk in zip(["D3", "C3", "R3", "L3", "G3", "T3"], "osv^Dx"):
            s = sr.loc[m][be].dropna()
            ax.plot(s.index, s.values, marker=mk, color=COL[be], alpha=0.8, label=f"{m} {be}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size seen for the first time (after warm-up at batch 1)")
    ax.set_ylabel("first inference at that shape (ms)")
    ax.legend(fontsize=6, ncol=4)
    ax.set_title("Fig 6. Cost of a never-seen input shape: WebGL recompiles, WebGPU mostly does not", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_shapes.png", dpi=150)
    plt.close(fig)

    # lifecycle transfer
    rows = []
    for st in ("warmup_lifecycle_if", "warmup_lifecycle_tf"):
        r = runs(first(P / st / "*"))
        f = r.sort_values("i").groupby(["model", "backend", "phase"]).dur_ms.first().unstack("phase")
        f["order"] = st
        rows.append(f)
    lc = pd.concat(rows)
    cold_inf = cs[("latency", "first_ms")]
    cold_tr = cs[("train", "first_ms")]
    lt = []
    for (m, be), r in lc.iterrows():
        if be == "wasm-8t":
            continue
        if r["order"] == "warmup_lifecycle_if":
            lt.append({"model": m, "backend": be, "transfer": "inference -> first training step",
                       "first_ms_after": r["lc_train_warmup"], "first_ms_cold": cold_tr.loc[m, be]})
        else:
            lt.append({"model": m, "backend": be, "transfer": "training -> first inference",
                       "first_ms_after": r["lc_infer_warmup"], "first_ms_cold": cold_inf.loc[m, be]})
    lt = pd.DataFrame(lt)
    lt["cold/after"] = lt.first_ms_cold / lt.first_ms_after
    save(lt.set_index(["transfer", "model", "backend"]).sort_index().round(2), "t11_lifecycle_transfer")

    # ---------------- memory ----------------
    ms = pd.read_csv(P / "analysis" / "memory" / "memory_summary.csv")
    nt = pd.read_csv(sorted(glob.glob(str(DS2 / "native_train" / "*" / "sessions.csv")))[0])
    npk = nt[nt.backend == "tf-cuda-fp32"].groupby("model").native_gpu_memory_peak_bytes.max() / 2**20
    ni = pd.read_csv(sorted(glob.glob(str(DS / "infer_native" / "*" / "sessions.csv")))[0])
    npi = ni[ni.backend == "tf-cuda-fp32"].groupby("model").native_gpu_memory_peak_bytes.max() / 2**20
    rows = []
    for mode, natv in (("profile", npi), ("train_profile", npk)):
        x_ = ms[ms["mode"] == mode]
        gpu = x_[x_.backend == "webgpu"].set_index("model").reindex(ORDER)
        gl = x_[x_.backend == "webgl"].set_index("model").reindex(ORDER)
        wa = x_[x_.backend == "wasm-8t"].set_index("model").reindex(ORDER)
        d = pd.DataFrame({"weights_mb": gpu.weights_mb, "persistent_state_mb": gpu.pool_used_after_runs_mb,
                          "webgpu_pool_mb": gpu.pool_allocated_after_runs_mb, "webgl_pool_mb": gl.pool_allocated_after_runs_mb,
                          "wasm_heap_mb": wa.wasm_heap_after_runs_mb, "native_cuda_peak_mb": natv.reindex(ORDER),
                          "webgpu_retained_after_dispose_mb": gpu.retained_pool_after_dispose_mb})
        d["webgpu/native"] = d.webgpu_pool_mb / d.native_cuda_peak_mb
        d["webgpu_pool/state"] = d.webgpu_pool_mb / d.persistent_state_mb
        d.columns = pd.MultiIndex.from_product([[mode], d.columns])
        rows.append(d)
    t12 = pd.concat(rows, axis=1)
    save(t12.round(2), "t12_memory")
    tm = t12["train_profile"]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(ORDER))
    ax.bar(x - 0.3, tm.persistent_state_mb, 0.2, color="gray", label="weights + Adam state + grads (live after step)")
    ax.bar(x - 0.1, tm.native_cuda_peak_mb, 0.2, color=COL["tf-cuda-fp32"], label="native TF CUDA peak")
    ax.bar(x + 0.1, tm.webgpu_pool_mb, 0.2, color=COL["webgpu"], label="WebGPU pool (= retained after dispose)")
    ax.bar(x + 0.3, tm.webgl_pool_mb, 0.2, color=COL["webgl"], label="WebGL pool")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=7)
    ax.set_ylabel("MB (log)")
    ax.legend(fontsize=7)
    ax.set_title("Fig 7. GPU memory of one training step (batch 64 / 32)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_train_memory.png", dpi=150)
    plt.close(fig)

    # ---------------- readback distribution ----------------
    rf = runs(first(DF / "infer" / "*"))
    rf = rf[rf.phase == "measure"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
    for k_, m in enumerate(("D1", "C3", "T3")):
        for be in ("webgl", "webgpu"):
            v = rf[(rf.model == m) & (rf.backend == be)].read_ms
            ax[k_].hist(v, bins=np.arange(0, 14, 0.25), color=COL[be], alpha=0.6, label=be, density=True)
        ax[k_].set_title(m, fontsize=9)
        ax[k_].set_xlabel("read_ms (await data())")
        ax[k_].legend(fontsize=7)
    fig.suptitle("Fig 8. Result readback time at batch 1 (full settings, 2 trials)")
    fig.tight_layout()
    fig.savefig(FIG / "fig8_readback_hist.png", dpi=150)
    plt.close(fig)
    rs = rf.groupby(["model", "backend"]).read_ms.describe(percentiles=[0.01, 0.5, 0.85, 0.95])
    rs["frac_lt_1ms"] = rf.groupby(["model", "backend"]).read_ms.apply(lambda s: (s < 1).mean())
    rs["frac_gt_7ms"] = rf.groupby(["model", "backend"]).read_ms.apply(lambda s: (s > 7).mean())
    save(rs.round(3), "t13_readback_distribution")

    # ---------------- Nsight ----------------
    ns = pd.read_csv(P / "analysis" / "gpu_trace" / "nsys_summary.csv")
    t14 = ns[["model", "config", "mode", "page_p50_ms", "gpu_busy_ms_per_inference", "gpu_busy_frac_in_inference", "lead_ms_p50", "tail_ms_p50",
              "idle_gap_ms_p50", "dispatch_per_inference", "execute_command_lists_per_inference", "map_per_inference",
              "kernels_per_inference", "gpc_clock_mhz_in_inference_mean"]].set_index(["model", "config", "mode"]).sort_index()
    save(t14.round(3), "t14_nsight")

    # ---------------- responsiveness ----------------
    rows = []
    for sub in ("infer_responsiveness", "train_responsiveness"):
        t = pd.read_csv(DS / "analysis" / sub / "responsiveness_summary.csv")
        t = t[t.phase != "baseline"]
        t["pillar"] = sub.split("_")[0]
        rows.append(t)
    t15 = pd.concat(rows)[["pillar", "model", "backend", "frame_interval_p95_ms", "missed_deadline_frac", "dropped_frames_per_s",
                           "click_input_delay_p50_ms", "click_input_delay_p95_ms", "loaf_per_s", "loaf_blocking_ms_per_s"]]
    save(t15.set_index(["pillar", "model", "backend"]).round(3), "t15_responsiveness")
    tt = t15[t15.pillar == "train"].set_index(["model", "backend"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ms_ = ["D3", "C3", "R3", "L3", "G3", "T3"]
    x = np.arange(len(ms_))
    for k_, (c, lab) in enumerate((("loaf_blocking_ms_per_s", "long-animation-frame blocking (ms per s)"),
                                    ("missed_deadline_frac", "fraction of frames that missed vsync"))):
        for j, be in enumerate(("webgl", "webgpu")):
            vals = [tt[c].get((m, be), np.nan) for m in ms_]
            ax[k_].bar(x + (j - 0.5) * 0.4, vals, 0.4, color=COL[be], label=be)
        ax[k_].set_xticks(x)
        ax[k_].set_xticklabels(ms_)
        ax[k_].set_title(lab, fontsize=9)
        ax[k_].legend(fontsize=7)
    fig.suptitle("Fig 9. Page responsiveness while training (animated page, scripted clicks)")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_train_responsiveness.png", dpi=150)
    plt.close(fig)

    # ---------------- fit epoch slowdown ----------------
    d = first(P / "train_fit" / "*")
    r = runs(d)
    gs = pd.read_csv(d / "gpu_samples.csv")
    us = pd.read_csv(d / "gpu_util_samples.csv")
    cm = pd.read_csv(d / "chrome_memory_samples.csv")
    rows = []
    for (m, be), x_ in r.groupby(["model", "backend"]):
        prev_cpu = None
        for phs in ("fit_warmup", "fit_measure"):
            y = x_[x_.phase == phs].iloc[0]
            t0, t1_ = y.t_start_epoch_ms, y.t_start_epoch_ms + y.dur_ms
            s = gs[(gs.model == m) & (gs.backend == be) & gs.t_epoch_ms.between(t0, t1_)]
            u = us[(us.model == m) & (us.backend == be) & us.t_epoch_ms.between(t0, t1_)]
            c = cm[(cm.model == m) & (cm.backend == be) & (cm.process_type == "renderer")]
            c0 = c[c.t_epoch_ms <= t0].cpu_user_s.max()
            c1 = c[c.t_epoch_ms <= t1_].cpu_user_s.max()
            rows.append({"model": m, "backend": be, "epoch": 1 if phs == "fit_warmup" else 2, "steps": int(y.steps),
                         "ms_per_step": y.dur_ms / y.steps, "gpu_util_pct": u.percent.mean(), "sm_clock_mhz": s.sm_clock_mhz.mean(),
                         "renderer_cpu_s": c1 - c0, "epoch_wall_s": y.dur_ms / 1000})
    fit = pd.DataFrame(rows)
    fit["renderer_cores_busy"] = fit.renderer_cpu_s / fit.epoch_wall_s
    save(fit.set_index(["model", "backend", "epoch"]).round(2), "t16_fit_epochs")
    cust = t4["webgpu"]
    f2 = fit[(fit.backend != "wasm-8t")].pivot_table(index="model", columns=["backend", "epoch"], values="ms_per_step")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ms_ = ["D3", "C3", "R3", "T3", "L3", "G3"]
    x = np.arange(len(ms_))
    for j, (be, ep, colr) in enumerate((("webgl", 1, "#ffbb78"), ("webgl", 2, COL["webgl"]), ("webgpu", 1, "#aec7e8"), ("webgpu", 2, COL["webgpu"]))):
        ax.bar(x + (j - 1.5) * 0.2, [f2[(be, ep)].get(m, np.nan) for m in ms_], 0.2, color=colr, label=f"{be} fit epoch {ep}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ms_)
    ax.set_ylabel("ms per step (log)")
    ax.legend(fontsize=7)
    ax.set_title("Fig 10. model.fit(yieldEvery 'never'): WebGPU's 2nd epoch degrades on dispatch-heavy models", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_fit_epochs.png", dpi=150)
    plt.close(fig)

    # ---------------- energy (DF) ----------------
    en = pd.read_csv(a.energy)
    en = en[en.window_ok >= 1]
    t17 = en.pivot_table(index=["src", "phase", "model"], columns="backend",
                         values=["energy_per_inference_mj", "dynamic_energy_per_inference_mj", "mean_power_w", "sm_clock_mean_mhz", "gpu_util_mean_pct"])
    save(t17.round(1), "t17_energy")

    # ---------------- batch scaling (DS) ----------------
    rows = []
    for st, ph in (("infer_batch", "batch"), ("train_batch", "train_batch")):
        r = runs(first(DS / st / "*"))
        rows.append(r[r.phase == ph].groupby(["model", "backend", "batch"]).dur_ms.median().rename(st))
    for st, ph, root in (("infer_native_batch", "batch", DS), ("native_train_batch", "train_batch", DS2)):
        r = runs(first(root / st / "*"))
        rows.append(r[r.phase == ph].groupby(["model", "backend", "batch"]).dur_ms.median().rename(st))
    bt = pd.concat(rows, axis=1)
    save(bt.round(3), "t18_batch")

    # ---------------- why-axis (P) ----------------
    rows = []
    for st, ph in (("infer_latency", "measure"), ("infer_why", "measure"), ("train", "train_measure"), ("train_why", "train_measure"),
                   ("train_sync_io", "train_measure")):
        r = runs(first(P / st / "*"))
        r = r[(r.phase == ph) & r.model.isin(["D3", "C3", "R3", "L3", "G3", "T3"])]
        rows.append(r.groupby(["model", "backend"]).agg(p50=("dur_ms", "median"), read=("read_ms", "median")).assign(stage=st))
    r = runs(first(P / "train_pipelined" / "*"))
    pp = r[r.phase == "train_pipelined"].groupby(["model", "backend"]).apply(
        lambda d: pd.Series({"p50": (d.dur_ms / d.window_steps).median(), "read": np.nan}), include_groups=False).assign(stage="train_pipelined")
    why = pd.concat(rows + [pp])
    save(why.round(3), "t19_why_configs")

    # readback micro
    save(pd.read_csv(P / "analysis" / "data_movement" / "readback_micro_summary.csv").round(3), "t20_readback_micro")
    # correctness
    ic = pd.read_csv(sorted(glob.glob(str(P / "infer_correctness" / "*" / "correctness.csv")))[0])
    nc = pd.read_csv(sorted(glob.glob(str(P / "infer_native_correctness" / "*" / "correctness.csv")))[0])
    corr = pd.concat([ic, nc]).groupby("backend").agg(max_abs_err=("max_abs_err", "max"), norm_max_err=("norm_max_err", "max"),
                                                       cosine_min=("cosine", "min"), top1_min=("top1_agreement", "min"), top5_min=("top5_overlap", "min"))
    save(corr, "t21_infer_correctness")
    tc = pd.read_csv(P / "analysis" / "train_correctness" / "train_correctness_vs_tf-cpu_tfjsadam.csv")
    save(tc.pivot_table(index="model", columns="config", values="grad_sketch_rel_max").reindex(ORDER), "t22_train_grad_rel")
    save(tc.pivot_table(index="model", columns="config", values="loss_rel_max_k").reindex(ORDER), "t23_train_loss_rel")
    print("written", sorted(os.listdir(TAB)), sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
