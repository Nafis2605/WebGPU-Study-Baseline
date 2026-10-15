"""
Three overview figures for the paper-level findings (same development data as make_prelim.py).

  fig11_latency_vs_kernels   batch-1 latency vs kernels per inference, WebGPU / WebGL / WASM, all 30 models
  fig12_cpu_cost_per_kernel  host CPU time per kernel call, WebGPU vs WebGL, by kernel type and by model family
  fig13_stability_load       between-trial CoV per backend (full settings) and latency inflation under open-loop load

Usage: python study_report/prelim/make_prelim_extra.py --prof-i DIR --sum-infer DIR --sum-train DIR
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "benchmark_results" / "dev_warmfloor" / "smoke"
OUT = Path(__file__).resolve().parent
FIG, TAB = OUT / "figures", OUT / "tables"
ORDER = [f + str(i) for f in "DCRLGT" for i in range(1, 6)]
FAM = {"D": "MLP", "C": "CNN", "R": "SimpleRNN", "L": "LSTM", "G": "GRU", "T": "Transformer"}
COL = {"webgpu": "#1f77b4", "webgl": "#ff7f0e", "wasm-8t": "#2ca02c", "tf-cuda-fp32": "#d62728", "tf-cpu": "#9467bd"}
NAME = {"webgpu": "WebGPU", "webgl": "WebGL", "wasm-8t": "WASM (8 threads)", "tf-cuda-fp32": "native CUDA", "tf-cpu": "native CPU"}


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, 1 - r.var() / y.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof-i", required=True)
    ap.add_argument("--sum-infer", required=True)
    ap.add_argument("--sum-train", required=True)
    a = ap.parse_args()

    # ---------- fig 11 ----------
    inf = pd.read_csv(DS / "analysis" / "latency" / "infer_summary.csv")
    p50 = inf.pivot(index="model", columns="backend", values="p50_ms").reindex(ORDER)
    pi = pd.read_csv(Path(a.prof_i) / "profile_summary.csv")
    kern = pi[pi.backend == "webgpu"].set_index("model").kernels_run.reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rows = []
    for be in ("webgpu", "webgl", "wasm-8t"):
        ax.scatter(kern, p50[be], s=22, color=COL[be], alpha=0.85, label=NAME[be])
        b, r2 = ols(kern.to_numpy(float), p50[be].to_numpy(float))
        rows.append({"backend": be, "intercept_ms": b[0], "us_per_kernel": b[1] * 1000, "R2": r2})
        if be != "wasm-8t":
            xs = np.linspace(kern.min(), kern.max(), 100)
            ax.plot(xs, b[0] + b[1] * xs, color=COL[be], lw=1.2, ls="--",
                    label=f"{NAME[be]} fit: {b[0]:.1f} ms + {b[1]*1000:.1f} us/kernel (R2 {r2:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("kernels per inference (log)")
    ax.set_ylabel("batch-1 inference latency, p50 (ms, log)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Fig 11. GPU backends track kernel count; WASM tracks compute\n(30 models, 6 families)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_latency_vs_kernels.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).round(4).to_csv(TAB / "t24_latency_vs_kernels_fit.csv", index=False)

    # ---------- fig 12 ----------
    tk = pd.read_csv(Path(a.prof_i) / "top_kernels.csv")
    g = tk[tk.backend.isin(["webgpu", "webgl"])].groupby(["name", "backend"])[["cpu_ms_per_run", "calls_per_run"]].sum()
    g["us"] = g.cpu_ms_per_run / g.calls_per_run * 1000
    us = g.us.unstack()
    calls = g.calls_per_run.unstack().webgpu
    us = us[calls > 50].assign(calls=calls[calls > 50]).sort_values("calls", ascending=False).drop(columns="calls")
    us = us.drop(index=[i for i in us.index if i == "Reshape"])  # metadata-only, no GPU work on either backend
    pi["cpu_us_per_kernel"] = pi.kernel_cpu_ms / pi.kernels_run * 1000
    fam = pi[pi.backend.isin(["webgpu", "webgl"])].assign(family=lambda d: d.model.str[0].map(FAM))
    famv = fam.groupby(["family", "backend"]).cpu_us_per_kernel.median().unstack()
    famv = famv.reindex([FAM[f] for f in "DCRLGT"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(us))
    for j, be in enumerate(("webgl", "webgpu")):
        ax[0].bar(x + (j - 0.5) * 0.4, us[be], 0.4, color=COL[be], label=NAME[be])
    for i, (n, r) in enumerate(us.iterrows()):
        ax[0].text(i, max(r.webgl, r.webgpu) + 0.6, f"{r.webgpu / r.webgl:.1f}x", ha="center", fontsize=7)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(us.index, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("host CPU time per kernel call (us)")
    ax[0].set_title("(a) by kernel type (label: WebGPU / WebGL)", fontsize=9)
    ax[0].legend(fontsize=8)
    x = np.arange(len(famv))
    for j, be in enumerate(("webgl", "webgpu")):
        ax[1].bar(x + (j - 0.5) * 0.4, famv[be], 0.4, color=COL[be], label=NAME[be])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(famv.index, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("CPU us per kernel (median over sizes)")
    ax[1].set_title("(b) by model family: many small kernels -> WebGPU costs more", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.suptitle("Fig 12. Host-side cost of issuing one kernel, WebGPU vs WebGL (TF.js, batch-1 inference)")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_cpu_cost_per_kernel.png", dpi=150)
    plt.close(fig)
    us.assign(ratio=us.webgpu / us.webgl).round(2).to_csv(TAB / "t25_cpu_us_per_kernel_type.csv")
    famv.assign(ratio=famv.webgpu / famv.webgl).round(2).to_csv(TAB / "t26_cpu_us_per_kernel_family.csv")

    # ---------- fig 13 ----------
    rows = []
    for pillar, d in (("inference", a.sum_infer), ("training", a.sum_train)):
        t = pd.read_csv(Path(d) / f"{'infer' if pillar == 'inference' else 'train'}_summary.csv")
        for _, r in t.iterrows():
            rows.append({"pillar": pillar, "model": r.model, "backend": r.backend, "between_trial_cov": r.between_trial_cov,
                         "tail_p99_p50": r.tail_ratio_p99_p50})
    st = pd.DataFrame(rows)
    st.round(4).to_csv(TAB / "t27_stability_full.csv", index=False)
    ld = pd.read_csv(DS / "analysis" / "load" / "load_summary.csv")
    cl = inf.set_index(["model", "backend"]).p50_ms
    ld["closed_p50"] = [cl.get((m, b), np.nan) for m, b in zip(ld.model, ld.backend)]
    ld["inflation"] = ld.service_p50_ms / ld.closed_p50
    ld[["model", "backend", "rate", "service_p50_ms", "closed_p50", "inflation", "deadline_miss_rate"]].round(3).to_csv(
        TAB / "t28_load_inflation.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bes = ["webgpu", "webgl", "wasm-8t", "tf-cuda-fp32", "tf-cpu"]
    pos = 0
    ticks, labs = [], []
    for k, pillar in enumerate(("inference", "training")):
        for j, be in enumerate(bes):
            v = st[(st.pillar == pillar) & (st.backend == be)].between_trial_cov.dropna().to_numpy()
            if not len(v):
                continue
            xpos = pos + j
            ax[0].scatter(np.full(len(v), xpos) + np.linspace(-0.15, 0.15, len(v)), v, color=COL[be], s=18)
            ax[0].hlines(np.median(v), xpos - 0.3, xpos + 0.3, color="k", lw=1.5)
            ticks.append(xpos)
            labs.append(NAME[be])
        ax[0].annotate(pillar, xy=(pos + 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -12),
                       textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        pos += len(bes) + 1
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(labs, rotation=35, ha="right", fontsize=7)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("between-trial CoV of the median (log)")
    ax[0].set_title("(a) run-to-run stability (full settings, 2 trials; bar = median)", fontsize=9)
    lw = ld[ld.backend.isin(["webgpu", "webgl", "wasm-8t"])]
    feasible = lw[(lw.deadline_miss_rate == 0)]
    for be in ("webgpu", "webgl", "wasm-8t"):
        for rate, mk in ((5, "o"), (50, "^")):
            v = feasible[(feasible.backend == be) & (feasible.rate == rate)]
            ax[1].scatter(v.closed_p50, v.inflation, color=COL[be], marker=mk, s=30,
                          label=f"{NAME[be]}, {rate} req/s")
    ax[1].axhline(1, color="gray", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("closed-loop latency p50 (ms, log)")
    ax[1].set_ylabel("latency inflation under load (x)")
    ax[1].set_title("(b) latency under Poisson arrivals (pairs with no deadline misses)", fontsize=9)
    ax[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Fig 13. WebGPU is the most predictable browser backend", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig13_stability_load.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(rows).groupby(["pillar", "backend"]).between_trial_cov.median().round(4).to_string())
    print(lw.groupby(["backend", "rate"]).inflation.median().round(2).to_string())
    print(pd.read_csv(TAB / "t24_latency_vs_kernels_fit.csv").to_string())
    print(famv.round(2).to_string())


if __name__ == "__main__":
    main()
