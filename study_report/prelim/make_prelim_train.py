"""
Training-specific overview figures, built from the tables written by make_prelim.py (run that first).

  fig14_optimizer_share      WebGPU training step split into forward / backward / optimizer (kernels and GPU time)
  fig15_sync_policy          step time under per-step sync, pipelined sync and model.fit; WebGPU read share per step
  fig16_gpu_time_vs_native   GPU-busy time per training step (Nsight): WebGPU vs native CUDA
  fig17_gradient_accuracy    step-0 gradient error vs the fp32 CPU reference, per backend

Usage: python study_report/prelim/make_prelim_train.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG, TAB = OUT / "figures", OUT / "tables"
ORDER = [f + str(i) for f in "DCRLGT" for i in range(1, 6)]
SUB = ["D3", "C3", "R3", "L3", "G3", "T3"]
COL = {"webgpu": "#1f77b4", "webgl": "#ff7f0e", "wasm-8t": "#2ca02c", "tf-cuda-fp32": "#d62728", "tf-cpu": "#9467bd"}
PH = {"forward": "#9ecae1", "backward": "#3182bd", "optimizer": "#e6550d"}


def fam_seps(ax, n=30):
    for k in range(5, n, 5):
        ax.axvline(k - 0.5, color="gray", lw=0.5, ls=":")


def main():
    # ---------- fig 14: optimizer share ----------
    t6 = pd.read_csv(TAB / "t6_webgpu_train_phases.csv", index_col=0).reindex(ORDER)
    k = t6[["kernels_fwd", "kernels_bwd", "kernels_opt"]]
    ks = k.div(k.sum(axis=1), axis=0)
    g = t6[["fwd_gpu_ms", "bwd_gpu_ms", "opt_gpu_ms"]]
    gs = g.div(g.sum(axis=1), axis=0)
    c = t6[["cpu_fwd_ms", "cpu_bwd_ms", "cpu_opt_ms"]]
    cs = c.div(c.sum(axis=1), axis=0)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    x = np.arange(len(ORDER))
    for a, d, lab in ((ax[0], ks, "share of kernels"), (ax[1], gs, "share of GPU kernel time"), (ax[2], cs, "share of host CPU time")):
        bottom = np.zeros(len(ORDER))
        for j, ph in enumerate(PH):
            v = d.iloc[:, j].to_numpy(float)
            a.bar(x, v, 0.8, bottom=bottom, color=PH[ph], label=ph)
            bottom += np.nan_to_num(v)
        a.set_ylabel(lab, fontsize=8)
        a.set_ylim(0, 1)
        fam_seps(a)
    ax[0].legend(fontsize=7, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(ORDER, fontsize=7)
    fig.suptitle("Fig 14. WebGPU training step by phase: Adam (optimizer) is a large, model-dependent share", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "fig14_optimizer_share.png", dpi=150)
    plt.close(fig)
    pd.concat({"kernels": ks, "gpu_time": gs, "cpu_time": cs}, axis=1).round(3).to_csv(TAB / "t29_train_phase_shares.csv")

    # ---------- fig 15: sync policy ----------
    t19 = pd.read_csv(TAB / "t19_why_configs.csv")
    t16 = pd.read_csv(TAB / "t16_fit_epochs.csv")
    t4 = pd.read_csv(TAB / "t4_train_step.csv", index_col=0).reindex(ORDER)
    sync = t19[t19.stage == "train"].set_index(["model", "backend"]).p50
    pipe = t19[t19.stage == "train_pipelined"].set_index(["model", "backend"]).p50
    fit2 = t16[t16.epoch == 2].set_index(["model", "backend"]).ms_per_step
    rows = []
    for m in SUB:
        for be in ("webgl", "webgpu"):
            rows.append({"model": m, "backend": be, "per-step sync": sync.get((m, be), np.nan),
                         "pipelined (10 steps / sync)": pipe.get((m, be), np.nan), "model.fit (no sync), epoch 2": fit2.get((m, be), np.nan)})
    sp = pd.DataFrame(rows).set_index(["model", "backend"])
    sp.round(3).to_csv(TAB / "t30_sync_policy.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    pol = list(sp.columns)
    hatch = {"per-step sync": "", "pipelined (10 steps / sync)": "//", "model.fit (no sync), epoch 2": ".."}
    w = 0.13
    x = np.arange(len(SUB))
    for i, be in enumerate(("webgl", "webgpu")):
        for j, p in enumerate(pol):
            off = (i * 3 + j - 2.5) * w
            v = [sp.loc[(m, be), p] for m in SUB]
            ax[0].bar(x + off, v, w, color=COL[be], hatch=hatch[p], edgecolor="white" if not hatch[p] else "k", lw=0.3,
                      label=f"{be}: {p}")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(SUB)
    ax[0].set_ylabel("ms per training step (log)")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("(a) step time by sync policy (preflight, 1 window each)", fontsize=9)
    xx = np.arange(len(ORDER))
    ax[1].bar(xx, t4.webgpu_read_share, color=COL["webgpu"])
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(ORDER, fontsize=6, rotation=90)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("share of WebGPU step spent in await loss.data()")
    ax[1].set_title("(b) per-step readback share (all 30 models)", fontsize=9)
    fam_seps(ax[1])
    fig.suptitle("Fig 15. Per-step synchronization is a first-order part of browser training cost", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig15_sync_policy.png", dpi=150)
    plt.close(fig)

    # ---------- fig 16: GPU time vs native ----------
    ns = pd.read_csv(TAB / "t14_nsight.csv")
    ns = ns[ns["mode"] == "train"]
    cfgs = [("webgpu", "WebGPU", COL["webgpu"]), ("webgl", "WebGL (no GPU trace)", COL["webgl"]),
            ("native-tf-cuda-fp32", "native CUDA (graph)", COL["tf-cuda-fp32"]), ("native-tf-cuda-eager", "native CUDA (eager)", "#ff9896")]
    models = ["D3", "L3", "T3"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    w = 0.2
    x = np.arange(len(models))
    for j, (cfg, lab, colr) in enumerate(cfgs):
        d = ns[ns.config == cfg].set_index("model").reindex(models)
        ax[0].bar(x + (j - 1.5) * w, d.page_p50_ms, w, color=colr, label=lab)
        if cfg != "webgl":
            jj = {"webgpu": -1, "native-tf-cuda-fp32": 0, "native-tf-cuda-eager": 1}[cfg]
            ax[1].bar(x + jj * 0.25, d.gpu_busy_ms_per_inference, 0.25, color=colr, label=lab)
    for a, t in ((ax[0], "(a) wall time per step (ms)"), (ax[1], "(b) GPU-busy time per step (ms, Nsight)")):
        a.set_xticks(x)
        a.set_xticklabels(models)
        a.set_yscale("log")
        a.set_title(t, fontsize=9)
        a.legend(fontsize=7)
    gpu = ns[ns.config == "webgpu"].set_index("model").gpu_busy_ms_per_inference
    nat = ns[ns.config == "native-tf-cuda-fp32"].set_index("model").gpu_busy_ms_per_inference
    for i, m in enumerate(models):
        ax[1].text(i - 0.125, gpu[m] * 1.25, f"{gpu[m] / nat[m]:.1f}x vs graph", ha="center", fontsize=8)
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 4)
    fig.suptitle("Fig 16. The training gap to native is also GPU work: WebGPU kernels take 1.8-5.5x the GPU time", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig16_gpu_time_vs_native.png", dpi=150)
    plt.close(fig)
    ns[["model", "config", "page_p50_ms", "gpu_busy_ms_per_inference", "gpu_busy_frac_in_inference"]].round(3).to_csv(
        TAB / "t31_train_gpu_time.csv", index=False)

    # ---------- fig 17: gradient accuracy ----------
    t22 = pd.read_csv(TAB / "t22_train_grad_rel.csv", index_col=0).reindex(ORDER)
    series = [("webgpu", "WebGPU", COL["webgpu"], "o"), ("webgl", "WebGL", COL["webgl"], "s"), ("wasm-8t", "WASM", COL["wasm-8t"], "^"),
              ("tf-cuda-fp32+tfjsadam", "native CUDA fp32", COL["tf-cuda-fp32"], "D"),
              ("tf-cuda+tfjsadam", "native CUDA TF32 (default)", "#7f7f7f", "x")]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(ORDER))
    for j, (c_, lab, colr, mk) in enumerate(series):
        v = t22[c_].to_numpy(float)
        ax.scatter(x + (j - 2) * 0.1, np.where(v > 0, v, np.nan), color=colr, marker=mk, s=22, label=lab)
    ax.set_yscale("log")
    ax.axhline(1e-4, color="k", lw=0.6, ls="--")
    ax.text(29.6, 1.2e-4, "P0.3 tolerance 1e-4", ha="right", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=7)
    ax.set_ylabel("step-0 gradient error, relative\n(vs native fp32 CPU, same Adam)")
    fam_seps(ax)
    ax.set_ylim(5e-8, 3)
    ax.legend(fontsize=7, ncol=5, loc="upper center")
    ax.set_title("Fig 17. Browser training gradients match CPU fp32; native CUDA's default TF32 does not", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig17_gradient_accuracy.png", dpi=150)
    plt.close(fig)
    med = {lab: float(np.nanmedian(t22[c_].replace(0, np.nan))) for c_, lab, *_ in series}
    print("median gradient rel err:", {k_: f"{v:.1e}" for k_, v in med.items()})
    print(sp.round(2).to_string())
    print(pd.concat({"kernels_opt": ks.kernels_opt, "gpu_opt": gs.opt_gpu_ms, "cpu_opt": cs.cpu_opt_ms}, axis=1).round(2)
          .groupby(lambda m: m[0]).agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
