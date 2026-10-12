"""
Where does a training step's time go? Forward / backward / optimizer, from three independent sources:

  kernel_phase.csv   train_profile launches (kernels.csv, tf.profile with the phase join exact): GPU (or
                     WASM) kernel time and kernel count per phase, per step (median over the profiled steps).
                     tf.profile perturbs WebGPU (one submit per kernel): these are kernel times, not latency.
  hook_phase.csv     train_instrumented launches (runs.csv hook columns): outermost kernels per phase, the
                     JavaScript CPU time spent issuing them (kernel_cpu_ms_<phase>), kernels forwarded to TF.js's
                     CPU implementation per phase; median over the instrumented steps.
  ablation.csv       train_phases launches (timing): medians of forward-only, forward+backward and the full
                     step, and the derived costs backward = fwdbwd - fwd, optimizer = step - fwdbwd, each with a
                     bootstrap 95 % CI of the difference of medians. Native launches (train_phases) appear with
                     their config label.

Usage:  python analysis/train_phase.py --profile <dir>... --instrumented <dir>... --phases <dir>... --out DIR
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PHASES = ["forward", "backward", "optimizer"]


def _ok(d, mode):
    s = pd.read_csv(Path(d) / "sessions.csv")
    return s[(s["status"] == "ok") & (s["mode"] == mode)][["trial", "model", "backend"]]


def kernel_phase(dirs):
    rows = []
    for d in map(Path, dirs):
        if not (d / "kernels.csv").exists():
            continue
        k = pd.read_csv(d / "kernels.csv").merge(_ok(d, "train_profile"), on=["trial", "model", "backend"])
        r = pd.read_csv(d / "runs.csv", low_memory=False)
        pr = r[r["phase"] == "profile_kernels"][["trial", "model", "backend", "i", "phase_join_ok"]]
        k = k[(k["source"] == "tf.profile") & ~k["name"].astype(str).str.endswith("__op")]
        k = k.merge(pr.rename(columns={"i": "run"}), on=["trial", "model", "backend", "run"])
        k = k[k["phase_join_ok"].astype(str).str.lower() == "true"]
        per_step = k.groupby(["model", "backend", "trial", "run", "train_phase"]).agg(
            kernel_ms=("kernel_ms", lambda x: x.sum(min_count=len(x))), kernels=("name", "size")).reset_index()
        rows.append(per_step)
    if not rows:
        return pd.DataFrame()
    x = pd.concat(rows, ignore_index=True)
    t = x.groupby(["model", "backend", "train_phase"])[["kernel_ms", "kernels"]].median().unstack("train_phase")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    tot = sum(t.get(f"kernel_ms_{p}", 0) for p in PHASES)
    for p in PHASES:
        if f"kernel_ms_{p}" in t:
            t[f"kernel_share_{p}"] = t[f"kernel_ms_{p}"] / tot
    return t


def hook_phase(dirs):
    rows = []
    for d in map(Path, dirs):
        r = pd.read_csv(Path(d) / "runs.csv", low_memory=False).merge(_ok(d, "train"), on=["trial", "model", "backend"])
        r = r[r["phase"].isin(["train_warmup", "train_measure"])]
        cols = [c for c in r.columns if c.startswith(("hooks_kernels_", "hooks_kernel_cpu_ms_", "hooks_forwarded_"))]
        if not cols:
            continue
        steady = r[r["phase"] == "train_measure"]
        first = r[(r["phase"] == "train_warmup") & (r["i"] == 0)]
        a = steady.groupby(["model", "backend"])[cols + [c for c in ("webgpu_dispatches", "webgpu_submits", "webgl_draws",
                                                                       "webgpu_pipelines_created", "webgl_programs_linked",
                                                                       "webgpu_write_buffer_bytes", "webgpu_mapped_at_creation_bytes",
                                                                       "webgl_tex_upload_bytes", "webgpu_bind_groups_created")
                                                       if c in steady]].median()
        b = first.groupby(["model", "backend"])[[c for c in ("webgpu_pipelines_created", "webgl_programs_linked",
                                                              "webgpu_buffers_created", "webgpu_mapped_at_creation_bytes",
                                                              "webgl_tex_upload_bytes") if c in first]].median()
        rows.append(a.join(b.add_prefix("step0_")).reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def diff_ci(a, b, n_boot=2000, seed=0):
    """median(a) - median(b) with a percentile-bootstrap 95 % CI (independent resampling)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = np.median(rng.choice(a, (n_boot, len(a))), axis=1) - np.median(rng.choice(b, (n_boot, len(b))), axis=1)
    return float(np.median(a) - np.median(b)), float(np.quantile(da, 0.025)), float(np.quantile(da, 0.975))


def ablation(dirs):
    rows = []
    frames = []
    for d in map(Path, dirs):
        r = pd.read_csv(d / "runs.csv", low_memory=False).merge(_ok(d, "train_phases"), on=["trial", "model", "backend"])
        frames.append(r[r["phase"].isin(["phase_fwd", "phase_fwdbwd", "phase_step"])])
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    for (m, b), g in x.groupby(["model", "backend"]):
        v = {p: g[g["phase"] == f"phase_{p}"]["dur_ms"].to_numpy() for p in ("fwd", "fwdbwd", "step")}
        if any(len(a) < 5 for a in v.values()):
            continue
        bw = diff_ci(v["fwdbwd"], v["fwd"])
        op = diff_ci(v["step"], v["fwdbwd"])
        rows.append({"model": m, "backend": b, "fwd_ms": np.median(v["fwd"]), "fwdbwd_ms": np.median(v["fwdbwd"]),
                     "step_ms": np.median(v["step"]), "backward_ms": bw[0], "backward_ci_lo": bw[1], "backward_ci_hi": bw[2],
                     "optimizer_ms": op[0], "optimizer_ci_lo": op[1], "optimizer_ci_hi": op[2],
                     "backward_over_forward": bw[0] / np.median(v["fwd"]), "n": int(min(len(a) for a in v.values()))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", nargs="*", default=[])
    ap.add_argument("--instrumented", nargs="*", default=[])
    ap.add_argument("--phases", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        for name, t in (("kernel_phase", kernel_phase(args.profile)), ("hook_phase", hook_phase(args.instrumented)),
                        ("ablation", ablation(args.phases))):
            if len(t):
                t.to_csv(out / f"{name}.csv", index=False)
                print(f"\n{name}:")
                print(t.round(3).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
