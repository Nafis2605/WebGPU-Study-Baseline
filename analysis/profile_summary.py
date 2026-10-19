"""
Group 5 execution and coordination (profile-mode launches). TF.js replacement for the ONNX study's
profile_summary.py + f_why decomposition / dispatch_overhead / op_speedup: ORT's trace (per-node
CPU time) doesn't exist in TF.js; per-kernel CPU time comes from src/backend-hooks.js and per-kernel
GPU time from tf.profile().

profile_summary.csv   per (model, config), medians over runs (all trials):
  profile_counters phase (the normal run, instrumented):
    wall_ms, exec_ms, read_ms            request start -> outputs on CPU; until execute() returned; data()
    kernel_cpu_ms                        sum of per-kernel CPU time (outermost engine.runKernel)
    outside_kernel_ms                    wall - kernel_cpu_ms (feeds, graph-executor bookkeeping,
                                         waiting for readback)
    kernels_run, forwarded_kernels, sync_reads, sync_read_ms, async_reads
    dispatches, submits, dispatches_per_submit, compute_passes, bind_groups, draws
    programs_or_pipelines                (should be 0 in steady state)
    upload_bytes, readback_bytes, map_read_wait_ms
    max_in_flight, submit_gap_median_ms, group_completion_median_ms   (bounded: queue completion)
  kernel time per run (kernel_time_ms, kernel_time_source):
    webgpu  sum of tf.profile kernelTimeMs (compute-pass timestamps)
    webgl   sum of the harness's own per-draw timer queries (profile_draws); tf.profile's WebGL
            kernel times are unreliable per kernel (single kernels spike by several ms; the
            per-run sum varied 2.6-10 ms where the draw timers gave a steady 6.3 ms), kept as
            kernel_time_tfprofile_ms for reference
    wasm    sum of tf.profile kernelTimeMs (synchronous wall time of each kernel = CPU compute)
    "<op>__op" wrapper records are excluded (see src/profile.js)
  gpu_busy_ms = kernel_time_ms for GPU backends (NaN for wasm)
  gpu_kernels, zero_time_frac, profile_wall_ms, profile_submits
  profile_time phase: time_kernel_ms, time_wall_ms (tf.time)
  profile_draws phase (WebGL): draw_gpu_ms (harness timer queries), draw_disjoint
  derived (ratios of measured medians):
    gpu_busy_frac      gpu_busy_ms / latency p50 (latency dirs given with --latency)
    cpu_over_gpu       kernel_cpu_ms / gpu_busy_ms
    profile_perturbation  profile_wall_ms / wall_ms
    counter_overhead   wall_ms / latency p50
    webgl_profile_vs_draws  gpu_busy_ms / draw_gpu_ms
top_kernels.csv       per (model, config, kernel name): gpu_ms_per_run (tf.profile kernelTimeMs, median
                      over runs), calls per run, share of the per-run sum, and cpu_ms_per_run (per-kernel
                      CPU time from the last profile_counters run of each launch, median over launches)
op_speedup.csv        per kernel name: wasm-8t kernel ms / webgpu kernel ms, per model and geomean;
                      retained speedup = end-to-end speedup / kernel-sum speedup
dispatch_overhead.csv per GPU config: linear fit kernel_cpu_ms ~ dispatches across models (us per dispatch)
invariants.csv        per (model, config): violations of gpu_busy <= profile wall, kernel_cpu <= wall

Usage:
    python analysis/profile_summary.py <profile dir> [...] [--latency <latency dir> ...] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stats import linfit

KEY = ["source", "trial", "model", "backend"]


def _bool(col):
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(dirs, mode):
    runs, sessions, kernels = [], [], []
    for d in map(Path, dirs):
        s = pd.read_csv(d / "sessions.csv").assign(source=d.name)
        r = pd.read_csv(d / "runs.csv", low_memory=False).assign(source=d.name)
        s = s[(s["status"] == "ok") & (s["mode"] == mode)]
        sessions.append(s)
        runs.append(r.merge(s[KEY], on=KEY))
        if (d / "kernels.csv").exists():
            # kernels.csv has its own "source" column (tf.profile / kernel_cpu / webgl_draw_timer);
            # keep it as kernel_source before tagging rows with the result folder.
            k = pd.read_csv(d / "kernels.csv").rename(columns={"source": "kernel_source"}).assign(source=d.name)
            kernels.append(k.merge(s[KEY], on=KEY))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return cat(runs), cat(sessions), cat(kernels)


def latency_p50(dirs, mode="latency", phase="measure"):
    if not dirs:
        return pd.Series(dtype=float)
    runs, sessions, _ = load(dirs, mode)
    if "instrumented" in sessions:
        ok = sessions[~_bool(sessions["instrumented"])][KEY]
        runs = runs.merge(ok, on=KEY)
    return runs[runs["phase"] == phase].groupby(["model", "backend"])["dur_ms"].median().rename("latency_p50_ms")


def col(df, name):
    return df[name] if name in df else pd.Series(np.nan, index=df.index)


def per_run(r):
    r = r.copy()
    r["upload_bytes"] = (col(r, "webgpu_write_buffer_bytes").fillna(0) + col(r, "webgpu_mapped_at_creation_bytes").fillna(0)
                         + col(r, "webgl_tex_upload_bytes").fillna(0))
    r["readback_bytes"] = (col(r, "webgpu_map_read_bytes").fillna(0) + col(r, "webgl_read_pixels_bytes").fillna(0))
    r["dispatches_per_submit"] = col(r, "webgpu_dispatches") / col(r, "webgpu_submits").where(col(r, "webgpu_submits") > 0)
    r["programs_or_pipelines"] = (col(r, "webgl_programs_linked").fillna(0) + col(r, "webgpu_pipelines_created").fillna(0)
                                  + col(r, "webgpu_pipelines_created_async").fillna(0))
    r["outside_kernel_ms"] = r["dur_ms"] - col(r, "hooks_kernel_cpu_ms")
    return r


def summarize(runs, lat):
    runs = per_run(runs)
    g = runs[runs["phase"] == "profile_counters"].groupby(["model", "backend"])
    med = lambda c: g[c].median() if c in runs else np.nan
    t = pd.DataFrame({
        "trials": g["trial"].nunique(), "runs": g.size(),
        "wall_ms": med("dur_ms"), "exec_ms": med("exec_ms"), "read_ms": med("read_ms"),
        "kernel_cpu_ms": med("hooks_kernel_cpu_ms"), "outside_kernel_ms": med("outside_kernel_ms"),
        "kernels_run": med("hooks_kernels_run"), "forwarded_kernels": med("hooks_forwarded_kernels"),
        "sync_reads": med("hooks_sync_reads"), "sync_read_ms": med("hooks_sync_read_ms"), "async_reads": med("hooks_async_reads"),
        "dispatches": med("webgpu_dispatches"), "submits": med("webgpu_submits"),
        "dispatches_per_submit": med("dispatches_per_submit"), "compute_passes": med("webgpu_compute_passes"),
        "bind_groups": med("webgpu_bind_groups_created"), "draws": med("webgl_draws"),
        "programs_or_pipelines": med("programs_or_pipelines"),
        "upload_bytes": med("upload_bytes"), "readback_bytes": med("readback_bytes"),
        "map_read_wait_ms": med("webgpu_map_read_wait_ms"),
        "max_in_flight": med("submits_max_in_flight"), "submit_gap_median_ms": med("submits_submit_gap_median_ms"),
        "group_completion_median_ms": med("submits_group_completion_median_ms"),
        "unfinished_at_end": med("submits_unfinished_at_end"),
    })
    k = runs[runs["phase"] == "profile_kernels"].groupby(["model", "backend"])
    t["kernel_time_tfprofile_ms"] = k["gpu_busy_ms"].median()
    t["gpu_kernels"] = k["gpu_kernels"].median()
    t["zero_time_frac"] = k.apply(lambda x: (x["zero_time_kernels"].sum() / x["gpu_kernels"].sum()) if x["gpu_kernels"].sum() else np.nan)
    t["profile_wall_ms"] = k["dur_ms"].median()
    t["profile_submits"] = k["webgpu_submits"].median() if "webgpu_submits" in runs else np.nan
    if "time_kernel_ms" in runs:
        tm = runs[runs["phase"] == "profile_time"].groupby(["model", "backend"])
        t["time_kernel_ms"] = tm["time_kernel_ms"].median()
        t["time_wall_ms"] = tm["time_wall_ms"].median()
    if "draw_gpu_ms" in runs:
        dr = runs[runs["phase"] == "profile_draws"].groupby(["model", "backend"])
        t["draw_gpu_ms"] = dr["draw_gpu_ms"].median()
        t["draw_disjoint"] = dr["draw_disjoint"].sum()
        t["webgl_profile_vs_draws"] = t["kernel_time_tfprofile_ms"] / t["draw_gpu_ms"]
    else:
        t["draw_gpu_ms"] = np.nan
    ep = t.index.get_level_values("backend").map(lambda b: "webgl" if b.startswith("webgl") else
                                                  "webgpu" if b.startswith("webgpu") else "wasm")
    t["kernel_time_source"] = np.where(ep == "webgl", "webgl_draw_timer",
                                       np.where(ep == "webgpu", "webgpu_timestamp", "wasm_sync_wall"))
    t["kernel_time_ms"] = np.where(ep == "webgl", t["draw_gpu_ms"], t["kernel_time_tfprofile_ms"])
    t["gpu_busy_ms"] = np.where(ep == "wasm", np.nan, t["kernel_time_ms"])
    if "submits_done_spread_ms" in runs:
        t["submit_done_spread_ms"] = g["submits_done_spread_ms"].median()
    t = t.join(lat, how="left") if len(lat) else t.assign(latency_p50_ms=np.nan)
    t["gpu_busy_frac"] = t["gpu_busy_ms"] / t["latency_p50_ms"]
    t["cpu_over_gpu"] = t["kernel_cpu_ms"] / t["gpu_busy_ms"]
    t["kernel_time_frac"] = t["kernel_time_ms"] / t["latency_p50_ms"]
    t["profile_perturbation"] = t["profile_wall_ms"] / t["wall_ms"]
    t["counter_overhead"] = t["wall_ms"] / t["latency_p50_ms"]
    return t.reset_index(), runs


def invariants(runs):
    rows = []
    for (m, c), g in runs.groupby(["model", "backend"]):
        kp = g[g["phase"] == "profile_kernels"]
        cp = g[g["phase"] == "profile_counters"]
        rows.append({"model": m, "backend": c,
                     "kernel_sum_gt_profile_wall": int((kp["gpu_busy_ms"] > kp["dur_ms"] + 1e-3).sum()),
                     "kernel_cpu_gt_wall": int((cp["hooks_kernel_cpu_ms"] > cp["dur_ms"] + 1e-3).sum()),
                     "kernel_time_invalid": int(col(kp, "kernel_time_invalid").fillna(0).sum()),
                     "runs_checked": len(kp) + len(cp)})
    return pd.DataFrame(rows)


def top_kernels(kernels):
    if kernels.empty:
        return pd.DataFrame()
    k = kernels[(kernels["kernel_source"] == "tf.profile") & ~kernels["name"].astype(str).str.endswith("__op")]
    per = k.groupby(["source", "trial", "model", "backend", "run", "name"])["kernel_ms"].sum().reset_index()
    med = per.groupby(["model", "backend", "name"])["kernel_ms"].median().rename("ms_per_run").reset_index()
    med["share"] = med["ms_per_run"] / med.groupby(["model", "backend"])["ms_per_run"].transform("sum")
    calls = k.groupby(["model", "backend", "run", "name", "trial"]).size().groupby(["model", "backend", "name"]).median()
    med = med.merge(calls.rename("calls_per_run").reset_index(), on=["model", "backend", "name"])
    cpu = kernels[kernels["kernel_source"] == "kernel_cpu"].groupby(["model", "backend", "name"])["kernel_ms"].median()
    med = med.merge(cpu.rename("cpu_ms_per_run").reset_index(), on=["model", "backend", "name"], how="left")
    return med.sort_values(["model", "backend", "ms_per_run"], ascending=[True, True, False])


def op_speedup(top, lat, cpu_cfg="wasm-8t", gpu_cfg="webgpu"):
    if top.empty:
        return pd.DataFrame(), pd.DataFrame()
    a = top[top["backend"] == cpu_cfg].set_index(["model", "name"])["ms_per_run"]
    b = top[top["backend"] == gpu_cfg].set_index(["model", "name"])["ms_per_run"]
    j = pd.concat([a.rename("cpu_ms"), b.rename("gpu_ms")], axis=1).dropna()
    j = j[(j["gpu_ms"] > 0) & (j["cpu_ms"] > 0)].reset_index()   # 0 ms (metadata-only) kernels: undefined
    j["speedup"] = j["cpu_ms"] / j["gpu_ms"]
    per_op = j.groupby("name").agg(models=("model", "nunique"), cpu_total_ms=("cpu_ms", "sum"),
                                   gpu_total_ms=("gpu_ms", "sum"),
                                   geomean_speedup=("speedup", lambda s: float(np.exp(np.log(s).mean())))).reset_index()
    per_op["cpu_share"] = per_op["cpu_total_ms"] / per_op["cpu_total_ms"].sum()
    retained = []
    for m, g in j.groupby("model"):
        kernel = top[(top["model"] == m) & (top["backend"] == cpu_cfg)]["ms_per_run"].sum() / \
            top[(top["model"] == m) & (top["backend"] == gpu_cfg)]["ms_per_run"].sum()
        if (m, cpu_cfg) in lat.index and (m, gpu_cfg) in lat.index:
            e2e = lat[(m, cpu_cfg)] / lat[(m, gpu_cfg)]
            retained.append({"model": m, "e2e_speedup": e2e, "kernel_sum_speedup": kernel, "retained": e2e / kernel})
    return per_op.sort_values("cpu_total_ms", ascending=False), pd.DataFrame(retained)


def dispatch_overhead(summary):
    rows = []
    for c, g in summary[summary["backend"].str.startswith("webgpu")].groupby("backend"):
        g = g.dropna(subset=["dispatches", "kernel_cpu_ms"])
        g = g[g["dispatches"] > 0]
        row = {"backend": c, "models": len(g)}
        if len(g) >= 3:
            fit = linfit(g["dispatches"], g["kernel_cpu_ms"])
            row.update({"us_per_dispatch": fit["slope"] * 1000, "intercept_ms": fit["intercept"], "r2": fit["r2"]})
        row["per_model_us_per_dispatch"] = json.dumps({m: round(v * 1000, 3) for m, v in
                                                       zip(g["model"], g["kernel_cpu_ms"] / g["dispatches"])})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--latency", nargs="*", default=[], help="latency-mode dirs for p50 (gpu_busy_frac, overhead)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="profile", choices=("profile", "train_profile"),
                    help="profile = one inference per record; train_profile = one training step per record "
                         "(--latency then takes train-stage dirs: step p50 from train_measure)")
    args = ap.parse_args()
    out = Path(args.out or args.dirs[0])
    out.mkdir(parents=True, exist_ok=True)
    runs, sessions, kernels = load(args.dirs, args.mode)
    lat = latency_p50(args.latency, *(("train", "train_measure") if args.mode == "train_profile" else ("latency", "measure")))
    summary, runs = summarize(runs, lat)
    inv = invariants(runs)
    top = top_kernels(kernels)
    per_op, retained = op_speedup(top, lat)
    disp = dispatch_overhead(summary)
    summary.to_csv(out / f"{args.mode}_summary.csv", index=False)
    inv.to_csv(out / "invariants.csv", index=False)
    if len(top):
        top.to_csv(out / "top_kernels.csv", index=False)
    if len(per_op):
        per_op.to_csv(out / "op_speedup.csv", index=False)
        retained.to_csv(out / "retained_speedup.csv", index=False)
    disp.to_csv(out / "dispatch_overhead.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 50):
        cols = ["model", "backend", "trials", "wall_ms", "exec_ms", "read_ms", "kernel_cpu_ms", "outside_kernel_ms",
                "kernel_time_ms", "kernel_time_source", "gpu_busy_frac", "cpu_over_gpu", "dispatches", "submits",
                "dispatches_per_submit", "max_in_flight", "submit_done_spread_ms", "draws", "forwarded_kernels",
                "sync_reads", "sync_read_ms", "profile_perturbation", "counter_overhead", "zero_time_frac"]
        print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
        extra = [c for c in ["kernel_time_tfprofile_ms", "time_kernel_ms", "time_wall_ms", "draw_gpu_ms",
                             "webgl_profile_vs_draws", "draw_disjoint",
                             "upload_bytes", "readback_bytes", "map_read_wait_ms", "programs_or_pipelines"] if c in summary]
        print(summary[["model", "backend"] + extra].round(3).to_string(index=False))
        print("\nInvariants:"); print(inv.to_string(index=False))
        if len(top):
            print("\nTop kernels (3 per model/config):")
            print(top.groupby(["model", "backend"]).head(3).round(4).to_string(index=False))
        if len(retained):
            print("\nRetained speedup:"); print(retained.round(3).to_string(index=False))
        print("\nDispatch overhead:"); print(disp.round(3).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
