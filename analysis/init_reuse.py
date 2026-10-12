"""
Group 1: initialization and reuse (port of the ONNX study's analysis/init_reuse.py + TF.js additions).

Timing launches (status ok, not instrumented):
    init_summary.csv        per (model, config): backend_init_ms (WASM instantiation / WebGPU device /
                            WebGL context), fetch_ms (model.json + weight shards from the local
                            server), session_create_ms (graph build + weight tensor creation),
                            precompile_ms (TF.js compile-only pass; *-precompile configs), first and
                            second inference, steady median (measure phase), and
                              first_run_overhead_ms = first_run - steady median
                              time_to_first_result_ms = backend_init + fetch + session_create
                                                        + precompile + first_run
                            (a difference / sum of measured values, per launch). Medians over
                            trials, with bootstrap CIs when there are >= 3 trials.
    precompile_summary.csv  cold vs pre-compiled config pairs (webgl / webgl-precompile, webgpu /
                            webgpu-precompile): first run, precompile cost, time to first result,
                            ratio of medians with a bootstrap CI.
                            Scope differs by backend (instrumented launches): on WebGPU the
                            compile-only pass also uploads the weights (mapped-at-creation bytes
                            move from the first run into precompile_ms); on WebGL the weights are
                            still uploaded by the first run.
    shape_reuse_summary.csv shapes mode, per (model, config, shape):
                              first_seen  first call of a shape never run before in the page
                              revisit     first call of a shape seen earlier, right after a different shape
                              steady      later calls of a block whose shape was already seen
                              post_first_seen  later calls of the block that introduced the shape
                                          (reported separately: on WebGL they stay slow after a
                                          compile, so pooling them into "steady" would inflate it;
                                          the ONNX study pooled them)
                            medians and first_seen/steady, revisit/steady ratios.
    shape_order_summary.csv shapes mode, first_seen calls by their order among the launch's new
                            shapes (1 = first departure from the base shape). The sequence seed
                            rotates with the trial, so order and shape value are not confounded;
                            a one-time cost of leaving the base shape shows up at order 1 for
                            every value.

Instrumented launches (--instrument-runs; their timings are never used above):
    first_run_counters.csv  per (model, config, mode): what run 0 (the first inference), the last
                            warm-up run, and each shape kind did: programs linked / pipelines
                            created, upload bytes, buffers created, forwarded kernels, sync reads.

Usage:
    python analysis/init_reuse.py <results dir> [<results dir> ...] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats import median_ci, ratio_of_medians_ci

KEYS = ["source", "trial", "model", "backend"]
PRECOMPILE_PAIRS = {"webgl-precompile": "webgl", "webgpu-precompile": "webgpu"}
COMPILE_COLS = ["webgl_programs_linked", "webgpu_pipelines_created", "webgpu_pipelines_created_async"]
COUNTER_COLS = COMPILE_COLS + ["webgl_tex_upload_bytes", "webgpu_write_buffer_bytes", "webgpu_mapped_at_creation_bytes",
                               "webgpu_copy_b2b_bytes",
                               "webgpu_buffers_created", "webgpu_buffer_bytes_created", "webgl_draws",
                               "webgpu_dispatches", "webgpu_submits",
                               "hooks_forwarded_kernels", "hooks_sync_reads", "hooks_sync_read_ms"]


def _bool(col):
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(dirs):
    runs, sessions = [], []
    for d in map(Path, dirs):
        r = pd.read_csv(d / "runs.csv", low_memory=False)
        s = pd.read_csv(d / "sessions.csv")
        runs.append(r.assign(source=d.name))
        sessions.append(s.assign(source=d.name))
    runs = pd.concat(runs, ignore_index=True)
    sessions = pd.concat(sessions, ignore_index=True)
    sessions["instrumented"] = _bool(sessions["instrumented"]) if "instrumented" in sessions else False
    return runs, sessions


def timing_sessions(sessions, mode):
    return sessions[(sessions["status"] == "ok") & (sessions["mode"] == mode) & ~sessions["instrumented"]]


def per_launch_init(runs, sessions):
    ok = timing_sessions(sessions, "latency")
    if ok.empty:
        return pd.DataFrame()
    steady = (runs[runs["phase"] == "measure"].groupby(KEYS)["dur_ms"].median()
              .rename("steady_median_ms").reset_index())
    cols = ["model_bytes", "backend_init_ms", "fetch_ms", "session_create_ms", "precompile_ms",
            "first_run_ms", "second_run_ms"]
    t = ok[KEYS + [c for c in cols if c in ok]].merge(steady, on=KEYS)
    t["first_run_overhead_ms"] = t["first_run_ms"] - t["steady_median_ms"]
    t["time_to_first_result_ms"] = (t["backend_init_ms"] + t["fetch_ms"] + t["session_create_ms"]
                                    + t["precompile_ms"].fillna(0) + t["first_run_ms"])
    return t


def _ci_cols(values, name):
    v = values.dropna().to_numpy()
    if len(v) >= 3:
        med, lo, hi = median_ci(v)
        return {f"{name}_median": med, f"{name}_ci_lo": lo, f"{name}_ci_hi": hi}
    return {f"{name}_median": float(np.median(v)) if len(v) else np.nan, f"{name}_ci_lo": np.nan, f"{name}_ci_hi": np.nan}


def init_table(t):
    rows = []
    cols = ["backend_init_ms", "fetch_ms", "session_create_ms", "precompile_ms", "first_run_ms", "second_run_ms",
            "steady_median_ms", "first_run_overhead_ms", "time_to_first_result_ms"]
    for (model, backend), g in t.groupby(["model", "backend"]):
        row = {"model": model, "backend": backend, "trials": len(g), "model_mb": g["model_bytes"].iloc[0] / 2**20}
        for c in cols:
            if g[c].notna().any():
                row.update(_ci_cols(g[c], c))
            else:
                row.update({f"{c}_median": np.nan, f"{c}_ci_lo": np.nan, f"{c}_ci_hi": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def precompile_table(t):
    rows = []
    for pre, cold in PRECOMPILE_PAIRS.items():
        for model in sorted(set(t["model"])):
            a = t[(t["model"] == model) & (t["backend"] == pre)]
            b = t[(t["model"] == model) & (t["backend"] == cold)]
            if a.empty or b.empty:
                continue
            row = {"model": model, "cold_config": cold, "precompile_config": pre,
                   "trials_cold": len(b), "trials_precompile": len(a),
                   "precompile_ms_median": a["precompile_ms"].median()}
            for c in ("first_run_ms", "time_to_first_result_ms", "steady_median_ms"):
                row[f"{c}_cold_median"] = b[c].median()
                row[f"{c}_precompile_median"] = a[c].median()
                if len(a) >= 2 and len(b) >= 2:
                    r, lo, hi = ratio_of_medians_ci(a[c], b[c])
                else:
                    r, lo, hi = a[c].median() / b[c].median(), np.nan, np.nan
                row[f"{c}_ratio"], row[f"{c}_ratio_ci_lo"], row[f"{c}_ratio_ci_hi"] = r, lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def classify_shapes(s):
    s = s.copy()
    s["first_seen"] = _bool(s["first_seen"])
    # A block is "introducing" if its first call is the first ever of that shape in the launch.
    intro = s[s["first_seen"]][KEYS + ["block"]].drop_duplicates().assign(intro_block=True)
    s = s.merge(intro, on=KEYS + ["block"], how="left")
    s["intro_block"] = s["intro_block"].fillna(False).astype(bool)
    s["kind"] = "steady"
    s.loc[s["intro_block"] & (s["block_pos"] > 0), "kind"] = "post_first_seen"
    s.loc[s["block_pos"] == 0, "kind"] = "revisit"
    s.loc[s["first_seen"], "kind"] = "first_seen"
    # Order of each first-seen call among the launch's new shapes (1 = first departure from base).
    fs = s[s["first_seen"]].sort_values(KEYS + ["i"])
    s["new_shape_order"] = np.nan
    s.loc[fs.index, "new_shape_order"] = fs.groupby(KEYS).cumcount() + 1
    return s


def shape_order_table(runs, sessions):
    """first_seen calls by new-shape order: timing launches for latency, instrumented ones for counters."""
    out = []
    ok = timing_sessions(sessions, "shapes")[KEYS]
    s = runs[runs["phase"] == "shape"].merge(ok, on=KEYS)
    if not s.empty:
        s = classify_shapes(s)
        steady = s[s["kind"] == "steady"].groupby(["model", "backend", "shape_value"])["dur_ms"].median().rename("steady_ms")
        fs = s[s["first_seen"]].join(steady, on=["model", "backend", "shape_value"])
        fs["over_steady"] = fs["dur_ms"] / fs["steady_ms"]
        t = fs.groupby(["model", "backend", "new_shape_order"]).agg(
            n=("dur_ms", "size"), first_seen_ms_median=("dur_ms", "median"),
            over_steady_median=("over_steady", "median"),
            shape_values=("shape_value", lambda v: ",".join(str(int(x)) for x in sorted(set(v))))).reset_index()
        out.append(t)
    inst = sessions[(sessions["status"] == "ok") & sessions["instrumented"] & (sessions["mode"] == "shapes")][KEYS]
    c = runs[runs["phase"] == "shape"].merge(inst, on=KEYS)
    if not c.empty:
        c = classify_shapes(c)
        c["pipelines_or_programs"] = c[[k for k in COMPILE_COLS if k in c]].fillna(0).sum(axis=1)
        cols = [k for k in ("pipelines_or_programs", "webgpu_buffers_created", "webgpu_buffer_bytes_created") if k in c]
        ct = (c[c["first_seen"]].groupby(["model", "backend", "new_shape_order"])[cols].median()
              .add_prefix("inst_").reset_index())
        out = [out[0].merge(ct, on=["model", "backend", "new_shape_order"], how="outer")] if out else [ct]
    return out[0] if out else pd.DataFrame()


def shapes_table(runs, sessions):
    ok = timing_sessions(sessions, "shapes")[KEYS]
    s = runs[runs["phase"] == "shape"].merge(ok, on=KEYS)
    if s.empty:
        return pd.DataFrame()
    s = classify_shapes(s)
    t = (s.groupby(["model", "backend", "shape", "shape_value", "kind"])["dur_ms"]
         .agg(["median", "count"]).unstack("kind"))
    t.columns = [f"{kind}_{stat}" for stat, kind in t.columns]
    t = t.reset_index()
    if "steady_median" not in t:   # too few repeats per block for a steady run (left empty)
        t["steady_median"] = float("nan")
    for kind in ("first_seen", "revisit", "post_first_seen"):
        if f"{kind}_median" in t:
            t[f"{kind}_over_steady"] = t[f"{kind}_median"] / t["steady_median"]
    return t.sort_values(["model", "backend", "shape_value"])


def counters_table(runs, sessions):
    inst = sessions[(sessions["status"] == "ok") & sessions["instrumented"]][KEYS]
    r = runs.merge(inst, on=KEYS)   # runs.csv carries the experiment's mode
    if r.empty:
        return pd.DataFrame()
    cols = [c for c in COUNTER_COLS if c in r]
    parts = []
    lat = r[r["mode"] == "latency"]
    if not lat.empty:
        first = lat[(lat["phase"] == "warmup") & (lat["i"] == 0)].assign(kind="first_run")
        last_warm = lat[lat["phase"] == "warmup"]
        last_warm = last_warm[last_warm["i"] == last_warm.groupby(KEYS)["i"].transform("max")].assign(kind="last_warmup")
        parts += [first, last_warm]
    shp = r[(r["mode"] == "shapes") & (r["phase"] == "shape")]
    if not shp.empty:
        parts.append(classify_shapes(shp).assign(kind=lambda d: "shape_" + d["kind"]))
    if not parts:
        return pd.DataFrame()
    x = pd.concat(parts, ignore_index=True)
    x["pipelines_or_programs"] = x[[c for c in COMPILE_COLS if c in x]].fillna(0).sum(axis=1)
    agg = x.groupby(["model", "backend", "mode", "kind"])[cols + ["pipelines_or_programs"]].median().reset_index()
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="output directory (default: the first input dir)")
    args = ap.parse_args()
    runs, sessions = load(args.dirs)
    out = Path(args.out or args.dirs[0])
    out.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        per_launch = per_launch_init(runs, sessions)
        if not per_launch.empty:
            init = init_table(per_launch)
            init.to_csv(out / "init_summary.csv", index=False)
            show = ["model", "backend", "trials"] + [c for c in init if c.endswith("_median")]
            print("Initialization (medians over trials, ms):")
            print(init[show].round(2).to_string(index=False))
            pre = precompile_table(per_launch)
            if not pre.empty:
                pre.to_csv(out / "precompile_summary.csv", index=False)
                print("\nPre-compilation vs cold (ratio = precompile / cold):")
                print(pre.round(3).to_string(index=False))
        shapes = shapes_table(runs, sessions)
        if not shapes.empty:
            shapes.to_csv(out / "shape_reuse_summary.csv", index=False)
            print("\nShape reuse (median ms):")
            print(shapes.round(3).to_string(index=False))
        order = shape_order_table(runs, sessions)
        if not order.empty:
            order.to_csv(out / "shape_order_summary.csv", index=False)
            print("\nFirst-seen shapes by new-shape order (1 = first departure from the base shape):")
            print(order.round(2).to_string(index=False))
        counters = counters_table(runs, sessions)
        if not counters.empty:
            counters.to_csv(out / "first_run_counters.csv", index=False)
            print("\nInstrumented launches: per-run counters (medians):")
            print(counters.round(1).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
