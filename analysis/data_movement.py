"""
Group 6: data movement.

transfer_summary.csv    from profile-mode launches (profile_counters phase), per (model, config),
                        medians per run of exact API-level byte counts and boundary counts:
                          upload_bytes    = webgl texImage/texSubImage bytes + WebGPU writeBuffer bytes
                                            + WebGPU mapped-at-creation bytes (TF.js uploads tensor
                                            data through mapped buffers)
                          readback_bytes  = webgl readPixels bytes + WebGPU mapAsync(READ) bytes
                          sync_reads      blocking GPU->CPU reads (readSync; kernels + dataSync)
                          async_reads     data() reads
                          forwarded_kernels  kernels run on the CPU because their inputs never left it
                          boundaries      sync_reads + async_reads (CPU-GPU execution boundaries)
                          read_share      read_ms / wall
io_summary.csv          latency-mode launches, per (model, config): median dur_ms, exec_ms,
                        gpu_done_ms (*-fence only: execute returned -> GPU finished), read_ms,
                        read share; with --baseline pairs: ratio of medians vs the default config
                        (webgl-gpuio/webgl-sync vs webgl, webgpu-gpuio/webgpu-sync vs webgpu)
readback_micro_summary.csv   readback micro-benchmark: per (path, bytes): n, median total_ms,
                        MB/s, median drain_ms

Usage:
    python analysis/data_movement.py [--profile DIR ...] [--latency DIR ...] [--micro DIR ...] --out DIR
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats import ratio_of_medians_ci

KEY = ["source", "trial", "model", "backend"]
PARENT = {"webgl-gpuio": "webgl", "webgl-sync": "webgl", "webgl-fence": "webgl",
          "webgpu-gpuio": "webgpu", "webgpu-sync": "webgpu", "webgpu-fence": "webgpu"}


def _bool(col):
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(dirs, mode):
    runs = []
    for d in map(Path, dirs):
        s = pd.read_csv(d / "sessions.csv").assign(source=d.name)
        inst = _bool(s["instrumented"]) if "instrumented" in s else False
        ok = s[(s["status"] == "ok") & (s["mode"] == mode) & (~inst if mode == "latency" else True)][KEY]
        r = pd.read_csv(d / "runs.csv", low_memory=False).assign(source=d.name)
        runs.append(r.merge(ok, on=KEY))
    return pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()


def col(df, name):
    return df[name] if name in df else pd.Series(0.0, index=df.index)


def transfer(profile_dirs):
    r = load(profile_dirs, "profile")
    r = r[r["phase"] == "profile_counters"].copy()
    if r.empty:
        return pd.DataFrame()
    r["upload_bytes"] = (col(r, "webgl_tex_upload_bytes").fillna(0) + col(r, "webgpu_write_buffer_bytes").fillna(0)
                         + col(r, "webgpu_mapped_at_creation_bytes").fillna(0))
    r["readback_bytes"] = col(r, "webgl_read_pixels_bytes").fillna(0) + col(r, "webgpu_map_read_bytes").fillna(0)
    r["boundaries"] = col(r, "hooks_sync_reads").fillna(0) + col(r, "hooks_async_reads").fillna(0)
    r["read_share"] = r["read_ms"] / r["dur_ms"]
    cols = {"upload_bytes": "upload_bytes", "readback_bytes": "readback_bytes",
            "webgl_get_buffer_sub_data_bytes": "webgl_pbo_bytes", "webgpu_copy_b2b_bytes": "webgpu_copy_b2b_bytes",
            "webgpu_map_read_wait_ms": "map_read_wait_ms", "hooks_sync_reads": "sync_reads",
            "hooks_sync_read_ms": "sync_read_ms", "hooks_async_reads": "async_reads",
            "hooks_forwarded_kernels": "forwarded_kernels", "boundaries": "boundaries",
            "read_ms": "read_ms", "dur_ms": "wall_ms", "read_share": "read_share"}
    have = {k: v for k, v in cols.items() if k in r}
    return r.groupby(["model", "backend"])[list(have)].median().rename(columns=have).reset_index()


def io_table(latency_dirs):
    r = load(latency_dirs, "latency")
    r = r[r["phase"] == "measure"]
    if r.empty:
        return pd.DataFrame()
    agg = {"n": ("dur_ms", "size"), "dur_p50_ms": ("dur_ms", "median"), "exec_p50_ms": ("exec_ms", "median"),
           "read_p50_ms": ("read_ms", "median")}
    if "gpu_done_ms" in r:
        agg["gpu_done_p50_ms"] = ("gpu_done_ms", "median")
    t = r.groupby(["model", "backend"]).agg(**agg).reset_index()
    t["read_share"] = t["read_p50_ms"] / t["dur_p50_ms"]
    rows = []
    for (m, c), g in r.groupby(["model", "backend"]):
        p = PARENT.get(c)
        if not p:
            continue
        base = r[(r["model"] == m) & (r["backend"] == p)]["dur_ms"]
        if base.empty:
            continue
        ratio, lo, hi = ratio_of_medians_ci(g["dur_ms"], base)
        rows.append({"model": m, "backend": c, "vs": p, "dur_ratio": ratio, "ci_lo": lo, "ci_hi": hi})
    return t.merge(pd.DataFrame(rows), on=["model", "backend"], how="left") if rows else t


def micro(dirs):
    frames = []
    for d in map(Path, dirs):
        f = d / "readback_micro.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    if "launch_backend" in x:
        # TF.js paths run on the launch's backend: keep them apart (raw API paths run once).
        tfjs = x["path"].str.startswith("tfjs_")
        x.loc[tfjs, "path"] = x.loc[tfjs, "path"] + "@" + x.loc[tfjs, "launch_backend"]
    agg = {"n": ("total_ms", "size"), "total_ms_median": ("total_ms", "median")}
    if "drain_ms" in x:
        agg["drain_ms_median"] = ("drain_ms", "median")
    t = x.groupby(["path", "bytes"]).agg(**agg).reset_index()
    t["mb_per_s"] = t["bytes"] / 1e6 / (t["total_ms_median"] / 1e3)
    return t.sort_values(["bytes", "path"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", nargs="*", default=[])
    ap.add_argument("--latency", nargs="*", default=[])
    ap.add_argument("--micro", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        if args.profile:
            t = transfer(args.profile)
            t.to_csv(out / "transfer_summary.csv", index=False)
            print("Per-run transfers (profile_counters medians):")
            print(t.round(3).to_string(index=False))
        if args.latency:
            t = io_table(args.latency)
            t.to_csv(out / "io_summary.csv", index=False)
            print("\nI/O variants (latency medians):")
            print(t.round(3).to_string(index=False))
        if args.micro:
            t = micro(args.micro)
            t.to_csv(out / "readback_micro_summary.csv", index=False)
            print("\nReadback micro-benchmark:")
            print(t.round(3).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
