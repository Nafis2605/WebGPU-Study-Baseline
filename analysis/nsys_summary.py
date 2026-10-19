"""
Summarise Nsight Systems captures (nsight/run_nsys.py) of the TF.js page. Diagnostic: every value is
trace-derived and labelled as such; the study's own measurements are not replaced.

Per capture (one row of nsys_summary.csv):
  page        runs in the collection window (from window.__nsight.stats(), epoch timestamps mapped
              onto the trace axis via TARGET_INFO_SESSION_START_TIME), inference rate, page p50
  D3D12 GPU   (DX12_WORKLOAD = GPU execution of each ExecuteCommandLists; WebGPU via Dawn)
              workloads per inference, workload duration p50/p90, GPU busy = union of workload
              intervals; busy fraction of the window and inside inference windows; idle gaps
              between consecutive workloads (p50/p90/max); lead = inference start -> first
              workload start; tail = last workload end -> inference end
  D3D12 CPU   per inference: Dispatch, CopyBufferRegion, ExecuteCommandLists, CreatePlacedResource,
              Map calls; ExecuteCommandLists CPU duration p50
  D3D11       call counts per inference (WebGL via ANGLE)
  GPU metrics (GPU_METRICS, device-wide, ~10 kHz): over inference windows: GPC clock mean and
              share of samples >= 1700 MHz (boost), GR Active / SMs Active / Sync Copy Engine
              Active means; same over the whole window

timeline_<capture>.csv  first --timeline-ms of the window: workload intervals and metric samples
                        (for figures)

Usage:
    python analysis/nsys_summary.py <capture dirs or parent> [--out DIR]
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"GPC Clock Frequency [MHz]": "gpc_clock_mhz", "GR Active [Throughput %]": "gr_active_pct",
           "SMs Active [Throughput %]": "sms_active_pct", "Sync Copy Engine Active [Throughput %]": "copy_active_pct",
           "GPU Active [Throughput %]": "gpu_active_pct", "Compute Warps in Flight [Throughput %]": "compute_warps_pct",
           "DRAM Read Bandwidth [Throughput %]": "dram_read_pct"}
BOOST_MHZ = 1700


def q(c, sql, *a):
    try:
        return pd.read_sql_query(sql, c, params=a)
    except Exception:  # noqa: BLE001  (table absent in this capture)
        return pd.DataFrame()


def union_length(starts, ends, lo=None, hi=None):
    """Total length covered by [start, end) intervals, optionally clipped to [lo, hi)."""
    if len(starts) == 0:
        return 0.0
    s, e = np.asarray(starts, float), np.asarray(ends, float)
    if lo is not None:
        s, e = np.maximum(s, lo), np.minimum(e, hi)
        keep = e > s
        s, e = s[keep], e[keep]
    if len(s) == 0:
        return 0.0
    o = np.argsort(s)
    s, e = s[o], e[o]
    total, cs, ce = 0.0, s[0], e[0]
    for a, b in zip(s[1:], e[1:]):
        if a > ce:
            total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return total + ce - cs


def capture_name(d):
    return d.name


def summarise(d, timeline_ms):
    db = d / "report.sqlite"
    meta = json.loads((d / "page_stats.json").read_text(encoding="utf-8"))
    cmd = json.loads((d / "command.json").read_text(encoding="utf-8"))
    parts = d.name.split("__")
    row = {"capture": d.name, "model": parts[0], "config": parts[1], "mode": parts[2]}
    c = sqlite3.connect(db)
    st = q(c, "select utcEpochNs from TARGET_INFO_SESSION_START_TIME")
    t0_ns = float(st["utcEpochNs"].iloc[0])
    # Analysis window: from collection start to the end of what both the trace and the page cover
    # (nsys may end collection, and terminate Chrome, before --duration; the page's last runs are
    # those fetched before Chrome went away).
    ends = []
    for sql in ("select max(end) as t from DX12_WORKLOAD", "select max(end) as t from CUPTI_ACTIVITY_KIND_KERNEL",
                "select max(timestamp) as t from GPU_METRICS"):
        v = q(c, sql)
        if len(v) and v["t"].iloc[0] is not None:
            ends.append(float(v["t"].iloc[0]))
    dur_ns = min(ends) if ends else float(meta.get("duration_s") or 0) * 1e9

    # Page inferences on the trace axis (ns from session start).
    runs = pd.DataFrame((meta.get("stats") or {}).get("runs") or [])
    skip = int((meta.get("window") or {}).get("skip_runs") or 0)   # warm-up runs (study phase)
    row["warmup_runs_excluded"] = skip
    runs = runs.iloc[skip:].reset_index(drop=True)
    if len(runs):
        runs["s"] = runs["start_epoch_ms"] * 1e6 - t0_ns
        runs["e"] = runs["end_epoch_ms"] * 1e6 - t0_ns
        if len(runs):
            dur_ns = min(dur_ns, float(runs["e"].max()))
        runs = runs[(runs["s"] >= 0) & (runs["e"] <= dur_ns)].reset_index(drop=True)
    row["window_ms"] = dur_ns / 1e6
    n = len(runs)
    row["inferences"] = n
    row["inference_rate_hz"] = n / (dur_ns / 1e9) if n else np.nan
    row["page_p50_ms"] = float(((runs["e"] - runs["s"]) / 1e6).median()) if n else np.nan
    row["page_note"] = (meta.get("error") or "") if not n else ""

    # D3D12 GPU workloads (GPU execution per ExecuteCommandLists).
    w = q(c, "select start, end, correlationId from DX12_WORKLOAD where gpu = 0 order by start")
    row["gpu_api"] = "d3d12" if len(w) else ""
    if not len(w):
        # Native TensorFlow (CUDA): GPU work = kernels + memory copies (CUPTI activity tables).
        kern = q(c, "select start, end from CUPTI_ACTIVITY_KIND_KERNEL")
        mcp = q(c, "select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY")
        if len(kern):
            row["gpu_api"] = "cuda"
            w = pd.concat([kern.assign(kind="kernel"), mcp.assign(kind="memcpy")], ignore_index=True).sort_values("start")
            if n:
                rs, re_ = runs["s"].to_numpy(), runs["e"].to_numpy()
                def per_inf(df):
                    k = np.searchsorted(rs, df["start"].to_numpy(), side="right") - 1
                    ok = (k >= 0) & (df["start"].to_numpy() < re_[np.clip(k, 0, n - 1)])
                    return df[ok]
                kin, min_ = per_inf(kern), per_inf(mcp)
                row["kernels_per_inference"] = len(kin) / n
                row["memcpy_per_inference"] = len(min_) / n
                row["memcpy_bytes_per_inference"] = float(min_["bytes"].sum()) / n if len(min_) else 0.0
    w = w[(w["end"] > 0) & (w["start"] < dur_ns)] if len(w) else w
    row["gpu_workloads"] = len(w)
    if len(w):
        wd = (w["end"] - w["start"]) / 1e6
        row["workload_ms_p50"], row["workload_ms_p90"] = float(wd.median()), float(wd.quantile(0.9))
        busy = union_length(w["start"], w["end"], 0, dur_ns)
        row["gpu_busy_frac_window"] = busy / dur_ns
        gaps = (w["start"].to_numpy()[1:] - np.maximum.accumulate(w["end"].to_numpy())[:-1]) / 1e6
        gaps = gaps[gaps > 0]
        if len(gaps):
            row["idle_gap_ms_p50"], row["idle_gap_ms_p90"] = float(np.median(gaps)), float(np.quantile(gaps, 0.9))
            row["idle_gap_ms_max"] = float(gaps.max())
        if n:
            ws, we = w["start"].to_numpy(), w["end"].to_numpy()
            per, busy_in, lead, tail = [], [], [], []
            for r in runs.itertuples():
                m = (we > r.s) & (ws < r.e)
                per.append(int(m.sum()))
                busy_in.append(union_length(ws[m], we[m], r.s, r.e) / (r.e - r.s))
                if m.any():
                    lead.append((ws[m].min() - r.s) / 1e6)
                    tail.append((r.e - we[m].max()) / 1e6)
            row["workloads_per_inference"] = float(np.median(per))
            row["gpu_busy_frac_in_inference"] = float(np.median(busy_in))
            row["gpu_busy_ms_per_inference"] = float(np.median(np.asarray(busy_in) * (runs["e"] - runs["s"]).to_numpy() / 1e6))
            if lead:
                row["lead_ms_p50"], row["tail_ms_p50"] = float(np.median(lead)), float(np.median(tail))

    # D3D12 / D3D11 API calls per inference.
    api = q(c, "select a.start, a.end, s.value as name from DX12_API a join StringIds s on s.id = a.nameId "
               "where a.start >= 0 and a.start < ?", dur_ns)
    # Count only calls issued inside the measured inferences (not warm-up runs or idle gaps).
    inside_runs = lambda t: (np.searchsorted(runs["s"].to_numpy(), t, side="right") - 1)  # noqa: E731
    if len(api) and n:
        k = inside_runs(api["start"].to_numpy())
        ok = (k >= 0) & (api["start"].to_numpy() < runs["e"].to_numpy()[np.clip(k, 0, n - 1)])
        api = api[ok]
    for short, full in [("dispatch", "ID3D12GraphicsCommandList::Dispatch"),
                        ("copy_buffer_region", "ID3D12GraphicsCommandList::CopyBufferRegion"),
                        ("execute_command_lists", "ID3D12CommandQueue::ExecuteCommandLists"),
                        ("create_placed_resource", "ID3D12Device::CreatePlacedResource"),
                        ("map", "ID3D12Resource::Map")]:
        k = int((api["name"] == full).sum()) if len(api) else 0
        row[f"{short}_per_inference"] = k / n if n else np.nan
    if len(api):
        ecl = api[api["name"] == "ID3D12CommandQueue::ExecuteCommandLists"]
        if len(ecl):
            row["ecl_cpu_us_p50"] = float(((ecl["end"] - ecl["start"]) / 1e3).median())
    d11 = q(c, "select a.start, s.value as name from DX11_API a join StringIds s on s.id = a.nameId "
               "where a.start >= 0 and a.start < ?", dur_ns)
    if len(d11) and n:
        k = inside_runs(d11["start"].to_numpy())
        d11 = d11[(k >= 0) & (d11["start"].to_numpy() < runs["e"].to_numpy()[np.clip(k, 0, n - 1)])]
        d11 = d11.groupby("name").size().rename("n").reset_index().sort_values("n", ascending=False)
    if len(d11) and n:
        row["dx11_calls_per_inference"] = float(d11["n"].sum() / n)
        row["dx11_top"] = "; ".join(f"{r.name.split('::')[-1]} {r.n / n:.1f}" for r in d11.head(6).itertuples())

    # GPU metrics (device-wide).
    names = q(c, "select metricId, metricName from TARGET_INFO_GPU_METRICS")
    gm = q(c, "select timestamp, metricId, value from GPU_METRICS where timestamp >= 0 and timestamp < ?", dur_ns)
    tl_metrics = pd.DataFrame()
    if len(gm) and len(names):
        idmap = {r.metricId: METRICS[r.metricName] for r in names.itertuples() if r.metricName in METRICS}
        gm = gm[gm["metricId"].isin(idmap)]
        gm["metric"] = gm["metricId"].map(idmap)
        piv = gm.pivot_table(index="timestamp", columns="metric", values="value", aggfunc="mean").sort_index()
        if "gpc_clock_mhz" in piv:
            piv["gpc_clock_mhz"] = piv["gpc_clock_mhz"] / 1e6   # reported in Hz
        for col in piv.columns:
            row[f"{col}_window_mean"] = float(piv[col].mean())
        if n:
            ts = piv.index.to_numpy()
            inside = np.zeros(len(ts), bool)
            for r in runs.itertuples():
                inside |= (ts >= r.s) & (ts < r.e)
            pin = piv[inside]
            row["metric_samples_in_inference"] = int(inside.sum())
            for col in piv.columns:
                row[f"{col}_in_inference_mean"] = float(pin[col].mean())
            if "gpc_clock_mhz" in pin:
                row["boost_share_in_inference"] = float((pin["gpc_clock_mhz"] >= BOOST_MHZ).mean())
        lo = float(runs["s"].min()) if n else 0.0
        tl_metrics = piv[(piv.index >= lo) & (piv.index < lo + timeline_ms * 1e6)].reset_index()
    nv = d / "nvml_clock.csv"
    if nv.exists():
        nvd = pd.read_csv(nv)
        if len(nvd):
            nvd["t"] = nvd["epoch_s"] * 1e9 - t0_ns
            nvw = nvd[(nvd["t"] >= 0) & (nvd["t"] < dur_ns)]
            if len(nvw):
                row["nvml_sm_mhz_median"] = float(nvw["sm_mhz"].median())
                row["nvml_sm_mhz_max"] = float(nvw["sm_mhz"].max())
                row["nvml_mem_mhz_median"] = float(nvw["mem_mhz"].median())
                row["nvml_util_median"] = float(nvw["util"].median())
    flat = [a for v in (cmd.values() if isinstance(cmd, dict) else [cmd]) for a in v]
    row["gpu_metrics_sampled"] = "--gpu-metrics-devices" in flat
    row["api_trace"] = flat[flat.index("--trace") + 1] if "--trace" in flat else ""
    row["clock_after"] = json.loads((d / "clock_check.json").read_text(encoding="utf-8")).get("after")
    row["chrome_extra"] = " ".join(a for a in flat if a.startswith("--disable-gpu-sandbox") or a.startswith("--in-process-gpu"))
    # Timelines start at the first measured inference (the window begins with warm-up / compilation).
    lo = float(runs["s"].min()) if n else 0.0
    hi = lo + timeline_ms * 1e6
    tl_work = w[(w["end"] > lo) & (w["start"] < hi)].assign(kind="workload") if len(w) else pd.DataFrame()
    tl_runs = runs[(runs["s"] >= lo) & (runs["s"] < hi)][["s", "e"]].rename(columns={"s": "start", "e": "end"}).assign(kind="inference") if n else pd.DataFrame()
    return row, tl_work, tl_runs, tl_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None, help="default: the parent folder of the first capture path")
    ap.add_argument("--timeline-ms", type=float, default=200)
    args = ap.parse_args()
    dirs = []
    for p in map(Path, args.paths):
        dirs += [p] if (p / "report.sqlite").exists() else sorted(x for x in p.iterdir() if (x / "report.sqlite").exists())
    out = Path(args.out) if args.out else Path(args.paths[0]).resolve().parent
    (out / "timelines").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in dirs:
        try:
            row, tw, tr, tm = summarise(d, args.timeline_ms)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {d.name}: {type(e).__name__}: {e}")
            continue
        rows.append(row)
        pd.concat([tw, tr], ignore_index=True).to_csv(out / "timelines" / f"{d.name}__intervals.csv", index=False)
        tm.to_csv(out / "timelines" / f"{d.name}__metrics.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(out / "nsys_summary.csv", index=False)
    cols = ["model", "config", "mode", "inferences", "page_p50_ms", "workloads_per_inference", "workload_ms_p50",
            "gpu_busy_frac_in_inference", "idle_gap_ms_p50", "lead_ms_p50", "tail_ms_p50", "dispatch_per_inference",
            "copy_buffer_region_per_inference", "gpc_clock_mhz_in_inference_mean", "boost_share_in_inference",
            "gr_active_pct_in_inference_mean", "sms_active_pct_in_inference_mean"]
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(t[[x for x in cols if x in t]].round(3).to_string(index=False))
    print(f"\n{len(t)} captures -> {out / 'nsys_summary.csv'}")


if __name__ == "__main__":
    main()
