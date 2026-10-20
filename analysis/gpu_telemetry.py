"""
NVIDIA GPU telemetry per experiment, aligned to each measured window (inference runs, training steps,
post-training inference). Port of the sibling TF.js study's file (itself the ONNX study's); generalised from
the latency "measure" window to every series in WINDOWS, with per-sample energy (records x batch).

Windows (Unix epoch ms, one clock for browser and NVML):
    baseline  idle GPU before the browser launched (run_benchmark.py --baseline-s)
    window    first measured record start .. last measured record end (runs.csv), per (mode, phase) in WINDOWS

Per experiment (status ok, latency mode):
    energy_j                 cumulative NVML energy counter, linearly interpolated at the window edges
                             (the counter refreshes about every 96 ms, so windows under ~1 s are flagged)
    mean_power_w             energy_j / window duration
    idle_power_w             mean power over the baseline window
    energy_per_inference_mj  energy_j / measured runs
    dynamic_energy_per_inference_mj  (mean_power_w - idle_power_w) * duration / runs
    gpu_util_mean_pct, mem_util_mean_pct   driver utilization samples inside the window (~200 ms apart)
    sm_clock_mean_mhz, mem_clock_mean_mhz, temp_max_c, pstates, clock_reasons (union of throttle reasons)
    mem_used_peak_delta_mb   peak device memory in the window minus the baseline mean (device-wide)
    pcie_tx_mean_kbps, pcie_rx_mean_kbps   PCIe throughput samples starting inside the window

Per-inference energy for a single run isn't resolvable: runs are shorter than the counter's refresh period.

Cross-check of the energy counter (added for the RTX 4090 laptop, where the counter implied about 120 W while the
sampled power and nvidia-smi showed about 25 W, above the 111 W board limit; the cause is not established):
    power_sampled_mean_w     time-weighted mean of the sampled power_w inside the window
    energy_sampled_j         the same samples integrated over the window (trapezoid)
    idle_power_sampled_w     mean sampled power over the baseline window
    counter_vs_sampled       mean_power_w / power_sampled_mean_w
    energy_counter_ok        True when that ratio is within COUNTER_TOLERANCE (0.75 .. 1.33)
The counter-based columns are unchanged and are not corrected. Where energy_counter_ok is False, quote the
sampled-power columns instead and say which source was used. Fields that NVML failed to read are empty in
gpu_samples.csv (see telemetry.field_errors in experiments.jsonl); they are skipped here, not interpolated over.

Usage:
    python analysis/gpu_telemetry.py <run folder>/<stage>/<timestamp>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchkit.telemetry import decode_clock_reasons  # noqa: E402

KEY = ["trial", "model", "backend"]
MIN_WINDOW_S = 1.0
COUNTER_TOLERANCE = (0.75, 1.33)   # energy-counter mean power / sampled mean power
# mode -> measured phases whose window gets an energy row
WINDOWS = {"latency": ["measure"], "train": ["train_measure", "infer_b1", "infer_bB"], "train_long": ["train_long"],
           "train_batch": ["train_batch"], "batch": ["batch"], "train_pipelined": ["train_pipelined"]}


def window_value(t, v, at):
    """Linear interpolation of a cumulative counter at time `at`, or NaN outside the samples."""
    if at < t[0] or at > t[-1]:
        return np.nan
    return float(np.interp(at, t, v))


def sampled_energy_j(t, p, w0, w1):
    """Integral of the sampled power (W, at epoch ms) over [w0, w1] in joules, or NaN if the window is not covered."""
    if len(t) < 2 or w0 < t[0] or w1 > t[-1] or w1 <= w0:
        return np.nan
    ts = np.concatenate([[w0], t[(t > w0) & (t < w1)], [w1]])
    vs = np.interp(ts, t, p)
    return float(np.sum((vs[1:] + vs[:-1]) / 2.0 * np.diff(ts)) / 1000.0)   # W x ms = mJ


def analyse(out_dir):
    out_dir = Path(out_dir)
    sessions = pd.read_csv(out_dir / "sessions.csv")
    runs = pd.read_csv(out_dir / "runs.csv")
    if not (out_dir / "gpu_samples.csv").exists():
        sys.exit(f"{out_dir}: no gpu_samples.csv, so this stage recorded no GPU samples (NVML errors: see "
                 "telemetry.errors / telemetry.field_errors in experiments.jsonl)")
    samples = pd.read_csv(out_dir / "gpu_samples.csv")
    util = pd.read_csv(out_dir / "gpu_util_samples.csv") if (out_dir / "gpu_util_samples.csv").exists() else pd.DataFrame(columns=KEY + ["t_epoch_ms", "kind", "percent"])
    pcie = pd.read_csv(out_dir / "gpu_pcie_samples.csv") if (out_dir / "gpu_pcie_samples.csv").exists() else pd.DataFrame(columns=KEY + ["t_epoch_ms", "tx_kbps", "rx_kbps"])
    marks = pd.read_csv(out_dir / "gpu_marks.csv")

    rows = []
    inst = sessions["instrumented"].astype(str).str.lower().isin(["true", "1", "1.0"]) if "instrumented" in sessions else False
    ok = sessions[(sessions["status"] == "ok") & sessions["mode"].isin(list(WINDOWS)) & ~inst]
    groups = []
    for _, s in ok.iterrows():
        k = tuple(s[c] for c in KEY)
        sel = lambda df, k=k: df[(df["trial"] == k[0]) & (df["model"] == k[1]) & (df["backend"] == k[2])]
        rs = sel(runs)
        for ph in WINDOWS[s["mode"]]:
            r_ph = rs[rs["phase"] == ph]
            if "batch" in r_ph and ph in ("batch", "train_batch"):
                groups += [(s, k, sel, ph, b, g) for b, g in r_ph.groupby("batch")]
            elif not r_ph.empty:
                groups.append((s, k, sel, ph, None, r_ph))
    for s, k, sel, phase, bval, r in groups:
        smp = sel(samples).sort_values("t_epoch_ms")
        mk = sel(marks).set_index("mark")["t_epoch_ms"]
        if r.empty or smp.empty or "baseline_start" not in mk:
            continue
        w0 = r["t_start_epoch_ms"].min()
        w1 = (r["t_start_epoch_ms"] + r["dur_ms"]).max()
        dur_s = (w1 - w0) / 1000.0
        # an NVML field that failed to read is empty: use only the samples where that field was read
        sm_e = smp.dropna(subset=["energy_mj"])
        sm_p = smp.dropna(subset=["power_w"])
        t = sm_e["t_epoch_ms"].to_numpy()
        energy = sm_e["energy_mj"].to_numpy(dtype=float)
        e_j = ((window_value(t, energy, w1) - window_value(t, energy, w0)) / 1000.0) if len(t) else np.nan
        e_samp_j = sampled_energy_j(sm_p["t_epoch_ms"].to_numpy(), sm_p["power_w"].to_numpy(dtype=float), w0, w1)

        base = smp[(smp["t_epoch_ms"] >= mk["baseline_start"]) & (smp["t_epoch_ms"] < mk["baseline_end"])]
        base_e = base.dropna(subset=["energy_mj"])
        b_t = base_e["t_epoch_ms"].to_numpy()
        idle_power = ((base_e["energy_mj"].iloc[-1] - base_e["energy_mj"].iloc[0]) / (b_t[-1] - b_t[0])
                      if len(base_e) > 1 and b_t[-1] > b_t[0] else np.nan)  # mJ/ms = W
        idle_power_sampled = base["power_w"].mean()
        inwin = smp[(smp["t_epoch_ms"] >= w0) & (smp["t_epoch_ms"] <= w1)]
        u = sel(util)
        u = u[(u["t_epoch_ms"] >= w0) & (u["t_epoch_ms"] <= w1)]
        p = sel(pcie)
        p = p[(p["t_epoch_ms"] >= w0) & (p["t_epoch_ms"] <= w1)]
        n = len(r)
        # samples processed in the window: records x batch (x steps for pipelined windows)
        per_rec = r["batch"].fillna(1) if "batch" in r else pd.Series(1, index=r.index)
        steps = r["window_steps"].fillna(1) if "window_steps" in r else pd.Series(1, index=r.index)
        n_samples = float((per_rec * steps).sum())
        mean_power = e_j / dur_s if dur_s > 0 else np.nan
        mean_power_sampled = e_samp_j / dur_s if dur_s > 0 else np.nan
        ratio = mean_power / mean_power_sampled if mean_power_sampled and mean_power_sampled > 0 else np.nan
        reasons = 0
        for m in inwin["clock_reasons"].dropna().astype(int):
            reasons |= m
        rows.append({
            **dict(zip(KEY, k)), "mode": s["mode"], "phase": phase, "batch": bval,
            "measured_runs": n, "samples": n_samples, "window_s": dur_s, "window_ok": dur_s >= MIN_WINDOW_S,
            "energy_j": e_j, "mean_power_w": mean_power, "idle_power_w": idle_power,
            "energy_sampled_j": e_samp_j, "power_sampled_mean_w": mean_power_sampled,
            "idle_power_sampled_w": idle_power_sampled, "counter_vs_sampled": ratio,
            "energy_counter_ok": bool(COUNTER_TOLERANCE[0] <= ratio <= COUNTER_TOLERANCE[1]) if ratio == ratio else None,
            # per record = per inference (inference phases) or per training step (train phases)
            "energy_per_inference_mj": e_j * 1000.0 / n,
            "dynamic_energy_per_inference_mj": (mean_power - idle_power) * dur_s * 1000.0 / n,
            "energy_per_sample_mj": e_j * 1000.0 / n_samples,
            "dynamic_energy_per_sample_mj": (mean_power - idle_power) * dur_s * 1000.0 / n_samples,
            "gpu_util_mean_pct": u.loc[u["kind"] == "gpu", "percent"].mean(),
            "mem_util_mean_pct": u.loc[u["kind"] == "mem", "percent"].mean(),
            "util_samples": int((u["kind"] == "gpu").sum()),
            "sm_clock_mean_mhz": inwin["sm_clock_mhz"].mean(),
            "mem_clock_mean_mhz": inwin["mem_clock_mhz"].mean(),
            "temp_max_c": inwin["temp_c"].max(),
            "pstates": ",".join(str(x) for x in sorted(inwin["pstate"].unique())),
            "clock_reasons": decode_clock_reasons(reasons),
            "mem_used_peak_delta_mb": inwin["mem_used_mb"].max() - base["mem_used_mb"].mean(),
            "pcie_tx_mean_kbps": p["tx_kbps"].mean(), "pcie_rx_mean_kbps": p["rx_kbps"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    args = ap.parse_args()
    t = analyse(args.dir)
    out = Path(args.dir) / "gpu_telemetry_summary.csv"
    t.to_csv(out, index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        cols = ["model", "backend", "trial", "mode", "phase", "batch", "window_s", "mean_power_w", "power_sampled_mean_w",
                "counter_vs_sampled", "energy_counter_ok", "idle_power_w", "idle_power_sampled_w", "energy_per_inference_mj",
                "dynamic_energy_per_inference_mj", "energy_per_sample_mj", "gpu_util_mean_pct", "sm_clock_mean_mhz", "temp_max_c", "clock_reasons",
                "mem_used_peak_delta_mb"]
        print(t[cols].round(2).to_string(index=False) if len(t) else "No telemetry for verified latency experiments.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
