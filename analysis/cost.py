"""
Group 5 cost of inference: CPU time and GPU energy per inference (latency-mode timing launches).

cpu_cost.csv      psutil CPU time of this launch's Chrome processes, per process type (renderer,
                  gpu-process, browser, utility, ...), over the measured-run window:
                    cpu_ms_per_inference = delta(user + system CPU s) * 1000 / runs in the window
                    cores_busy           = delta(CPU s) / window seconds
                  (the ONNX study's formula; psutil samples every 100 ms and Windows accounts CPU
                  time in ~15.6 ms ticks, so short windows are coarse). Median over trials.
energy.csv        analysis/gpu_telemetry.py (NVML, device-wide: a proxy), median over trials:
                  energy per inference, dynamic energy per inference, mean/idle power, utilization,
                  clocks, throttle reasons; window_ok = window >= 1 s.

Usage:
    python analysis/cost.py <latency results dir> [<dir> ...] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import gpu_telemetry

KEY = ["trial", "model", "backend"]


def _ok_measure_runs(d):
    s = pd.read_csv(d / "sessions.csv")
    inst = s["instrumented"].astype(str).str.lower().isin(["true", "1", "1.0"]) if "instrumented" in s else False
    ok = s[(s["status"] == "ok") & (s["mode"] == "latency") & ~inst][KEY]
    r = pd.read_csv(d / "runs.csv", low_memory=False)
    return r[r["phase"] == "measure"].merge(ok, on=KEY)


def cpu_cost(d):
    path = d / "chrome_memory_samples.csv"
    if not path.exists():
        return pd.DataFrame()
    cm = pd.read_csv(path)
    if "cpu_user_s" not in cm:
        return pd.DataFrame()
    r = _ok_measure_runs(d)
    rows = []
    for (trial, m, c), g in r.groupby(KEY):
        w0, w1 = g["t_start_epoch_ms"].min(), (g["t_start_epoch_ms"] + g["dur_ms"]).max()
        c_ = cm[(cm["trial"] == trial) & (cm["model"] == m) & (cm["backend"] == c)]
        for ptype, pg in c_.groupby("process_type"):
            pg = pg.sort_values("t_epoch_ms")
            inw = pg[(pg["t_epoch_ms"] >= w0) & (pg["t_epoch_ms"] <= w1)]
            if len(inw) < 2:
                continue
            cpu = inw["cpu_user_s"] + inw["cpu_system_s"]
            dt = (inw["t_epoch_ms"].iloc[-1] - inw["t_epoch_ms"].iloc[0]) / 1000
            n = ((g["t_start_epoch_ms"] >= inw["t_epoch_ms"].iloc[0]) & (g["t_start_epoch_ms"] <= inw["t_epoch_ms"].iloc[-1])).sum()
            if n == 0:
                continue
            rows.append({"source": d.name, "trial": trial, "model": m, "backend": c, "process": ptype,
                         "cpu_ms_per_inference": (cpu.iloc[-1] - cpu.iloc[0]) * 1000 / n,
                         "cores_busy": (cpu.iloc[-1] - cpu.iloc[0]) / dt if dt > 0 else np.nan,
                         "window_s": dt})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dirs = [Path(d) for d in args.dirs]
    out = Path(args.out or dirs[0])
    out.mkdir(parents=True, exist_ok=True)

    cpu = pd.concat([cpu_cost(d) for d in dirs], ignore_index=True)
    energy = []
    for d in dirs:
        if (d / "gpu_samples.csv").exists():
            e = gpu_telemetry.analyse(d)
            if len(e):
                energy.append(e.assign(source=d.name))
    energy = pd.concat(energy, ignore_index=True) if energy else pd.DataFrame()

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        if len(cpu):
            agg = cpu.groupby(["model", "backend", "process"])[["cpu_ms_per_inference", "cores_busy", "window_s"]].median().reset_index()
            agg.to_csv(out / "cpu_cost.csv", index=False)
            print("CPU time per inference (median over trials):")
            print(agg[agg["process"].isin(["renderer", "gpu-process"])].round(3).to_string(index=False))
        else:
            print("No CPU-time telemetry.")
        if len(energy):
            cols = [c for c in energy.columns if c not in ("trial", "source", "pstates", "clock_reasons")]
            eagg = energy.groupby(["model", "backend"])[[c for c in cols if c not in ("model", "backend")]].median(numeric_only=True).reset_index()
            reasons = energy.groupby(["model", "backend"])["clock_reasons"].agg(lambda x: "|".join(sorted(set(map(str, x)))))
            eagg = eagg.merge(reasons.reset_index(), on=["model", "backend"])
            eagg.to_csv(out / "energy.csv", index=False)
            show = [c for c in ["model", "backend", "energy_per_inference_mj", "dynamic_energy_per_inference_mj",
                                "mean_power_w", "idle_power_w", "gpu_util_mean_pct", "sm_clock_mean_mhz",
                                "window_ok", "clock_reasons"] if c in eagg]
            print("\nGPU energy (NVML, device-wide; median over trials):")
            print(eagg[show].round(3).to_string(index=False))
        else:
            print("No GPU telemetry.")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
