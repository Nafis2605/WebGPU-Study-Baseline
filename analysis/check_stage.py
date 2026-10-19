"""
Post-stage checks for a run_study.py stage (port of the sibling study's check_stage.py):
  - launches recorded vs planned (trials x models x configs from env.json, minus exclusions)
  - outcome per config (ok / backend_not_verified / strict_violation / failed + failure category)
  - strict configs: forwarded_kernels == 0 on every ok launch
  - GPU configs: dispatches (WebGPU) or draws (WebGL) > 0 on every ok launch, none on WASM
  - launches whose measure phase was cut by the time cap
  - optional reference dir (e.g. the group-2 verification run): per (model, config) median of
    launch medians here vs there

Usage:
    python analysis/check_stage.py <stage result dir> [--reference <dir>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bool(col):
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    env = json.loads((d / "env.json").read_text(encoding="utf-8"))
    a = env["args"]
    s = pd.read_csv(d / "sessions.csv")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_benchmark import CONFIGS, KNOWN_GAPS, TRAIN_MODES  # noqa: E402
    spec_p = Path(__file__).resolve().parents[1] / "public" / "spec" / "models.json"
    reg = json.loads(spec_p.read_text(encoding="utf-8"))["models"]
    models = [m for m in reg if a.get("filter") is None or any(m["name"] == f or m["name"].startswith(f) for f in a["filter"])]
    native = env.get("runtime") == "native-tf"
    pillar = "train" if a["mode"] in TRAIN_MODES else "infer"

    def skipped(m, c):
        if native or not a.get("skip_known_gaps"):
            return False
        b = CONFIGS[c]["backend"]
        return (m["family"], b, pillar) in KNOWN_GAPS or (m["name"], b, pillar) in KNOWN_GAPS
    excl = [e.split(":", 1) for e in (a.get("exclude") or [])]
    planned = sum(1 for _ in range(a["trials"]) for m in models for c in a["config"]
                  if not skipped(m, c) and not any(m["name"] == sub and c == cfg for sub, cfg in excl))
    print(f"{d}\nmode {a['mode']}: {len(s)} launches recorded, {planned} planned"
          + ("" if len(s) == planned else "  <-- MISMATCH"))

    s["failure_category"] = s["failure_category"].fillna("")
    out = s.groupby(["backend", "status"]).size().unstack(fill_value=0)
    print("\noutcome per config:\n" + out.to_string())
    bad = s[s["status"] != "ok"]
    if len(bad):
        print("\nnot ok:")
        print(bad.groupby(["model", "backend", "status", "failure_category"]).size().rename("n").reset_index().to_string(index=False))
        msgs = bad.drop_duplicates(["model", "backend"])[["model", "backend", "failure_message"]]
        for r in msgs.itertuples():
            print(f"  {r.model}:{r.backend}: {str(r.failure_message)[:200]}")

    ok = s[s["status"] == "ok"]
    strict = ok[_bool(ok["strict"])] if "strict" in ok else ok.iloc[0:0]
    if len(strict):
        v = strict[strict["forwarded_kernels"].fillna(0) > 0]
        print(f"\nstrict launches ok: {len(strict)}, with forwarded kernels: {len(v)}" + ("  <-- VIOLATION" if len(v) else ""))
    sig = {"webgpu": ("webgpu_dispatches", "webgl_draws"), "webgl": ("webgl_draws", "webgpu_dispatches")}
    for fam, (need, forbid) in sig.items():
        g = ok[ok["backend"].str.startswith(fam)]
        if len(g) and need in g:
            n_bad = int(((g[need].fillna(0) <= 0) | (g[forbid].fillna(0) > 0)).sum())
            print(f"{fam} signature: {len(g)} ok launches, {n_bad} without the expected API activity"
                  + ("  <-- CHECK" if n_bad else ""))
    if "measure_capped" in s:
        cap = s[_bool(s["measure_capped"])]
        if len(cap):
            print(f"\ncapped launches: {len(cap)}")
            print(cap.groupby(["model", "backend"])["n_measure"].agg(["size", "min", "max"]).to_string())

    # GPU clock regime per launch (NVML): a launch whose GPU never exceeded 1000 MHz under load ran in
    # a low-clock regime (e.g. the driver's "GPU idle" state persisting; the pre-stage guard only
    # checks that the GPU can idle). Cross-stage ratios involving such launches are not comparable.
    if (d / "gpu_samples.csv").exists():
        g = pd.read_csv(d / "gpu_samples.csv", usecols=["trial", "model", "backend", "sm_clock_mhz"])
        g = g[g["backend"].str.startswith(("webgl", "webgpu"))]
        if len(g):
            mx = g.groupby(["trial", "model", "backend"])["sm_clock_mhz"].max()
            low = mx[mx <= 1000]
            print(f"\nGPU launches whose SM clock never exceeded 1000 MHz: {len(low)} of {len(mx)}"
                  + ("  <-- LOW-CLOCK REGIME" if len(low) else ""))

    if args.reference and (d / "runs.csv").exists():
        def med(path):
            r = pd.read_csv(Path(path) / "runs.csv", low_memory=False)
            ss = pd.read_csv(Path(path) / "sessions.csv")
            keep = ss[(ss["status"] == "ok") & ~_bool(ss.get("instrumented", pd.Series(False, index=ss.index)))]
            r = r.merge(keep[["trial", "model", "backend"]], on=["trial", "model", "backend"])
            r = r[r["phase"].isin(["measure", "train_measure"])]
            return r.groupby(["model", "backend", "trial"])["dur_ms"].median().groupby(["model", "backend"]).median()
        here, ref = med(d), med(args.reference)
        j = pd.concat([here.rename("here_ms"), ref.rename("reference_ms")], axis=1).dropna()
        j["ratio"] = j["here_ms"] / j["reference_ms"]
        print("\nvs reference (median of launch medians):\n" + j.round(3).to_string())


if __name__ == "__main__":
    main()
