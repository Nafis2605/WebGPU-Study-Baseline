"""
Memory and reliability (port of the sibling TF.js study's memory.py; this study adds training: profile and
train_profile launches are summarised separately, plus optimizer state and per-step tensor-count growth).

memory_summary.csv   profile-mode launches, per (model, config), medians over launches of the
                     src/memory.js snapshots (bytes):
                       weights_mb              model weight bytes (model.json shards)
                       tf_bytes_after_load     tf.memory().numBytes after the graph model is built
                       pool_allocated_*        backend pool bytes allocated (WebGPU buffer pool,
                                               WebGL texture pool) after first run / warm-up / runs
                       pool_used_after_runs, pool_overhead = allocated / used after the runs
                       logical_peak_mb         peak live bytes of page-created GPU buffers/textures
                                               (instrument.js) during the measured phases
                       wasm_heap_after_runs_mb WASM linear memory (the only WASM byte statistic)
                       page_memory_*_mb        measureUserAgentSpecificMemory()
                       retained_after_dispose_mb  pool bytes still allocated after model.dispose()
                                               (and tf.memory().numBytes left)
chrome_memory.csv    per (model, config, process type): peak private MB over the launch (psutil,
                     all modes given), median over launches
shape_memory.csv     shapes-mode launches: pool bytes allocated after each shape block, growth vs
                     the base-shape snapshot, split by first-seen vs revisited blocks
reliability.csv      per (config, mode): launches, ok, failed by failure category (all dirs given)

Usage:
    python analysis/memory.py <results dirs...> --out DIR
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["source", "trial", "model", "backend"]
MB = 2 ** 20


def load_sessions(dirs):
    return pd.concat([pd.read_csv(Path(d) / "sessions.csv").assign(source=Path(d).name) for d in dirs], ignore_index=True)


def memory_summary(s):
    p = s[(s["status"] == "ok") & s["mode"].isin(["profile", "train_profile"])].copy()
    if p.empty:
        return pd.DataFrame()
    def c(name):
        return p[name] if name in p else pd.Series(np.nan, index=p.index)
    p["weights_mb"] = c("model_bytes") / MB
    p["tf_bytes_after_load_mb"] = c("mem_after_load_tf_bytes") / MB
    for stage in ("after_first_run", "after_warmup", "after_runs"):
        p[f"pool_allocated_{stage}_mb"] = c(f"mem_{stage}_pool_bytes_allocated") / MB
    # WebGPU reports bytes used; the WebGL texture pool reports bytes free (used = allocated - free).
    used = c("mem_after_runs_pool_bytes_used").fillna(
        c("mem_after_runs_pool_bytes_allocated") - c("mem_after_runs_pool_bytes_free"))
    p["pool_used_after_runs_mb"] = used / MB
    p["pool_overhead"] = c("mem_after_runs_pool_bytes_allocated") / used
    p["logical_peak_mb"] = c("mem_after_runs_logical_peak_bytes") / MB
    p["logical_live_after_runs_mb"] = c("mem_after_runs_logical_live_bytes") / MB
    p["wasm_heap_after_runs_mb"] = c("mem_after_runs_wasm_heap_bytes") / MB
    for k in ("before_fetch", "after_session_create", "after_runs"):
        p[f"page_memory_{k}_mb"] = c(f"page_memory_{k}") / MB
    p["retained_pool_after_dispose_mb"] = c("mem_after_dispose_pool_bytes_allocated") / MB
    p["retained_tf_bytes_after_dispose_mb"] = c("mem_after_dispose_tf_bytes") / MB
    p["retained_wasm_heap_after_dispose_mb"] = c("mem_after_dispose_wasm_heap_bytes") / MB
    # Adam keeps two moments per parameter (float32): computed from the parameter count, labelled as such.
    p["optimizer_state_mb_computed"] = np.where(p["mode"] == "train_profile", c("params") * 2 * 4 / MB, np.nan)
    cols = [x for x in p.columns if x.endswith("_mb") or x == "pool_overhead" or x.endswith("_computed")]
    return p.groupby(["mode", "model", "backend"])[cols].median().reset_index()


def tensor_growth(s):
    """Training timing launches: tensors alive at the first warm-up step vs the last measured step, and the
    verification step's leak count (status leak_detected if non-zero)."""
    t = s[s["mode"] == "train"].copy()
    if t.empty or "num_tensors_first_step" not in t:
        return pd.DataFrame()
    t["tensor_growth"] = t["num_tensors_last_step"] - t["num_tensors_first_step"]
    return t.groupby(["model", "backend"]).agg(launches=("status", "size"),
                                               growth_max=("tensor_growth", "max"),
                                               leaked_max=("tensors_leaked", "max")).reset_index()


def chrome_memory(dirs):
    rows = []
    for d in map(Path, dirs):
        f = d / "chrome_memory_samples.csv"
        if not f.exists():
            continue
        cm = pd.read_csv(f)
        if "private_mb" not in cm:
            continue
        s = pd.read_csv(d / "sessions.csv")
        ok = s[s["status"] == "ok"][["trial", "model", "backend", "mode"]]
        cm = cm.merge(ok, on=["trial", "model", "backend"])
        peak = cm.groupby(["mode", "trial", "model", "backend", "process_type"])["private_mb"].max().reset_index()
        rows.append(peak)
    if not rows:
        return pd.DataFrame()
    x = pd.concat(rows, ignore_index=True)
    return x.groupby(["mode", "model", "backend", "process_type"])["private_mb"].median().rename("peak_private_mb").reset_index()


def shape_memory(dirs):
    frames = []
    for d in map(Path, dirs):
        f = d / "shape_memory.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(source=d.name))
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    # GPU backends: pool bytes allocated; WASM: linear-memory (heap) size.
    x["alloc"] = x["mem_pool_bytes_allocated"].fillna(x.get("mem_wasm_heap_bytes"))
    base = x[x["block"] == -1].set_index(KEY)["alloc"].rename("base_alloc")
    x = x[x["block"] >= 0].join(base, on=KEY)
    x["growth_mb"] = (x["alloc"] - x["base_alloc"]) / MB
    x["kind"] = np.where(x["first_seen"].astype(str).str.lower() == "true", "after_first_seen_block", "after_revisit_block")
    last = x.sort_values("block").groupby(KEY).tail(1)
    t = x.groupby(["model", "backend", "kind"])["growth_mb"].median().unstack("kind")
    t["final_growth_mb"] = last.groupby(["model", "backend"])["growth_mb"].median()
    return t.reset_index()


def reliability(s):
    s = s.copy()
    s["failure_category"] = s["failure_category"].fillna("")
    t = s.groupby(["backend", "mode"]).agg(launches=("status", "size"), ok=("status", lambda x: int((x == "ok").sum())))
    t["failure_rate"] = 1 - t["ok"] / t["launches"]
    cats = s[s["status"] != "ok"].groupby(["backend", "mode", "failure_category"]).size().rename("n").reset_index()
    if len(cats):
        cats["cat"] = cats["failure_category"].replace("", "status_not_ok")
        t = t.join(cats.pivot_table(index=["backend", "mode"], columns="cat", values="n", fill_value=0))
    return t.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = load_sessions(args.dirs)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        for name, t in (("memory_summary", memory_summary(s)), ("chrome_memory", chrome_memory(args.dirs)),
                        ("shape_memory", shape_memory(args.dirs)), ("tensor_growth", tensor_growth(s)),
                        ("reliability", reliability(s))):
            if len(t):
                t.to_csv(out / f"{name}.csv", index=False)
                print(f"\n{name}:")
                print(t.round(3).to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
