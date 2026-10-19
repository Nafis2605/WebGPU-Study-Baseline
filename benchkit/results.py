"""Result files. experiments.jsonl is the raw source of truth; the CSVs are derived from it.
(Port of the sibling study's results.py; schema 3 adds the training / warm-up fields.)"""
import csv
import json
from pathlib import Path

import pandas as pd


class ResultWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "logs").mkdir(exist_ok=True)
        self.jsonl = self.out_dir / "experiments.jsonl"

    def write_json(self, name, obj):
        (self.out_dir / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")

    def append_rows(self, name, rows):
        """Append rows to a CSV in the output dir, writing the header on first use."""
        if not rows:
            return
        path = self.out_dir / name
        header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if header:
                w.writeheader()
            w.writerows(rows)

    def append(self, record):
        """Append one experiment immediately, so a crash later in the sweep loses nothing."""
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def export_csvs(self):
        return export_csvs(self.out_dir)


def load_experiments(out_dir: Path):
    path = Path(out_dir) / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flatten(record):
    """Flatten nested per-run dicts (counters, memory) into prefixed columns; lists become JSON."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[k2 if k == "counters" else f"{k}_{k2}"] = v2
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _snapshot_cols(prefix, snap):
    """Flatten a src/memory.js snapshot: tf.memory() byte fields, backend pool stats, logical bytes."""
    if not isinstance(snap, dict) or "tf_memory" not in snap:
        return {}
    tm, pool, lg = snap.get("tf_memory") or {}, snap.get("pool") or {}, snap.get("logical") or {}
    return {
        f"{prefix}_tf_bytes": tm.get("numBytes"),
        f"{prefix}_tf_tensors": tm.get("numTensors"),
        # WebGL: numBytesInGPU / numBytesInGPUAllocated; WebGPU: numBytesInGPU / numBytesAllocatedInGPU
        f"{prefix}_tf_gpu_bytes": tm.get("numBytesInGPU"),
        f"{prefix}_tf_gpu_allocated_bytes": tm.get("numBytesInGPUAllocated", tm.get("numBytesAllocatedInGPU")),
        f"{prefix}_pool_bytes_allocated": pool.get("bytes_allocated"),
        f"{prefix}_pool_bytes_used": pool.get("bytes_used"),
        f"{prefix}_pool_bytes_free": pool.get("bytes_free"),
        f"{prefix}_wasm_heap_bytes": pool.get("heap_bytes"),
        f"{prefix}_logical_live_bytes": lg.get("live_bytes"),
        f"{prefix}_logical_peak_bytes": lg.get("peak_live_bytes"),
    }


def _mem(e, *path):
    node = e
    for key in path:
        node = (node or {}).get(key)
    return node


def export_csvs(out_dir: Path):
    """(Re)build sessions.csv, runs.csv and (profile mode) kernels.csv from experiments.jsonl."""
    out_dir = Path(out_dir)
    experiments = load_experiments(out_dir)
    sessions, runs, kernels = [], [], []
    extra = {name: [] for name in ("frames", "interactions", "clicks", "loaf", "phases", "load_requests", "load_frames",
                                   "shape_memory", "train_correctness", "long_val")}
    for e in experiments:
        # "backend" is the config label (e.g. wasm-4t, webgpu-strict); "ep" the TF.js backend name.
        key = {"trial": e["trial"], "model": e["model"], "backend": e.get("config_label", e["backend"])}
        v = e.get("verification") or {}
        c = v.get("counters") or {}
        h = v.get("hooks") or {}
        t = e.get("tfjs") or {}
        cfg = e.get("config") or {}
        f = e.get("failure") or {}
        all_runs = e.get("runs", [])
        measured = [r for r in all_runs if r["phase"] in ("measure", "train_measure")]
        tw = [r for r in all_runs if r["phase"] in ("train_warmup", "lc_train1_warmup", "lc_train_warmup")]
        iw = [r for r in all_runs if r["phase"] in ("warmup", "lc_infer1_warmup")]
        sessions.append({
            **key,
            "family": e.get("family"),
            "size": e.get("size"),
            "params": e.get("params"),
            "pillar": e.get("pillar"),
            "mode": e.get("mode", "latency"),
            "ep": e["backend"],
            "strict": cfg.get("strict"),
            "wasm_threads": t.get("wasm_threads"),
            "flags_effective": json.dumps(t.get("flags_effective")) if t.get("flags_effective") else None,
            "exec_mode": cfg.get("exec"),
            "outputs": cfg.get("outputs"),
            "tfjs_version": t.get("tfjs_version"),
            "tfjs_layers_version": t.get("tfjs_layers_version"),
            "backend_epsilon": t.get("backend_epsilon"),
            "float_precision": t.get("float_precision"),
            "weights_kind": cfg.get("weights"),
            "weights_sha256": e.get("weights_sha256"),
            "train_batch": cfg.get("train_batch"),
            "infer_batch": cfg.get("infer_batch"),
            "sync_policy": cfg.get("sync_policy"),
            "data_io": cfg.get("data_io"),
            "io": cfg.get("io"),
            "fence": cfg.get("fence"),
            "readback": cfg.get("readback"),
            "webgpu_timestamp_query": t.get("timestamp_query"),
            "webgl_timer_reliable": t.get("webgl_timer_reliable"),
            "status": e["status"],
            "failure_stage": f.get("stage"),
            "failure_category": f.get("category"),
            "failure_message": f.get("message"),
            "known_gap": e.get("known_gap"),
            "precompile": cfg.get("precompile"),
            "instrumented": cfg.get("instrumented"),
            "shape_sequence": json.dumps(cfg.get("shape_sequence")) if cfg.get("shape_sequence") else None,
            "backend_init_ms": e.get("backend_init_ms"),
            "model_bytes": e.get("model_bytes"),
            "fetch_ms": e.get("fetch_ms"),
            "weights_verify_ms": e.get("weights_verify_ms"),
            "data_fetch_ms": e.get("data_fetch_ms"),
            "data_verify_ms": e.get("data_verify_ms"),
            "build_ms": e.get("build_ms"),
            "weights_assign_ms": e.get("weights_assign_ms"),
            "compile_ms": e.get("compile_ms"),
            # TF.js: graph build + weight tensor creation (GPU weight upload is lazy -> first run).
            "session_create_ms": e.get("session_create_ms"),
            # TF.js compile-only pass before the first run (webgl/webgpu-precompile configs).
            "precompile_ms": e.get("precompile_ms"),
            "precompile_programs_linked": _mem(e, "precompile_counters", "webgl_programs_linked"),
            "precompile_pipelines_created": ((_mem(e, "precompile_counters", "webgpu_pipelines_created") or 0)
                                             + (_mem(e, "precompile_counters", "webgpu_pipelines_created_async") or 0))
                                            if e.get("precompile_counters") else None,
            # The first and second inferences of the session (inference pillar: warm-up runs 0 and 1).
            "first_run_ms": iw[0]["dur_ms"] if len(iw) > 0 else None,
            "second_run_ms": iw[1]["dur_ms"] if len(iw) > 1 else None,
            # The first and second training steps (training pillar: train_warmup steps 0 and 1).
            "first_step_ms": tw[0]["dur_ms"] if len(tw) > 0 else None,
            "first_step_exec_ms": tw[0].get("exec_ms") if len(tw) > 0 else None,
            "second_step_ms": tw[1]["dur_ms"] if len(tw) > 1 else None,
            "num_tensors_first_step": tw[0].get("num_tensors") if tw else None,
            "num_tensors_last_step": measured[-1].get("num_tensors") if measured else None,
            "n_measure": len(measured),
            # Phases cut short by the measurement time cap (--max-measure-s): fewer samples, not estimates.
            "measure_capped": bool(e.get("capped")),
            "capped_phases": json.dumps(e.get("capped")) if e.get("capped") else None,
            # Locked environment and GPU power-state guard (every launch).
            "env_lock_sha256": e.get("env_lock_sha256"),
            "gpu_sm_clock_max_mhz": e.get("gpu_sm_clock_max_mhz"),
            "low_clock_reruns": e.get("low_clock_reruns"),
            # Native TensorFlow launches (native/worker.py).
            "native_gpu_memory_peak_bytes": (e.get("native_gpu_memory") or {}).get("peak_bytes"),
            "native_tf32": (e.get("tfjs") or {}).get("tf32_enabled"),
            "native_input_casts": json.dumps(e.get("input_casts")) if e.get("input_casts") else None,
            "backend_verified": v.get("verified"),
            "verification_detail": v.get("detail"),
            "webgpu_submits": c.get("webgpu_submits"),
            "webgpu_dispatches": c.get("webgpu_dispatches"),
            "webgl_draws": c.get("webgl_draws"),
            "webgpu_dispatches_per_submit": v.get("dispatches_per_submit"),
            "webgpu_bind_groups_created": c.get("webgpu_bind_groups_created"),
            # Backend-instance hooks on the verification run (src/backend-hooks.js): kernels TF.js
            # ran on its JS CPU implementation, blocking GPU->CPU reads, async reads.
            "kernels_run": h.get("kernels_run"),
            "forwarded_kernels": h.get("forwarded_kernels"),
            "forwarded_by_kernel": json.dumps(h.get("forwarded_by_kernel")) if h.get("forwarded_by_kernel") else None,
            "sync_reads": h.get("sync_reads"),
            "sync_read_ms": h.get("sync_read_ms"),
            "sync_reads_by_kernel": json.dumps(h.get("sync_reads_by_kernel")) if h.get("sync_reads_by_kernel") else None,
            "async_reads": h.get("async_reads"),
            "math_random_calls": h.get("math_random_calls"),
            "tensors_leaked": v.get("tensors_leaked"),
            "kernels_by_phase": json.dumps(h.get("kernels_run_by_phase")) if h.get("kernels_run_by_phase") else None,
            "kernel_cpu_ms_by_phase": json.dumps(h.get("kernel_cpu_ms_by_phase")) if h.get("kernel_cpu_ms_by_phase") else None,
            "forwarded_by_phase": json.dumps(h.get("forwarded_by_phase")) if h.get("forwarded_by_phase") else None,
            "diskcache_prime_status": (e.get("diskcache_prime") or {}).get("status"),
            "diskcache_prime_first_run_ms": (e.get("diskcache_prime") or {}).get("first_run_ms"),
            "native_cudnn_rnn": json.dumps(e.get("native_cudnn_rnn")) if e.get("native_cudnn_rnn") is not None else None,
            "native_trace_ms": e.get("trace_ms"),
            "browser_version": e.get("browser_version"),
            # Page memory from measureUserAgentSpecificMemory() (profile mode).
            "page_memory_before_fetch": _mem(e, "page_memory", "before_fetch", "bytes"),
            "page_memory_after_session_create": _mem(e, "page_memory", "after_session_create", "bytes"),
            "page_memory_after_runs": _mem(e, "page_memory", "after_runs", "bytes"),
            "profile_phases": json.dumps(_mem(e, "profile", "phases")) if _mem(e, "profile", "phases") else None,
            "profile_time_skipped": _mem(e, "profile", "profile_time_skipped"),
            # Memory snapshots (src/memory.js): after model load and after dispose in every mode;
            # after first run / warm-up / runs in profile mode.
            **_snapshot_cols("mem_after_load", e.get("memory_after_load")),
            **_snapshot_cols("mem_after_first_run", _mem(e, "profile", "memory", "after_first_run")),
            **_snapshot_cols("mem_after_warmup", _mem(e, "profile", "memory", "after_warmup")),
            **_snapshot_cols("mem_after_runs", _mem(e, "profile", "memory", "after_runs")),
            **_snapshot_cols("mem_after_dispose", e.get("memory_after_dispose")),
        })
        for sm in e.get("shape_memory") or []:
            extra["shape_memory"].append({**key, "block": sm.get("block"), "shape_value": sm.get("shape_value"),
                                          "first_seen": sm.get("first_seen"), **_snapshot_cols("mem", sm)})
        for r in all_runs:
            runs.append({**key, "mode": e.get("mode", "latency"), **_flatten(r)})
        tc = e.get("train_correctness")
        if tc:
            for kind in ("grads", "weights_after"):
                for g in tc.get(kind, []):
                    extra["train_correctness"].append({**key, "kind": kind, "name": g["name"], "sumsq": g["sumsq"],
                                                       **{f"sketch_{j}": x for j, x in enumerate(g["sketch"])},
                                                       "step0_loss": tc.get("step0_loss"), "steps": tc.get("steps")})
        for lv in (e.get("long") or {}).get("val", []):
            extra["long_val"].append({**key, **lv})
        for k in _mem(e, "profile", "kernels") or []:
            kernels.append({**key, **k})

        # Responsiveness mode: times are page-relative ms (performance.now()); phase windows too.
        resp = e.get("responsiveness")
        if resp:
            phases = resp["phases"]

            def phase_of(t):
                for name, w in phases.items():
                    if w["start"] <= t <= w["end"]:
                        return name
                return "outside"
            fr = resp["frames"]
            for a, b in zip(fr, fr[1:]):
                extra["frames"].append({**key, "t_ms": b, "interval_ms": b - a, "phase": phase_of(b)})
            for ck in resp.get("clicks", []):
                extra["clicks"].append({**key, **ck, "phase": phase_of(ck["event_ts"]),
                                        "input_delay_ms": ck["handler_start"] - ck["event_ts"],
                                        "to_next_frame_ms": (ck["next_frame"] - ck["event_ts"]) if ck["next_frame"] is not None else None})
            for it in resp["interactions"]:
                extra["interactions"].append({**key, **it, "phase": phase_of(it["start"]),
                                              "input_delay_ms": it["processing_start"] - it["start"]})
            for lf in resp["loaf"]:
                # A long frame belongs to the phase containing its midpoint (one that starts just
                # before inference and spans it is caused by inference).
                extra["loaf"].append({**key, **lf, "phase": phase_of(lf["start"] + lf["duration"] / 2)})
            for name, w in phases.items():
                extra["phases"].append({**key, "phase": name, "start_ms": w["start"], "end_ms": w["end"]})

        load = e.get("load")
        if load:
            for r in load["requests"]:
                extra["load_requests"].append({**key, **r})
            for w in load["windows"]:
                fr = w.get("frames") or []
                for a, b in zip(fr, fr[1:]):
                    extra["load_frames"].append({**key, "rate": w["rate"], "t_ms": b, "interval_ms": b - a})

    pd.DataFrame(sessions).to_csv(out_dir / "sessions.csv", index=False)
    pd.DataFrame(runs).to_csv(out_dir / "runs.csv", index=False)
    if kernels:
        pd.DataFrame(kernels).to_csv(out_dir / "kernels.csv", index=False)
    for name, rows in extra.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    return len(sessions), len(runs)
