"""
TFLite baseline on the phone (Android target): the counterpart of native/run_native.py's inference modes. On the Pixel 8a
the native CUDA stages are replaced by TFLite with the GPU delegate (OpenCL) and XNNPACK on the CPU.

Every launch is one fresh `benchmark_model` process on the phone (Google's prebuilt nightly, tools/android/tflite/,
sha256 in SOURCE.json), with the same launch discipline as the page: wait until the phone is cool, phone-side telemetry
with a baseline, the thermal-throttle guard and its re-runs, shuffled trials. The laptop only sends adb commands and
collects the output.

Rules (same as the rest of the study):
- No fallback. tflite-gpu runs with --require_full_delegation=true and the OpenCL backend pinned (--gpu_backend=cl): a model
  the GPU delegate cannot run whole, or a delegate that falls back to OpenGL, is recorded as a gap, never measured.
  XNNPACK configs run the ops XNNPACK does not take on TFLite's builtin CPU kernels (still the CPU); the node counts are
  recorded.
- benchmark_model reports summary statistics (count, min, max, avg, std, p5, median, p95 in us), not one record per run:
  the records here carry those values as reported (`per_run_records: false`).
- Training has no TFLite GPU path: recorded as a gap by the study stage table, not run.

Modes
  latency      b1 model; warm-up >= W runs and >= --warmup-min-s, then >= N runs and >= --min-measure-s
  correctness  b32 model on the first 32 validation images; output vs the Keras reference written by convert.py
               (public/tflite/ref), compared here on the laptop

    .venv/Scripts/python native/tflite/run_tflite_android.py --mode latency --filter D1 C1 --trials 1
"""
import argparse
import hashlib
import json
import random
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchkit import android, layout  # noqa: E402
from benchkit.results import ResultWriter  # noqa: E402

TFL = ROOT / "public" / "tflite"
TOOL = ROOT / "tools" / "android" / "tflite" / "android_aarch64_benchmark_model"
REMOTE = f"{android.TMP}/tflite"
SPEC = json.loads((ROOT / "public" / "spec" / "models.json").read_text(encoding="utf-8"))

CONFIGS = {
    "tflite-gpu": ["--use_gpu=true", "--gpu_backend=cl", "--gpu_precision_loss_allowed=false",
                   "--require_full_delegation=true"],
    "tflite-xnnpack-1t": ["--use_xnnpack=true", "--num_threads=1"],
    "tflite-xnnpack-4t": ["--use_xnnpack=true", "--num_threads=4"],
    "tflite-xnnpack-8t": ["--use_xnnpack=true", "--num_threads=8"],
}
MODES = ("latency", "correctness")
TOL_ABS = 1e-3   # fp32 on both sides; XNNPACK b32 vs Keras measured 1.1e-6 (D1)
STAT = re.compile(r"count=(\d+) first=(\d+) curr=(\d+) min=(\d+) max=(\d+) avg=([\d.e+-]+) std=(\d+)"
                  r"(?: p5=(\d+) median=(\d+) p95=(\d+))?")
TIMINGS = re.compile(r"Init: ([\d.]+), First inference: ([\d.]+), Warmup \(avg\): ([\d.e+-]+), Inference \(avg\): ([\d.e+-]+)")
REPLACE = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\) node, yielding (\d+) partitions")
MEMORY = re.compile(r"Memory footprint delta from the start of the tool \(MB\): init=([\d.]+) overall=([\d.]+)")
FAILURES = [("Disallowed CPU fallback detected", "gpu_partial_delegation"),
            ("Falling back to OpenGL", "gpu_backend_fallback"),
            ("Failed to apply GPU delegate", "gpu_delegate_failed"),
            ("Benchmarking failed", "benchmark_failed")]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def push_checked(local, remote):
    """Push once; the phone-side sha256 must equal the local one (also for files already there)."""
    want = sha256(local)
    have = android.sh(f"sha256sum {remote} 2>/dev/null", check=False).split(" ")[0].strip()
    if have != want:
        android.adb("push", str(local), remote, timeout=1800)
        have = android.sh(f"sha256sum {remote}").split(" ")[0].strip()
    if have != want:
        raise RuntimeError(f"push_mismatch: {remote}")
    return want


def parse(log):
    out = {"delegation": [m.groups() for m in REPLACE.finditer(log)]}
    out["delegation"] = [{"replaced": int(a), "nodes": int(b), "delegate": c, "partitions": int(d)} for a, b, c, d in out["delegation"]]
    unsupported = re.findall(r"^([A-Z_0-9]+): (.+)$", log.split("not supported by GPU delegate:", 1)[-1], re.M) \
        if "not supported by GPU delegate" in log else []
    out["gpu_unsupported_ops"] = sorted({f"{a}: {b.strip()}" for a, b in unsupported})
    stats = STAT.findall(log)
    keys = ("count", "first_us", "curr_us", "min_us", "max_us", "avg_us", "std_us", "p5_us", "median_us", "p95_us")
    conv = lambda t: {k: (float(v) if v != "" else None) for k, v in zip(keys, t)}  # noqa: E731
    if len(stats) >= 2:
        out["warmup"], out["measure"] = conv(stats[-2]), conv(stats[-1])
    t = TIMINGS.search(log)
    if t:
        out.update({"init_us": float(t.group(1)), "first_inference_us": float(t.group(2)),
                    "warmup_avg_us": float(t.group(3)), "inference_avg_us": float(t.group(4))})
    m = MEMORY.search(log)
    if m:
        out.update({"mem_delta_init_mb": float(m.group(1)), "mem_delta_overall_mb": float(m.group(2))})
    out["opencl"] = "Initialized OpenCL-based API" in log
    out["opengl"] = "Initialized OpenGL-based API" in log
    return out


def failure(log, config):
    for needle, cat in FAILURES:
        if needle in log:
            return {"stage": "tflite", "category": cat, "message": needle}
    if config == "tflite-gpu" and "Initialized OpenCL-based API" not in log:
        return {"stage": "tflite", "category": "gpu_backend_unverified", "message": "OpenCL API init line missing"}
    return None


def correctness(model, out_bin):
    ent = MANIFEST[f"{model}.b32"]
    ref = np.frombuffer((TFL / "ref" / ent["reference"]["output"]).read_bytes(), np.float32).reshape(ent["reference"]["output_shape"])
    got = np.frombuffer(Path(out_bin).read_bytes(), np.float32)
    if got.size != ref.size:
        return {"ok": False, "error": f"output size {got.size} != {ref.size}"}
    got = got.reshape(ref.shape)
    err = np.abs(got.astype(np.float64) - ref)
    # `ok`: the run produced finite outputs (the record is valid); `matches_reference` is the verdict on the values
    return {"ok": bool(np.isfinite(got).all()), "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "max_rel_err": float((err / np.maximum(np.abs(ref), 1e-6)).max()),
            "argmax_agree": float((got.argmax(-1) == ref.argmax(-1)).mean()), "rows": int(ref.shape[0]),
            "tolerance_abs": TOL_ABS, "matches_reference": bool(err.max() <= TOL_ABS)}


def row_loop(cmd, ent, ref, rows, tag):
    """Correctness at batch 1: one process per validation image (row file cut from the batch-32 input with dd), outputs
    concatenated in row order. Measured 2026-11-02: the GPU delegate gives wrong outputs for D1 / C1 at batch 32 (fp32,
    OpenCL and OpenGL alike) and correct ones at batch 1, so the GPU's correctness is checked here row by row."""
    row_bytes = 4 * int(np.prod(ent["input_shape"][1:]))
    name = ent["input_name"]
    lines = [f"cd {REMOTE}", f"rm -f out_{tag}.bin"]
    for i in range(rows):
        one = [*cmd, f"--input_layer={name}", "--input_layer_shape=" + ",".join(str(v) for v in ent["input_shape"]),
               f"--input_layer_value_files={name.replace(':', '::')}:{REMOTE}/row_{tag}.f32",
               f"--output_filepath={REMOTE}/o_{tag}.bin", "--warmup_runs=0", "--num_runs=1", "--min_secs=0",
               "--warmup_min_secs=0"]
        lines += [f"dd if={ref['input']} of=row_{tag}.f32 bs={row_bytes} skip={i} count=1 2>/dev/null",
                  " ".join(shlex.quote(c) for c in one) + " 2>&1 | grep -E 'ERROR|Replacing|Initialized Open|Falling'",
                  f"cat o_{tag}.bin >> out_{tag}.bin; rm -f o_{tag}.bin"]
    lines.append(f"rm -f row_{tag}.f32")
    script = f"{android.TMP}/rows_{tag}.sh"
    android.push_text("\n".join(lines) + "\n", script)
    return script


def run_one(args, model, config, trial, writer, telemetry, attempt=0):
    batch = args.correctness_batch if args.mode == "correctness" else args.infer_batch
    key = f"{model['name']}.b{batch}"
    ent = MANIFEST.get(key)
    tag = f"{model['name']}_{config}_{args.mode}_t{trial}" + (f"_r{attempt}" if attempt else "")
    rec = {"schema": 1, "model": model["name"], "family": model["family"], "size": model["size"], "params": model["params"],
           "backend": "tflite", "config_label": config, "mode": args.mode, "pillar": "infer", "trial": trial,
           "batch": batch, "started_at": datetime.now().isoformat(timespec="seconds"), "per_run_records": False,
           "low_clock_reruns": attempt}
    if not ent or ent.get("status") != "ok":
        rec.update({"status": "failed", "failure": {"stage": "conversion", "category": "conversion_gap",
                                                     "message": (ent or {}).get("error", f"{key} not converted")}})
        return rec
    rec["tflite_sha256"] = push_checked(TFL / ent["file"], f"{REMOTE}/{ent['file']}")
    rec["weights_sha256"] = ent["weights_sha256"]
    cmd = [f"{REMOTE}/benchmark_model", f"--graph={REMOTE}/{ent['file']}", *CONFIGS[config], "--verbose=true"]
    if args.mode == "latency":
        cmd += [f"--warmup_runs={args.warmup}", f"--warmup_min_secs={args.warmup_min_s}", f"--num_runs={args.measure}",
                f"--min_secs={args.min_measure_s}", f"--max_secs={args.max_measure_s}"]
    elif batch == 1:
        ref = MANIFEST[f"{model['name']}.b32"]["reference"]
        push_checked(TFL / "ref" / ref["input"], f"{REMOTE}/{ref['input']}")
        script = row_loop(cmd, ent, ref, ref["output_shape"][0], tag)
        cmd = ["sh", script]
    else:
        ref = ent["reference"]
        push_checked(TFL / "ref" / ref["input"], f"{REMOTE}/{ref['input']}")
        shape = ",".join(str(v) for v in ent["input_shape"])
        # value files use ':' as separator; this build takes '::' for a ':' in the name (its --help says '\:', which fails)
        name = ent["input_name"]
        cmd += [f"--input_layer={name}", f"--input_layer_shape={shape}",
                f"--input_layer_value_files={name.replace(':', '::')}:{REMOTE}/{ref['input']}",
                f"--output_filepath={REMOTE}/out_{tag}.bin",
                "--warmup_runs=1", "--num_runs=1", "--min_secs=0", "--warmup_min_secs=0"]
    cool = android.wait_cool()
    rec["android_cool"], rec["cooled_s"] = cool, cool["waited_s"]
    clock = android.AndroidLowClock(telemetry, cpu=config != "tflite-gpu") if telemetry else None
    if telemetry:
        telemetry.start()
        telemetry.mark("baseline_start")
        time.sleep(args.baseline_s)
        telemetry.mark("baseline_end")
        telemetry.mark("experiment_start")
    t0 = time.time()
    log = android.sh(" ".join(shlex.quote(c) for c in cmd) + " 2>&1", timeout=args.experiment_timeout_s, check=False)
    rec["wall_s"] = round(time.time() - t0, 2)
    if telemetry:
        telemetry.mark("experiment_end")
    (writer.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (writer.out_dir / "logs" / f"{tag}.log").write_text(" ".join(cmd) + "\n\n" + log, encoding="utf-8")
    rec["command"] = cmd
    rec["tflite"] = parse(log)
    fail = failure(log, config)
    if fail is None and args.mode == "latency" and "measure" not in rec["tflite"]:
        fail = {"stage": "tflite", "category": "no_timings", "message": "timing lines missing from the tool output"}
    if fail is None and args.mode == "correctness":
        local = writer.out_dir / "io" / f"{tag}.out.f32"
        local.parent.mkdir(parents=True, exist_ok=True)
        android.adb("pull", f"{REMOTE}/out_{tag}.bin", str(local))
        android.sh(f"rm -f {REMOTE}/out_{tag}.bin", check=False)
        rec["correctness"] = correctness(model["name"], local)
        if not rec["correctness"].get("ok"):
            fail = {"stage": "tflite", "category": "non_finite_or_bad_output", "message": str(rec["correctness"])[:300]}
    rec["status"] = "failed" if fail else "ok"
    if fail:
        rec["failure"] = fail
    if clock is not None:
        rec["gpu_sm_clock_max_mhz"] = clock.stop()
        rec["android_throttle"] = clock.throttle
        if rec["status"] == "ok" and clock.throttle["throttled"]:
            if attempt < args.low_clock_retries:
                return {"retry_low_clock": True, "throttle": clock.throttle}
            rec["status"] = "failed"
            rec["failure"] = {"stage": "guard", "category": "thermal_throttle", "message": str(clock.throttle)}
        data = telemetry.stop()
        rec["telemetry"] = {"samples": len(data["samples"]), "field_errors": data["field_errors"]}
        writer.append_rows("phone_samples.csv", [{"trial": trial, "model": model["name"], "config": config, **s}
                                                 for s in data["samples"]])
        writer.append_rows("phone_marks.csv", [{"trial": trial, "model": model["name"], "config": config, **m}
                                               for m in data["marks"]])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="latency", choices=MODES)
    ap.add_argument("--config", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--filter", nargs="+", default=None)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--warmup-min-s", type=float, default=1.0)
    ap.add_argument("--measure", type=int, default=200)
    ap.add_argument("--min-measure-s", type=float, default=2.0)
    ap.add_argument("--max-measure-s", type=float, default=150.0)
    ap.add_argument("--infer-batch", type=int, default=1)
    ap.add_argument("--correctness-batch", type=int, default=32, choices=(1, 32),
                    help="32: one batch-32 run; 1: the same 32 images one at a time (batch-1 model)")
    ap.add_argument("--baseline-s", type=float, default=3.0)
    ap.add_argument("--telemetry-interval-s", type=float, default=0.1)
    ap.add_argument("--no-telemetry", action="store_true")
    ap.add_argument("--low-clock-retries", type=int, default=2)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--experiment-timeout-s", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--output-dir", default=None, help="default: results/<device>/tflite/adhoc")
    ap.add_argument("--exact-output-dir", action="store_true")
    args = ap.parse_args()
    if not android.is_android():
        sys.exit("set TRAINBENCH_TARGET=android")
    global MANIFEST
    MANIFEST = json.loads((TFL / "manifest.json").read_text(encoding="utf-8"))
    models = [m for m in SPEC["models"] if not args.filter or any(m["name"] == f or m["name"].startswith(f) for f in args.filter)]
    base = Path(args.output_dir) if args.output_dir else layout.device_dir() / "tflite" / "adhoc"
    out_dir = base if args.exact_output_dir else base / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ResultWriter(out_dir)
    android.sh(f"mkdir -p {REMOTE}")
    tool_sha = push_checked(TOOL, f"{REMOTE}/benchmark_model")
    android.sh(f"chmod 755 {REMOTE}/benchmark_model")
    source = json.loads((TOOL.parent / "SOURCE.json").read_text(encoding="utf-8-sig"))
    if source["sha256"] != tool_sha:
        sys.exit("benchmark_model sha256 differs from tools/android/tflite/SOURCE.json")
    writer.write_json("env.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "target": "android",
        "tool": {**source, "remote": f"{REMOTE}/benchmark_model"}, "configs": CONFIGS,
        "tflite_manifest_meta": MANIFEST.get("_meta"), "android_device": android.device_info(),
        "android_settings": android.settings_snapshot(),
        "android_guards": {"cool_skin_c": android.COOL_SKIN_C, "cool_gpu_c": android.COOL_GPU_C,
                           "cool_timeout_s": android.COOL_TIMEOUT_S,
                           "launch_guard": "re-run when thermal status > 0 or the cooling device of the used unit > 0"},
        "laptop_role": "orchestration only: adb commands and result collection; benchmark_model runs on the phone"})
    android.apply_study_settings()
    telemetry = None if args.no_telemetry else android.AndroidRunTelemetry(max(0.1, args.telemetry_interval_s), out_dir)
    plan = [(t, m, c) for t in range(1, args.trials + 1) for m in models for c in args.config]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(plan)
    print(f"Output: {out_dir}\n{len(plan)} launches", flush=True)
    for i, (trial, model, config) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] trial {trial} {model['name']} / {config} / {args.mode} ...", flush=True)
        attempt = 0
        while True:
            r = run_one(args, model, config, trial, writer, telemetry, attempt)
            if not r.get("retry_low_clock"):
                break
            attempt += 1
            telemetry.stop()
            print(f"    thermal throttle {r['throttle']}: re-run {attempt}", flush=True)
            time.sleep(args.cooldown_s)
        writer.append(r)
        t = r.get("tflite", {})
        if r["status"] == "ok" and args.mode == "latency":
            print(f"    ok, median {t['measure']['median_us'] / 1000:.3f} ms over {int(t['measure']['count'])} runs, "
                  f"init {t['init_us'] / 1000:.0f} ms, first {t['first_inference_us'] / 1000:.2f} ms, delegation {t['delegation']}",
                  flush=True)
        elif r["status"] == "ok":
            print(f"    ok, {r['correctness']}", flush=True)
        else:
            print(f"    failed: {r['failure']['category']} {r['failure']['message'][:120]}", flush=True)
        time.sleep(args.cooldown_s)
    print(f"Done: {len(plan)} launches -> {out_dir}")


MANIFEST = {}

if __name__ == "__main__":
    main()
