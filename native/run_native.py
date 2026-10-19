"""
Native TensorFlow driver (the counterpart of run_benchmark.py; port of the sibling TF.js study's
native/run_native.py). Runs in the main .venv; every (trial, model, config) launch is a fresh process of
native/worker.py in the conda env `tfgpu` (TF 2.10.1 + CUDA 11.2 + cuDNN 8.1), so cold-start costs are
measured like the browser's cold launches.

Shares the browser harness's machinery: ResultWriter (experiments.jsonl -> CSVs), NVML telemetry, GPU
cooling wait, shuffled order within each trial, measurement caps, the low-clock re-run guard, and the
environment lock (verified per stage and per launch).

Inference correctness (--mode correctness) compares the native output with the browser correctness
stage's TF.js cpu reference launch for the same model (--io-dir), after checking the input bytes are
identical. Training correctness is compared offline (analysis/train_correctness.py).

Usage:
    python native/run_native.py --mode latency --config tf-cuda tf-cuda-fp32 tf-cpu --trials 3
    python native/run_native.py --mode train --config tf-cuda-fp32 tf-cpu --filter L1
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit.correctness import compare  # noqa: E402
from benchkit.envinfo import host_info  # noqa: E402
from benchkit import clockguard  # noqa: E402
from benchkit.clockguard import LOW_CLOCK_MHZ, ClockSampler  # noqa: E402
from benchkit.results import ResultWriter  # noqa: E402
from benchkit.paths import SPEC, TFGPU, tfgpu_env  # noqa: E402

CONFIGS = ("tf-cuda", "tf-cuda-fp32", "tf-cuda-eager", "tf-cuda-resident", "tf-cpu")
MODES = ("latency", "batch", "correctness", "train", "train_batch", "train_long", "train_fit", "train_phases",
         "train_correctness", "lifecycle")


def wait_for_gpu_temp(limit_c, timeout_s=None):
    timeout_s = clockguard.COOL_TIMEOUT_S if timeout_s is None else timeout_s
    import pynvml as nv
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    t0 = time.time()
    while nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU) > limit_c and time.time() - t0 < timeout_s:
        time.sleep(2)
    nv.nvmlShutdown()
    return time.time() - t0


def run_worker(job, timeout_s, cpu_only):
    tmp = Path(tempfile.mkdtemp(prefix="trainbench-native-"))
    jf = tmp / "job.json"
    jf.write_text(json.dumps(job), encoding="utf-8")
    try:
        p = subprocess.run([str(TFGPU / "python.exe"), str(ROOT / "native" / "worker.py"), str(jf)],
                           capture_output=True, text=True, env=tfgpu_env(cpu_only=cpu_only), timeout=timeout_s,
                           errors="replace", cwd=ROOT)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            return json.loads(line[7:]), p.stderr
        return None, (p.stderr or "")[-4000:]
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        jf.unlink(missing_ok=True)
        try:
            tmp.rmdir()
        except OSError:
            pass


def build_job(args, model, config):
    return {
        "model": model["name"], "config": config, "mode": args.mode, "weights": args.weights,
        "verify_weights": args.verify_weights, "threads": 8, "determinism": args.determinism,
        "warmup": args.warmup, "measure": args.measure, "max_measure_s": args.max_measure_s,
        "warmup_min_s": args.warmup_min_s,
        "min_measure": args.min_measure, "min_measure_s": args.min_measure_s, "infer_batch": args.infer_batch, "batch_values": args.batch_values,
        "correctness_batch": args.correctness_batch,
        "train_warmup": args.train_warmup, "train_measure": args.train_measure, "train_max_s": args.train_max_s,
        "train_min_measure": args.train_min_measure, "train_batch": args.train_batch,
        "train_batch_values": args.train_batch_values, "batch_warmup": args.batch_warmup,
        "batch_max_s": args.batch_max_s, "batch_min_measure": args.batch_min_measure,
        "infer_warmup": args.infer_warmup, "infer_measure": args.infer_measure, "infer_max_s": args.infer_max_s,
        "infer_min_measure": args.infer_min_measure, "infer_after_train": not args.no_infer_after_train,
        "long_epochs": args.long_epochs, "fit_epochs": args.fit_epochs, "phase_max_s": args.phase_max_s,
        "correctness_steps": args.correctness_steps, "optimizer_impl": args.optimizer_impl,
        "lifecycle_order": args.lifecycle_order,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="latency", choices=MODES)
    ap.add_argument("--config", nargs="+", default=["tf-cuda-fp32", "tf-cpu"], choices=CONFIGS)
    ap.add_argument("--filter", nargs="+", default=None)
    ap.add_argument("--weights", default=None, choices=("init", "trained"))
    ap.add_argument("--verify-weights", action="store_true")
    ap.add_argument("--determinism", action="store_true", help="tf.config.experimental.enable_op_determinism()")
    ap.add_argument("--io-dir", default=None, help="browser correctness stage io/ (TF.js cpu reference, --mode correctness)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--warmup-min-s", type=float, default=1.0,
                    help="timed warm-ups run >= the warm-up count and >= this many seconds after the first record")
    ap.add_argument("--measure", type=int, default=200)
    ap.add_argument("--max-measure-s", type=float, default=None)
    ap.add_argument("--min-measure", type=int, default=30)
    ap.add_argument("--min-measure-s", type=float, default=2.0)
    ap.add_argument("--infer-batch", type=int, default=1)
    ap.add_argument("--batch-values", type=int, nargs="+", default=None)
    ap.add_argument("--correctness-batch", type=int, default=32)
    ap.add_argument("--train-warmup", type=int, default=15)
    ap.add_argument("--train-measure", type=int, default=100)
    ap.add_argument("--train-max-s", type=float, default=30.0)
    ap.add_argument("--train-min-measure", type=int, default=20)
    ap.add_argument("--train-batch", type=int, default=None)
    ap.add_argument("--train-batch-values", type=int, nargs="+", default=None)
    ap.add_argument("--batch-warmup", type=int, default=5)
    ap.add_argument("--batch-max-s", type=float, default=10.0)
    ap.add_argument("--batch-min-measure", type=int, default=10)
    ap.add_argument("--infer-warmup", type=int, default=5)
    ap.add_argument("--infer-measure", type=int, default=50)
    ap.add_argument("--infer-max-s", type=float, default=10.0)
    ap.add_argument("--infer-min-measure", type=int, default=20)
    ap.add_argument("--no-infer-after-train", action="store_true")
    ap.add_argument("--long-epochs", type=int, default=3)
    ap.add_argument("--fit-epochs", type=int, default=2)
    ap.add_argument("--phase-max-s", type=float, default=10.0)
    ap.add_argument("--correctness-steps", type=int, default=20)
    ap.add_argument("--optimizer-impl", default="keras", choices=("keras", "tfjs"))
    ap.add_argument("--lifecycle-order", default="infer_first", choices=("infer_first", "train_first"))
    ap.add_argument("--cool-to-c", type=float, default=None)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--baseline-s", type=float, default=3.0)
    ap.add_argument("--no-telemetry", action="store_true")
    ap.add_argument("--experiment-timeout-s", type=int, default=3600)
    ap.add_argument("--low-clock-retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--no-prime", action="store_true", help="skip the unrecorded priming launch")
    ap.add_argument("--output-dir", default=None, help="default: results/<device>/native/adhoc")
    ap.add_argument("--exact-output-dir", action="store_true")
    args = ap.parse_args(argv)
    if args.output_dir is None:
        from benchkit import layout
        args.output_dir = str(layout.native_dir("adhoc"))

    from benchkit import envlock
    lock = envlock.verify(scope="native")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    models = [m for m in spec["models"]
              if not args.filter or any(m["name"] == f or m["name"].startswith(f) for f in args.filter)]
    if args.mode == "correctness" and not args.io_dir:
        sys.exit("--mode correctness needs --io-dir (the browser correctness stage's io/ folder)")
    out_dir = Path(args.output_dir) if args.exact_output_dir else Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ResultWriter(out_dir)
    writer.write_json("env.json", {"created_at": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                                   "host": host_info(), "env_lock_sha256": lock["sha256"], "runtime": "native-tf",
                                   "page_env": None})
    plan = [(t, m, c) for t in range(1, args.trials + 1) for m in models for c in args.config]
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        plan = [x for t in range(1, args.trials + 1)
                for x in rng.sample([p for p in plan if p[0] == t], k=sum(p[0] == t for p in plan))]
    telemetry = None
    if not args.no_telemetry:
        from benchkit.telemetry import NvmlTelemetry
        telemetry = NvmlTelemetry(interval_s=0.05, pcie=True)
    print(f"Output: {out_dir}\n{len(plan)} experiments", flush=True)
    if plan and not args.no_prime:
        # One unrecorded launch first: the first TensorFlow process after a while reads the CUDA / cuDNN
        # DLLs from disk (a cold OS file cache made one first step ~4x slower than any later launch), so
        # the first recorded launch would carry a cost the others don't.
        t0 = time.time()
        _, _ = run_worker({**build_job(args, plan[0][1], plan[0][2]), "mode": "latency", "weights": "init", "warmup": 2, "measure": 2},
                          args.experiment_timeout_s, plan[0][2] == "tf-cpu")
        print(f"priming launch done ({time.time() - t0:.0f}s, not recorded)", flush=True)
    corr_rows = []
    for n, (trial, model, config) in enumerate(plan, 1):
        print(f"[{n}/{len(plan)}] trial {trial} {model['name']} / {config} / {args.mode} ...", flush=True)
        attempt = 0
        while True:
            t0 = time.time()
            envlock.verify(scope="launch")
            clock = ClockSampler().start() if config.startswith("tf-cuda") else None
            cooled = wait_for_gpu_temp(args.cool_to_c) if args.cool_to_c is not None else None
            if telemetry:
                telemetry.start()
                telemetry.mark("baseline_start")
                time.sleep(args.baseline_s)
                telemetry.mark("baseline_end")
                telemetry.mark("experiment_start")
            r, err = run_worker(build_job(args, model, config), args.experiment_timeout_s, config == "tf-cpu")
            if r is None:
                r = {"schema": 3, "model": model["name"], "family": model["family"], "size": model["size"],
                     "params": model["params"], "backend": "native", "config_label": config, "mode": args.mode,
                     "status": "failed", "runs": [],
                     "failure": {"stage": "process", "category": "native_process_failure", "message": err[-2000:]}}
            data = None
            if telemetry:
                telemetry.mark("experiment_end")
                data = telemetry.stop()
            low = False
            sm_max = clock.stop() if clock else None
            r["gpu_sm_clock_max_mhz"] = sm_max
            if sm_max is not None and r["status"] == "ok":
                low = sm_max <= LOW_CLOCK_MHZ
            if low and attempt < args.low_clock_retries:
                attempt += 1
                print(f"    low-clock launch (SM never above {LOW_CLOCK_MHZ} MHz): re-run {attempt}", flush=True)
                time.sleep(args.cooldown_s)
                continue
            break
        if args.mode == "train_correctness" and args.optimizer_impl == "tfjs":
            r["config_label"] = f"{config}+tfjsadam"   # TF.js's Adam formula (worker.TfjsAdam), not Keras's
        r.update({"trial": trial, "started_at": datetime.now().isoformat(timespec="seconds"), "cooled_s": cooled,
                  "low_clock_reruns": attempt, "low_clock_final": bool(low), "env_lock_sha256": lock["sha256"]})
        if low:
            r["status"] = "failed"
            r["failure"] = {"stage": "guard", "category": "gpu_low_clock", "message": "SM clock never exceeded 1000 MHz"}
        tag = f"t{trial}_{model['name']}_{r['config_label']}"
        if r.get("io"):
            (out_dir / "io").mkdir(exist_ok=True)
            (out_dir / "io" / f"{tag}.json").write_text(json.dumps(r["io"]), encoding="utf-8")
            if args.mode == "correctness":
                ref = Path(args.io_dir) / f"t1_{model['name']}_cpu.json"
                if ref.exists():
                    rows = compare(r["io"], json.loads(ref.read_text(encoding="utf-8")), ("probs",))
                else:
                    rows = [{"error": f"reference io file missing: {ref}"}]
                corr_rows += [{"trial": trial, "model": model["name"], "backend": config, "status": r["status"], **x}
                              for x in rows]
            r.pop("io")
            r["io_file"] = f"io/{tag}.json"
        if data:
            key = {"trial": trial, "model": model["name"], "backend": config}
            writer.append_rows("gpu_samples.csv", [{**key, **s} for s in data["samples"]])
            writer.append_rows("gpu_util_samples.csv", [{**key, **s} for s in data["util"]])
            writer.append_rows("gpu_pcie_samples.csv", [{**key, **s} for s in data["pcie"]])
            writer.append_rows("gpu_marks.csv", [{**key, **s} for s in data["marks"]])
            writer.append_rows("system_cpu_samples.csv", [{**key, **s} for s in data["system_cpu"]])
        writer.append(r)
        timed = ("measure", "batch", "train_measure", "train_batch", "train_long", "phase_step", "lc_train", "lc_train1")
        meas = sorted(x["dur_ms"] for x in r.get("runs", []) if x["phase"] in timed)
        med = meas[len(meas) // 2] if meas else None
        msg = (r.get("failure") or {}).get("message", "")
        print(f"    {r['status']}" + (f", median {med:.3f} ms over {len(meas)} runs" if med is not None else "")
              + (" [capped]" if r.get("capped") else "") + f" ({time.time() - t0:.0f}s) {msg[:200]}", flush=True)
        if n < len(plan):
            time.sleep(args.cooldown_s)
    if corr_rows:
        writer.append_rows("correctness.csv", corr_rows)
    n_s, n_r = writer.export_csvs()
    print(f"Done: {n_s} experiments, {n_r} run records -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
