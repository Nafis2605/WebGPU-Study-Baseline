"""
Nsight Systems capture of native TensorFlow (CUDA) inference or training: the native counterpart of
nsight/run_nsys.py (port of the sibling study's file). Runs native/worker.py (tfgpu env) under
`nsys profile --trace cuda,cudnn,cublas` with GPU metrics, using the study's protocol (warm-up, then
measured runs / steps) on the study's weight and data files. The worker's per-run timestamps are written in the same page_stats.json
format as the browser captures, so analysis/nsys_summary.py segments both the same way
(GPU work = CUDA kernels + memcpys instead of D3D12 workloads).

Usage:
    python nsight/run_nsys_native.py --model mobilenet_v2_100_224 --config tf-cuda --io-dir <correctness>/io
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit import envlock  # noqa: E402
from benchkit.clockguard import idle_check  # noqa: E402

from benchkit.paths import NSYS, TFGPU, tfgpu_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="tf-cuda-fp32", choices=("tf-cuda", "tf-cuda-fp32", "tf-cuda-eager", "tf-cuda-resident"))
    ap.add_argument("--task", default="infer", choices=("infer", "train"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "nsight" / "out" / "nsys_native"))
    args = ap.parse_args()
    lock = envlock.verify(scope="launch")
    t_wait = time.time()
    ok, msg = idle_check()
    while not ok and time.time() - t_wait < 30:
        time.sleep(2)
        ok, msg = idle_check()
    if not ok:
        sys.exit(f"GPU clocks look locked before the capture: {msg}")
    name = f"{args.model}__native-{args.config}__{args.task}__closed__run{args.warmup}"
    out = Path(args.out) / f"{name}__{time.strftime('%Y%m%d_%H%M%S')}{('_' + args.tag) if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    if args.task == "infer":
        job = {"model": args.model, "config": args.config, "mode": "latency", "warmup": args.warmup,
               "measure": args.measure, "infer_batch": 1}
    else:
        job = {"model": args.model, "config": args.config, "mode": "train", "train_warmup": args.warmup,
               "train_measure": args.measure, "infer_after_train": False}
    job.update({"threads": 8, "min_measure_s": 0, "result_file": str(out / "worker_result.json")})
    fd, jf = tempfile.mkstemp(prefix="trainbench-nsysnative-", suffix=".json")
    os.close(fd)
    Path(jf).write_text(json.dumps(job), encoding="utf-8")
    cmd = [str(NSYS), "profile", "--trace", "cuda,cuDNN,cublas", "--sample", "none", "--cpuctxsw", "none",
           "--gpu-metrics-devices", "all", "--gpu-metrics-frequency", "10000",
           "--output", str(out / "report"), "--export", "sqlite", "--force-overwrite", "true",
           str(TFGPU / "python.exe"), str(ROOT / "native" / "worker.py"), jf]
    env = tfgpu_env()
    (out / "command.json").write_text(json.dumps({"profile": cmd, "job": job}, indent=1), encoding="utf-8")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, errors="replace", timeout=1800)
    Path(jf).unlink(missing_ok=True)
    (out / "nsys.log").write_text(p.stdout[-20000:] + "\n" + p.stderr[-20000:], encoding="utf-8")
    rf = out / "worker_result.json"
    res = json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else         {"status": "failed", "runs": [], "failure": {"message": p.stderr[-2000:]}}
    timed = ("warmup", "measure", "train_warmup", "train_measure")
    runs = [{"start_epoch_ms": r["t_start_epoch_ms"], "end_epoch_ms": r["t_start_epoch_ms"] + r["dur_ms"]}
            for r in res.get("runs", []) if r.get("phase") in timed]
    (out / "page_stats.json").write_text(json.dumps({
        "stats": {"mode": "closed", "runs": runs}, "error": None if res.get("status") == "ok" else res.get("failure"),
        "window": {"skip_runs": args.warmup}, "launch_epoch_s": t0, "duration_s": None,
        "env_lock_sha256": lock["sha256"], "runtime": "native-tf"}, indent=0), encoding="utf-8")
    ok_after, msg_after = idle_check()
    (out / "clock_check.json").write_text(json.dumps({"after": msg_after, "after_ok": ok_after}, indent=1), encoding="utf-8")
    print(f"{name}: status={res.get('status')} runs={len(runs)} -> {out} ({time.time() - t0:.0f}s); clock after: {msg_after}")


if __name__ == "__main__":
    main()
