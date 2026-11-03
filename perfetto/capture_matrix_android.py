"""
GPU-trace matrix on the phone (Android target): the counterpart of nsight/capture_matrix.py, with Perfetto
(perfetto/run_perfetto.py) instead of Nsight Systems. Same browser groups, models and repeats as the laptop matrix:
  B1 training steps    webgpu, webgl, every model (start after 15 runs, 100 measured, 60 s cap)
  B2 inference         webgpu, webgl, every model at batch 1 (start after 10 runs, 200 measured)
The laptop's native groups (N1 / N2, CUDA) have no counterpart: TFLite is not traced here (it is measured by the tflite
stages). A capture is done when its folder has trace.pftrace and a trace_check.json with GPU counter samples; finished
captures are skipped on re-run (resume).

    python perfetto/capture_matrix_android.py --out <dir> [--only B1] [--models D3 C3] [--repeats 1] [--dry-run]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [f + str(i) for f in "DCRLGT" for i in range(1, 6)]
TRAIN_RUNS = ["--start-at-run", "15", "--measure-runs", "100", "--window-s", "60"]
INFER_RUNS = ["--start-at-run", "10", "--measure-runs", "200", "--window-s", "60"]

MATRIX = [
    *[("B1", m, c, "train", TRAIN_RUNS, 2) for c in ("webgpu", "webgl") for m in MODELS],
    *[("B2", m, c, "infer", INFER_RUNS, 2) for c in ("webgpu", "webgl") for m in MODELS],
]


def done(sub, model, config, task, extra, tag):
    start = extra[extra.index("--start-at-run") + 1]
    for d in sub.glob(f"{model}__{config}__{task}__closed__run{start}__*_{tag}"):
        try:
            chk = json.loads((d / "trace_check.json").read_text(encoding="utf-8"))
            if (d / "trace.pftrace").exists() and int(chk.get("gpu_counter_samples") or 0) > 0:
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--filter", nargs="+", default=None, help="same as --models (run_study's verify flight passes --filter)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    models = args.models or args.filter
    sub = Path(args.out) / "perfetto"
    todo = [(g, m, c, t, e, f"{g}r{i + 1}") for g, m, c, t, e, n in MATRIX
            if (not args.only or g in args.only) and (not models or m in models) for i in range(args.repeats or n)]
    print(f"{len(todo)} captures", flush=True)
    failed = 0
    for k, (g, m, c, task, e, tag) in enumerate(todo, 1):
        if done(sub, m, c, task, e, tag):
            print(f"[{k}/{len(todo)}] {tag} {m} {c} {task}: done, skipped")
            continue
        cmd = [sys.executable, str(ROOT / "perfetto" / "run_perfetto.py"), "--model", m, "--config", c, "--task", task,
               *e, "--tag", tag, "--out", str(sub)]
        print(f"[{k}/{len(todo)}] {tag} {m} {c} {task}", flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
        last = [ln for ln in (r.stdout + r.stderr).splitlines() if "runs=" in ln or "Error" in ln or "exit" in ln.lower()]
        ok = r.returncode == 0 and done(sub, m, c, task, e, tag)
        failed += 0 if ok else 1
        print("    " + ("ok " if ok else "FAILED ") + (" | ".join(x[:300] for x in last[-3:]) or f"rc={r.returncode}"),
              flush=True)
    print(f"Done: {len(todo)} captures, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
