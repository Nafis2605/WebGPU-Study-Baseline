"""
Run the GPU-trace matrix with Nsight Systems (port of the sibling study's capture_matrix.py): browser
captures (nsight/run_nsys.py) and native TensorFlow CUDA captures (nsight/run_nsys_native.py), for the
representative subset, inference and training. Each entry is repeated; finished captures are skipped on
re-run (resume), keyed by output-folder prefix and tag.

Browser (<out>/nsys), D3D12 workloads (WebGPU) / GPU metrics (WebGL: D3D11 API tracing slows ANGLE):
  B1 training steps    webgpu, webgl on the subset: GPU busy fraction per step, idle gaps, workloads/step
  B2 inference         webgpu, webgl on the subset at batch 1
Native (<out>/nsys_native), CUDA kernels / copies:
  N1 training steps    tf-cuda-fp32 (graph) and tf-cuda-eager on the subset
  N2 inference         tf-cuda-fp32 on the subset

Usage:
    python nsight/capture_matrix.py --out <dir> [--only B1 N1] [--repeats 1] [--dry-run]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ["D3", "C3", "R3", "L3", "G3", "T3"]
D12 = ["--trace", "dx12", "--gpu-metrics"]
MET = ["--trace", "none", "--gpu-metrics"]
TRAIN_RUNS = ["--start-at-run", "15", "--measure-runs", "100", "--window-s", "60"]

# (group, model, config, task, extra args, repeats)
MATRIX = [
    *[("B1", m, "webgpu", "train", [*D12, *TRAIN_RUNS], 2) for m in SUBSET],
    *[("B1", m, "webgl", "train", [*MET, *TRAIN_RUNS], 2) for m in SUBSET],
    *[("B2", m, "webgpu", "infer", D12, 2) for m in SUBSET],
    *[("B2", m, "webgl", "infer", MET, 2) for m in SUBSET],
    *[("N1", m, c, "train", [], 2) for m in SUBSET for c in ("tf-cuda-fp32", "tf-cuda-eager")],
    *[("N2", m, "tf-cuda-fp32", "infer", [], 2) for m in SUBSET],
]


def folder_prefix(model, config, task, extra):
    if config.startswith("tf-"):
        warm = 15 if task == "train" else 10
        return f"{model}__native-{config}__{task}__closed__run{warm}__"
    start = extra[extra.index("--start-at-run") + 1] if "--start-at-run" in extra else "10"
    return f"{model}__{config}__{task}__closed__run{start}__"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=None, help="override every entry's repeat count (preflight: 1)")
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these models (preflight)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    todo = [(g, m, c, t, e, f"{g}r{i + 1}") for g, m, c, t, e, n in MATRIX
            if (not args.only or g in args.only) and (not args.models or m in args.models)
            for i in range(args.repeats or n)]
    print(f"{len(todo)} captures", flush=True)
    failed = 0
    for k, (g, m, c, task, e, tag) in enumerate(todo, 1):
        native = c.startswith("tf-")
        sub = out / ("nsys_native" if native else "nsys")
        prefix = folder_prefix(m, c, task, e)
        if sub.exists() and any(d.name.endswith(f"_{tag}") and (d / "report.sqlite").exists()
                                for d in sub.glob(f"{prefix}*")):
            print(f"[{k}/{len(todo)}] {tag} {m} {c} {task}: done, skipped")
            continue
        if native:
            warm = "15" if task == "train" else "10"
            meas = "100" if task == "train" else "200"
            cmd = [sys.executable, str(ROOT / "nsight" / "run_nsys_native.py"), "--model", m, "--config", c,
                   "--task", task, "--warmup", warm, "--measure", meas, "--tag", tag, "--out", str(sub)]
        else:
            cmd = [sys.executable, str(ROOT / "nsight" / "run_nsys.py"), "--model", m, "--config", c, "--task", task,
                   *e, "--tag", tag, "--out", str(sub)]
        print(f"[{k}/{len(todo)}] {tag} {m} {c} {task} {' '.join(e)}", flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
        last = [l for l in (r.stdout + r.stderr).splitlines()
                if "runs=" in l or "LOCKED" in l or "NO report" in l or "status=" in l]
        caps = sorted(sub.glob(f"{prefix}*_{tag}"))
        ok = r.returncode == 0 and bool(caps) and (caps[-1] / "report.sqlite").exists()
        if ok:
            try:
                st = json.loads((caps[-1] / "page_stats.json").read_text(encoding="utf-8"))
                ok = len((st.get("stats") or {}).get("runs") or []) > 0 and not st.get("error")
            except Exception:  # noqa: BLE001
                ok = False
        failed += 0 if ok else 1
        print("    " + ("ok " if ok else "FAILED ") + (" | ".join(x[:160] for x in last) or f"rc={r.returncode}"), flush=True)
    print(f"Done: {len(todo)} captures, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
