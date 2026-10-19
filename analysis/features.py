"""
Model features, computed from measured data only (port of the sibling TF.js study's features.py; this
study adds training steps: profile = one inference, train_profile = one training step, split by the
forward / backward / optimizer phase tag each kernel carries in kernels.csv):

    macs          multiply-accumulates of one inference, from the input/output shapes that
                  tf.profile recorded for every kernel of a profiled run (kernels.csv):
                    Conv2D, FusedConv2D           N*Ho*Wo*Cout * kh*kw*Cin      (filter [kh,kw,Cin,Cout])
                    DepthwiseConv2dNative,
                    FusedDepthwiseConv2D          N*Ho*Wo*C*mult * kh*kw        (filter [kh,kw,C,mult])
                    Conv2DBackpropInput           numel(dy) * kh*kw*Cin         (inputs dy, filter)
                    Conv2DBackpropFilter          numel(dy) * kh*kw*Cin         (inputs x, dy; output filter)
                    BatchMatMul, _FusedMatMul,
                    MatMul                        numel(out) * K; K is the contracted dim, taken from
                                                  the last two dims of the inputs and validated
                                                  (must appear in both inputs); otherwise the kernel
                                                  is listed as skipped
                  Every other kernel (elementwise, pooling, normalisation, ...) contributes 0, as in the
                  ONNX study. Kernels whose shapes don't validate are counted in skipped_kernels and
                  never guessed.
    weight_bytes  model weight bytes (sum of the weight shards; sessions.csv model_bytes)
    kernels       kernels executed per inference / step (outermost engine.runKernel calls, backend hooks)
    macs_<phase>  training steps only: MACs of the kernels tagged forward / backward / optimizer (the
                  tag join must be exact, phase_join_ok; otherwise these are left empty)

Usage:
    python analysis/features.py <profile dir> [...] --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONV = {"Conv2D", "FusedConv2D"}
DEPTHWISE = {"DepthwiseConv2dNative", "FusedDepthwiseConv2D"}
MATMUL = {"BatchMatMul", "_FusedMatMul", "MatMul", "FusedMatMul"}


def prod(xs):
    return int(np.prod(xs)) if len(xs) else 1


def kernel_macs(name, ins, outs):
    """(macs, ok) for one kernel record. ok=False means the shapes couldn't be interpreted."""
    try:
        out = outs[0]
        if name in CONV:
            x, f = ins[0], ins[1]
            if len(out) != 4 or len(f) != 4 or f[3] != out[3]:
                return 0, False
            return prod(out) * f[0] * f[1] * f[2], True
        if name in DEPTHWISE:
            f = ins[1]
            if len(out) != 4 or len(f) != 4 or f[2] * f[3] != out[3]:
                return 0, False
            return prod(out) * f[0] * f[1], True
        if name == "Conv2DBackpropFilter":
            dy, f = ins[1], out
            if len(f) != 4 or len(dy) != 4 or dy[-1] != f[3]:
                return 0, False
            return prod(dy) * f[0] * f[1] * f[2], True
        if name == "Conv2DBackpropInput":
            dy, f = ins[0], ins[1]
            if len(f) != 4 or dy[-1] != f[3]:
                return 0, False
            return prod(dy) * f[0] * f[1] * f[2], True
        if name in MATMUL:
            a, b = ins[0], ins[1]
            if len(a) < 2 or len(b) < 2 or len(out) < 2:
                return 0, False
            m, n = out[-2], out[-1]
            a2, b2 = a[-2:], b[-2:]
            k = a2[1] if a2[0] == m else a2[0] if a2[1] == m else None
            if k is None or sorted(b2) != sorted([k, n]):
                return 0, False
            return prod(out) * k, True
        return 0, True
    except (IndexError, TypeError):
        return 0, False


def model_macs(kernels):
    """kernels: tf.profile rows of one run of one model. Returns (macs, counted, skipped names)."""
    total, counted, skipped = 0, 0, []
    for _, k in kernels.iterrows():
        name = str(k["name"])
        if name.endswith("__op"):
            continue
        macs, ok = kernel_macs(name, json.loads(k["input_shapes"]), json.loads(k["output_shapes"]))
        if not ok:
            skipped.append(name)
        elif macs:
            counted += 1
            total += macs
    return total, counted, skipped


def features(dirs):
    rows = []
    seen = set()
    for d in map(Path, dirs):
        s = pd.read_csv(d / "sessions.csv")
        s = s[(s["status"] == "ok") & s["mode"].isin(["profile", "train_profile"])]
        if s.empty or not (d / "kernels.csv").exists():
            continue
        k = pd.read_csv(d / "kernels.csv")
        k = k[k["source"] == "tf.profile"]
        runs = pd.read_csv(d / "runs.csv", low_memory=False)
        # Shapes are identical on every backend: use the first ok launch of each (model, mode), run 0.
        for _, sess in s.sort_values(["trial"]).iterrows():
            m, mode = sess["model"], sess["mode"]
            if (m, mode) in seen:
                continue
            kk = k[(k["trial"] == sess["trial"]) & (k["model"] == m) & (k["backend"] == sess["backend"]) & (k["run"] == 0)]
            if kk.empty:
                continue
            macs, counted, skipped = model_macs(kk)
            row = {"model": m, "unit": "training step" if mode == "train_profile" else "inference",
                   "family": sess.get("family"), "params": sess.get("params"), "batch":
                   sess.get("train_batch") if mode == "train_profile" else sess.get("infer_batch"),
                   "macs": macs, "gmacs": macs / 1e9, "mac_kernels": counted,
                   "skipped_kernels": len(skipped), "skipped_names": ",".join(sorted(set(skipped))),
                   "weight_bytes": sess["model_bytes"], "kernels": sess["kernels_run"],
                   "shape_source": f"{d.name}/{sess['backend']}/trial{sess['trial']}"}
            if mode == "train_profile":
                pr = runs[(runs["trial"] == sess["trial"]) & (runs["model"] == m) & (runs["backend"] == sess["backend"])
                          & (runs["phase"] == "profile_kernels") & (runs["i"] == 0)]
                join_ok = bool(len(pr)) and str(pr["phase_join_ok"].iloc[0]).lower() == "true"
                row["phase_join_ok"] = join_ok
                for ph in ("forward", "backward", "optimizer"):
                    row[f"macs_{ph}"] = model_macs(kk[kk["train_phase"] == ph])[0] if join_ok else None
                    row[f"kernels_{ph}"] = int((kk["train_phase"] == ph).sum()) if join_ok else None
            rows.append(row)
            seen.add((m, mode))
    t = pd.DataFrame(rows)
    if len(t):
        t["log_macs"] = np.log10(t["macs"].clip(lower=1))
        t["log_weight_bytes"] = np.log10(t["weight_bytes"].clip(lower=1))
        t["log_kernels"] = np.log10(t["kernels"].clip(lower=1))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = features(args.dirs)
    t.to_csv(out / "model_features.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(t.round(4).to_string(index=False))
    print(f"\nWritten to {out / 'model_features.csv'}")


if __name__ == "__main__":
    main()
