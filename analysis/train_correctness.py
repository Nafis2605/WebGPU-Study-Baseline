"""
Training correctness: every train_correctness launch compared with a reference launch of the same model
on identical initial weights and batches (the native worker and the page use the same files).

Per (model, config) vs the reference config:
  step0_loss_rel        |loss0 - ref| / |ref|
  grad_sketch_rel_max   max over variables and the 4 sketch vectors of |s - s_ref| / ||g_ref||_2, with
                        ||g_ref||_2 the norm over ALL variables (a +-1 projection of g has standard deviation
                        ||g||_2). Per-variable normalisation is meaningless for variables whose exact gradient is
                        zero (e.g. the attention key bias: softmax ignores a per-query constant), whose computed
                        gradient is rounding noise on both sides; these are listed in null_grad_vars
                        (||g_ref,v|| < 1e-6 ||g_ref||)
  grad_rel_l2           ||g - g_ref||_2 / ||g_ref||_2 over all variables (full dumps: models <= 3 M params)
  grad_cosine           cosine(g, g_ref) over all variables (full dumps)
  grad_max_abs_err      max |g - g_ref| (full dumps)
  loss_rel_max_k        max over the K steps of |loss_i - ref_i| / |ref_i|
  first_divergence_step first step with |loss_i - ref_i| / |ref_i| > 1e-2 (None if never within K)
  weights_rel_l2        ||w_K - w_K,ref||_2 / ||w_K,ref - w_0||_2 after K steps: error relative to how far
                        training moved the weights (full dumps)
  weights_sketch_rel_max   as grad_sketch_rel_max on the weights after K steps, relative to the global ||w_K,ref||_2
  nonfinite             any non-finite loss / gradient / weight

Usage:
  python analysis/train_correctness.py <dir> [<dir> ...] --reference tf-cpu+tfjsadam [--out DIR]
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIVERGE = 1e-2


def decode(t):
    return np.frombuffer(base64.b64decode(t["b64"]), dtype=np.float32).reshape(t["dims"]).astype(np.float64)


def load(dirs):
    launches = {}
    for d in map(Path, dirs):
        for line in (d / "experiments.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("mode") != "train_correctness":
                continue
            io = json.loads((d / e["io_file"]).read_text(encoding="utf-8")) if e.get("io_file") else None
            launches[(e["model"], e["config_label"], e.get("trial", 1))] = {"e": e, "io": io, "dir": str(d)}
    return launches


def init_weights(model):
    root = Path(__file__).resolve().parents[1]
    entry = json.loads((root / "public" / "weights" / "manifest.json").read_text(encoding="utf-8"))[model]["init"]
    raw = (root / "public" / "weights" / entry["file"]).read_bytes()
    return {f"{t['layer']}/{t['name']}": np.frombuffer(raw, "<f4", t["bytes"] // 4, t["offset"]).astype(np.float64)
            for t in entry["tensors"]}


def losses(e):
    return [r["loss"] for r in sorted((r for r in e["runs"] if r["phase"] == "correctness_step"), key=lambda r: r["i"])]


def compare(a, ref, model):
    ea, er = a["e"], ref["e"]
    ca, cr = ea["train_correctness"], er["train_correctness"]
    row = {"model": model, "config": ea["config_label"], "reference": er["config_label"],
           "status": ea["status"], "ref_status": er["status"]}
    row["step0_loss"] = ca["step0_loss"]
    row["step0_loss_ref"] = cr["step0_loss"]
    row["step0_loss_rel"] = abs(ca["step0_loss"] - cr["step0_loss"]) / abs(cr["step0_loss"])
    gr = {g["name"]: g for g in cr["grads"]}
    gnorm = np.sqrt(sum(g["sumsq"] for g in cr["grads"]))
    row["grad_sketch_rel_max"] = max(max(abs(x - y) for x, y in zip(g["sketch"], gr[g["name"]]["sketch"]))
                                     for g in ca["grads"]) / gnorm if gnorm > 0 else None
    row["null_grad_vars"] = ",".join(g["name"] for g in cr["grads"] if np.sqrt(g["sumsq"]) < 1e-6 * gnorm) or None
    wr = {g["name"]: g for g in cr["weights_after"]}
    wnorm = np.sqrt(sum(g["sumsq"] for g in cr["weights_after"]))
    row["weights_sketch_rel_max"] = max(max(abs(x - y) for x, y in zip(g["sketch"], wr[g["name"]]["sketch"]))
                                        for g in ca["weights_after"]) / wnorm if wnorm > 0 else None
    la, lr = losses(ea), losses(er)
    k = min(len(la), len(lr))
    rel = [abs(x - y) / abs(y) for x, y in zip(la[:k], lr[:k])]
    row["steps"] = k
    row["loss_rel_max_k"] = max(rel) if rel else None
    row["first_divergence_step"] = next((i for i, x in enumerate(rel) if x > DIVERGE), None)
    row["nonfinite"] = bool(any(not np.isfinite(x) for x in la))
    if a["io"] and ref["io"]:
        ga = np.concatenate([decode(t).ravel() for t in a["io"]["grads"]])
        gb = np.concatenate([decode(t).ravel() for t in ref["io"]["grads"]])
        row["grad_rel_l2"] = float(np.linalg.norm(ga - gb) / np.linalg.norm(gb))
        row["grad_cosine"] = float(ga @ gb / (np.linalg.norm(ga) * np.linalg.norm(gb)))
        row["grad_max_abs_err"] = float(np.max(np.abs(ga - gb)))
        row["nonfinite"] = row["nonfinite"] or not np.isfinite(ga).all()
        w0 = init_weights(model)
        wa = {t["name"]: decode(t).ravel() for t in a["io"]["weights_after"]}
        wb = {t["name"]: decode(t).ravel() for t in ref["io"]["weights_after"]}
        names = list(wb)
        da = np.concatenate([wa[n] - wb[n] for n in names])
        moved = np.concatenate([wb[n] - w0[n] for n in names])
        row["weights_rel_l2"] = float(np.linalg.norm(da) / np.linalg.norm(moved))
        row["nonfinite"] = row["nonfinite"] or not np.isfinite(da).all()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--reference", default="tf-cpu+tfjsadam")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    L = load(args.dirs)
    rows = []
    for (model, cfg, trial), a in sorted(L.items()):
        if cfg == args.reference:
            continue
        ref = L.get((model, args.reference, trial)) or next((v for (m, c, _), v in L.items()
                                                              if m == model and c == args.reference), None)
        if ref is None:
            rows.append({"model": model, "config": cfg, "reference": args.reference, "error": "no reference launch"})
            continue
        if a["e"]["status"] != "ok" or ref["e"]["status"] != "ok":
            rows.append({"model": model, "config": cfg, "reference": args.reference, "status": a["e"]["status"],
                         "error": "launch not ok"})
            continue
        rows.append(compare(a, ref, model))
    df = pd.DataFrame(rows)
    out = Path(args.out or args.dirs[0])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"train_correctness_vs_{args.reference.replace('+', '_')}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3g}".format):
        cols = [c for c in ["model", "config", "step0_loss_rel", "grad_sketch_rel_max", "grad_rel_l2", "grad_cosine",
                            "loss_rel_max_k", "first_divergence_step", "weights_rel_l2", "weights_sketch_rel_max",
                            "null_grad_vars", "nonfinite", "error"] if c in df]
        print(df[cols].to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
