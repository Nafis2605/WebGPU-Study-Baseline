"""Group 10: compare each config's outputs with the TF.js cpu backend on identical input bytes.

The reference is TF.js's own pure-JavaScript `cpu` backend (float32), run in a separate browser
launch with the same seed. It is a matched-framework reference (same graph, same TF.js op
semantics), not native TensorFlow: tfjs-node has no Windows prebuilt for 4.22.0 (see
study_report/FEASIBILITY.md). Before comparing, the input bytes of the two launches are checked
to be identical (sha256); a mismatch is reported instead of a comparison.

Metrics per output (formulas unchanged from the ONNX study's benchkit/correctness.py; a = config,
b = reference, both flattened to float64, d = |a - b|):
    nonfinite     count of non-finite values in a (not masked out of the other metrics)
    max_abs_err, mean_abs_err, rmse = sqrt(mean(d^2)), max_rel_err = max(d / max(|b|, 1e-6))
    ref_abs_max = max|b|, norm_max_err = max(d) / max|b|
    cosine        a.b / (|a||b|)
    top1_agreement, top5_overlap   for outputs listed in models.json logits_outputs (last axis >= 2):
                                   argmax / top-5 agreement per row of the last axis
"""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

DTYPES = {"float32": np.float32, "int32": np.int32, "bool": np.uint8}


def decode(t):
    return np.frombuffer(base64.b64decode(t["b64"]), dtype=DTYPES[t["type"]]).reshape(t["dims"])


def sha(t):
    return hashlib.sha256(base64.b64decode(t["b64"])).hexdigest()


def compare(io, ref_io, logits_outputs=()):
    """Rows for one (config launch, reference launch) pair."""
    ref_in = {t["name"]: sha(t) for t in ref_io["inputs"]}
    got_in = {t["name"]: sha(t) for t in io["inputs"]}
    if ref_in != got_in:
        return [{"error": "input bytes differ from the reference launch", "inputs_identical": False}]
    ref_out = {t["name"]: t for t in ref_io["outputs"]}
    rows = []
    for t in io["outputs"]:
        row = {"output": t["name"], "dims": "x".join(map(str, t["dims"])), "dtype": t["type"], "inputs_identical": True}
        r = ref_out.get(t["name"])
        if r is None or list(r["dims"]) != list(t["dims"]):
            row["error"] = "reference output missing or shape differs"
            rows.append(row)
            continue
        a = decode(t).astype(np.float64).ravel()
        b = decode(r).astype(np.float64).ravel()
        d = np.abs(a - b)
        ref_max = float(np.abs(b).max()) if b.size else 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        row.update({
            "nonfinite": int((~np.isfinite(a)).sum()),
            "max_abs_err": float(d.max()) if d.size else 0.0,
            "mean_abs_err": float(d.mean()) if d.size else 0.0,
            "rmse": float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
            "max_rel_err": float(np.max(d / np.maximum(np.abs(b), 1e-6))) if d.size else 0.0,
            "ref_abs_max": ref_max,
            "norm_max_err": float(d.max() / ref_max) if ref_max > 0 else None,
            "cosine": float(a @ b / denom) if denom > 0 else None,
        })
        shape = list(t["dims"])
        if t["name"] in logits_outputs and shape and shape[-1] >= 2:
            g = decode(t).reshape(-1, shape[-1])
            e = decode(r).reshape(-1, shape[-1])
            row["top1_agreement"] = float(np.mean(g.argmax(1) == e.argmax(1)))
            if shape[-1] >= 5:
                tg, te = np.argsort(-g, 1)[:, :5], np.argsort(-e, 1)[:, :5]
                row["top5_overlap"] = float(np.mean([len(set(x) & set(y)) / 5 for x, y in zip(tg, te)]))
        rows.append(row)
    return rows


def compare_all(out_dir, registry, reference="cpu"):
    """Compare every correctness launch in out_dir with the reference config's launch of the same
    (trial, model). Returns CSV rows."""
    out_dir = Path(out_dir)
    exps = [json.loads(line) for line in (out_dir / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    exps = [e for e in exps if e.get("mode") == "correctness" and e.get("io_file")]
    refs = {(e["trial"], e["model"]): e for e in exps if e.get("config_label") == reference and e.get("status") == "ok"}
    rows = []
    for e in exps:
        if e.get("config_label") == reference:
            continue
        key = {"trial": e["trial"], "model": e["model"], "backend": e["config_label"], "status": e.get("status")}
        ref = refs.get((e["trial"], e["model"]))
        if ref is None:
            rows.append({**key, "error": f"no ok {reference} reference launch for this trial/model"})
            continue
        io = json.loads((out_dir / e["io_file"]).read_text(encoding="utf-8"))
        ref_io = json.loads((out_dir / ref["io_file"]).read_text(encoding="utf-8"))
        logits = tuple(registry.get(e["model"], {}).get("logits_outputs", ()))
        rows += [{**key, **r} for r in compare(io, ref_io, logits)]
    return rows
