"""
Initial weights for every spec model: public/weights/<model>.init.bin (+ public/weights/manifest.json).

Both runtimes start training from these exact bytes (TF.js: src/weights.js; Keras: native/weights.py),
so neither framework's RNG or initialiser implementation enters the comparison. Each weight is drawn
with numpy from its own generator, seeded by sha256("<STUDY_SEED>|<model>|<layer>|<weight>"), so a
weight's values don't depend on any other weight or on generation order.

Initialisers (Keras 2.10 definitions, fan computation as keras.initializers._compute_fans):
  glorot_uniform     U(-l, l), l = sqrt(6 / (fan_in + fan_out))
  glorot_normal      truncated normal (redrawn beyond 2 sd), sd = sqrt(2 / (fan_in + fan_out)) / 0.87962566103423978
  orthogonal         QR of a standard-normal (rows, cols) matrix, sign-corrected; gain 1
  lstm_unit_forget   zeros with the forget-gate block (units..2*units) set to 1 (Keras unit_forget_bias)
  uniform_0.05       U(-0.05, 0.05) (Keras Embedding default; positional table)
  zeros / ones

File layout: float32 little-endian, weights concatenated in spec order (manifest lists offset,
shape and sha256 of each). Usage:  .venv/Scripts/python scripts/make_init_weights.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchkit.paths import SPEC, WEIGHTS_DIR  # noqa: E402

STUDY_SEED = 20261012


def fans(shape):
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    rf = int(np.prod(shape[:-2]))
    return shape[-2] * rf, shape[-1] * rf


def draw(init, shape, rng, units=None):
    shape = tuple(shape)
    if init == "zeros":
        return np.zeros(shape, np.float32)
    if init == "ones":
        return np.ones(shape, np.float32)
    if init == "glorot_uniform":
        fi, fo = fans(shape)
        lim = np.sqrt(6.0 / (fi + fo))
        return rng.uniform(-lim, lim, shape).astype(np.float32)
    if init == "glorot_normal":
        fi, fo = fans(shape)
        sd = np.sqrt(2.0 / (fi + fo)) / 0.87962566103423978
        a = rng.standard_normal(shape)
        bad = np.abs(a) > 2.0
        while bad.any():
            a[bad] = rng.standard_normal(int(bad.sum()))
            bad = np.abs(a) > 2.0
        return (a * sd).astype(np.float32)
    if init == "orthogonal":
        rows, cols = int(np.prod(shape[:-1])), shape[-1]
        a = rng.standard_normal((max(rows, cols), min(rows, cols)))
        q, r = np.linalg.qr(a)
        q *= np.sign(np.diag(r))
        if rows < cols:
            q = q.T
        return q.reshape(shape).astype(np.float32)
    if init == "lstm_unit_forget":
        b = np.zeros(shape, np.float32)
        b[units:2 * units] = 1.0
        return b
    if init == "uniform_0.05":
        return rng.uniform(-0.05, 0.05, shape).astype(np.float32)
    raise ValueError(init)


def weight_seed(model, layer, weight):
    return int.from_bytes(hashlib.sha256(f"{STUDY_SEED}|{model}|{layer}|{weight}".encode()).digest()[:8], "little")


def write_weights(model_name, arrays, kind, extra=None):
    """arrays: [(layer, name, ndarray)] in spec order -> <model>.<kind>.bin; returns its manifest entry."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f"{model_name}.{kind}.bin"
    tensors, off = [], 0
    blob = bytearray()
    for layer, name, a in arrays:
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        tensors.append({"layer": layer, "name": name, "shape": list(a.shape), "offset": off, "bytes": len(b),
                        "sha256": hashlib.sha256(b).hexdigest()})
        blob += b
        off += len(b)
    path.write_bytes(bytes(blob))
    return {"file": path.name, "bytes": len(blob), "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
            "tensors": tensors, **(extra or {})}


def update_manifest(model_name, kind, entry):
    man_p = WEIGHTS_DIR / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    man.setdefault(model_name, {})[kind] = entry
    man_p.write_text(json.dumps(dict(sorted(man.items())), indent=1), encoding="utf-8")


def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for m in spec["models"]:
        arrays = []
        for l in m["layers"]:
            for w in l.get("weights", []):
                rng = np.random.default_rng(weight_seed(m["name"], l["name"], w["name"]))
                arrays.append((l["name"], w["name"], draw(w["init"], w["shape"], rng, units=l.get("units"))))
        entry = write_weights(m["name"], arrays, "init",
                              {"generator": "scripts/make_init_weights.py", "study_seed": STUDY_SEED,
                               "numpy": np.__version__})
        update_manifest(m["name"], "init", entry)
        print(f"{m['name']:3s} {entry['bytes'] / 2**20:8.2f} MiB  sha256 {entry['sha256'][:16]}")


if __name__ == "__main__":
    main()
