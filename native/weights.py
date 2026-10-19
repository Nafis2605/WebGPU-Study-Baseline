"""Load public/weights/<model>.<kind>.bin (sha256-checked) into a Keras model built from the spec."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "public" / "weights"


def read_weights(model, kind):
    entry = json.loads((WEIGHTS / "manifest.json").read_text(encoding="utf-8"))[model][kind]
    raw = (WEIGHTS / entry["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"weights_mismatch: {entry['file']} sha256 differs from manifest")
    arrays = [np.frombuffer(raw, "<f4", t["bytes"] // 4, t["offset"]).reshape(t["shape"]) for t in entry["tensors"]]
    return entry, arrays


def assign(slots, entry, arrays):
    """slots: keras_models.weight_slots(); writes every variable in spec order."""
    if len(slots) != len(arrays):
        raise RuntimeError(f"weights_mismatch: {len(arrays)} tensors for {len(slots)} slots")
    for (layer, name, shape, var), t, a in zip(slots, entry["tensors"], arrays):
        if (t["layer"], t["name"], t["shape"]) != (layer, name, shape):
            raise RuntimeError(f"weights_mismatch: file {t['layer']}/{t['name']} vs slot {layer}/{name}")
        var.assign(a)


def verify(slots, entry):
    """Read every variable back and compare its bytes with the manifest. Returns mismatching names."""
    bad = []
    for (layer, name, _, var), t in zip(slots, entry["tensors"]):
        if hashlib.sha256(np.ascontiguousarray(var.numpy(), "<f4").tobytes()).hexdigest() != t["sha256"]:
            bad.append(f"{layer}/{name}")
    return bad
