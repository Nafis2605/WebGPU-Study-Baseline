"""
The study's CIFAR-10 subsets for the native worker: the same public/data/ files the page loads
(src/data.js), sha256-checked against manifest.json, normalised with the same LUT, and batched with
the same rules (trainRows / valRows).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public" / "data"
PIXELS, CLASSES = 32 * 32 * 3, 10


def lut():
    return (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


class Data:
    def __init__(self, base=DATA):
        man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        arrays = {}
        for name in ("train_x.u8", "train_y.u8", "val_x.u8", "val_y.u8", "order.u32"):
            raw = (base / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != man["files"][name]["sha256"]:
                raise RuntimeError(f"data_mismatch: {name} sha256 differs from manifest")
            info = man["files"][name]
            arrays[name] = np.frombuffer(raw, dtype=np.dtype(info["dtype"]).newbyteorder("<")).reshape(info["shape"])
        self.lut = lut()
        if hashlib.sha256(self.lut.tobytes()).hexdigest() != man["lut_sha256"]:
            raise RuntimeError("data_mismatch: normalisation LUT sha256 differs from manifest")
        self.manifest = man
        self.train_x, self.train_y = arrays["train_x.u8"], arrays["train_y.u8"]
        self.val_x, self.val_y = arrays["val_x.u8"], arrays["val_y.u8"]
        self.order = arrays["order.u32"]
        self.n_train, self.n_val = len(self.train_y), len(self.val_y)

    def images(self, x, rows):
        return self.lut[x[rows]]                       # [B, 32, 32, 3] float32

    def one_hot(self, y, rows):
        out = np.zeros((len(rows), CLASSES), np.float32)
        out[np.arange(len(rows)), y[rows]] = 1.0
        return out

    def train_rows(self, k, B):
        N = self.n_train
        if N % B:
            raise ValueError(f"batch {B} does not divide the training subset ({N})")
        start = k * B
        epoch = start // N
        if epoch >= self.order.shape[0]:
            raise ValueError(f"step {k} at batch {B} is beyond the epochs in order.u32")
        off = start % N
        return self.order[epoch, off:off + B], epoch

    def val_rows(self, i, B):
        return (i * B + np.arange(B)) % self.n_val

    def train_batch(self, k, B):
        rows, epoch = self.train_rows(k, B)
        return self.images(self.train_x, rows), self.one_hot(self.train_y, rows), epoch

    def val_batch(self, i, B):
        rows = self.val_rows(i, B)
        return self.images(self.val_x, rows), self.one_hot(self.val_y, rows)
