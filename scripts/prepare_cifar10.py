"""
Fixed CIFAR-10 subsets for the study, from the official binary release (no pickles):
  downloads/cifar10/cifar-10-binary.tar.gz   (https://www.cs.toronto.edu/~kriz/, md5 checked)

Outputs in public/data/ (served same-origin; binaries gitignored, manifest tracked):
  train_x.u8   [5120, 32, 32, 3] uint8 HWC   512 images per class from the 50k training batches
  train_y.u8   [5120] uint8 class labels
  val_x.u8     [1000, 32, 32, 3] uint8 HWC   100 images per class from the 10k test batch
  val_y.u8     [1000] uint8
  order.u32    [10, 5120] uint32             one permutation of the train subset per epoch
  train_src.u32 / val_src.u32                source index of every selected image (provenance)
  manifest.json                              shapes, sha256 of every file, seed, class counts, the
                                             source archive's md5/sha256, and the normalisation LUT

Normalisation (both runtimes): x_f32 = LUT[x_u8], LUT[i] = float32(i / 255) with the division done
in float64 and rounded once to float32 (JS: Math.fround(i / 255)). The LUT's sha256 is in the
manifest and checked by the page and the native worker.

5120 = 512 per class: divisible by every training batch size in the study (16..256), so no partial
batch (and no extra shape) ever occurs. Usage:  .venv/Scripts/python -I scripts/prepare_cifar10.py
"""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "downloads" / "cifar10" / "cifar-10-binary.tar.gz"
OUT = ROOT / "public" / "data"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"   # published on https://www.cs.toronto.edu/~kriz/cifar.html
SEED = 20261012
TRAIN_PER_CLASS, VAL_PER_CLASS, EPOCHS = 512, 100, 10
REC = 1 + 3072


def lut():
    return (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


def read_batches(tar, names):
    xs, ys = [], []
    for n in names:
        raw = tar.extractfile(f"cifar-10-batches-bin/{n}").read()
        a = np.frombuffer(raw, np.uint8).reshape(-1, REC)
        ys.append(a[:, 0].copy())
        xs.append(a[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy())   # CHW -> HWC
    return np.concatenate(xs), np.concatenate(ys)


def stratified(y, per_class, rng):
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(10)])
    return rng.permutation(idx)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    raw = ARCHIVE.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"archive md5 {md5} != published {MD5}")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tx, ty = read_batches(tar, [f"data_batch_{i}.bin" for i in range(1, 6)])
        vx, vy = read_batches(tar, ["test_batch.bin"])
    assert tx.shape == (50000, 32, 32, 3) and vx.shape == (10000, 32, 32, 3)
    rng = np.random.default_rng(SEED)
    ti = stratified(ty, TRAIN_PER_CLASS, rng)
    vi = stratified(vy, VAL_PER_CLASS, rng)
    n = len(ti)
    order = np.stack([rng.permutation(n) for _ in range(EPOCHS)]).astype(np.uint32)
    files = {
        "train_x.u8": (tx[ti], "uint8"), "train_y.u8": (ty[ti], "uint8"),
        "val_x.u8": (vx[vi], "uint8"), "val_y.u8": (vy[vi], "uint8"),
        "order.u32": (order, "uint32"),
        "train_src.u32": (ti.astype(np.uint32), "uint32"), "val_src.u32": (vi.astype(np.uint32), "uint32"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"dataset": "CIFAR-10 (binary release)", "source": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
           "archive_md5": md5, "archive_sha256": sha(raw), "seed": SEED, "layout": "NHWC uint8, little-endian",
           "normalisation": "float32(x / 255) via LUT", "lut_sha256": sha(lut().tobytes()),
           "epochs_in_order": EPOCHS, "files": {}}
    for name, (a, dt) in files.items():
        b = np.ascontiguousarray(a, dtype=np.dtype(dt).newbyteorder("<")).tobytes()
        (OUT / name).write_bytes(b)
        man["files"][name] = {"shape": list(a.shape), "dtype": dt, "bytes": len(b), "sha256": sha(b)}
    man["train_class_counts"] = np.bincount(ty[ti], minlength=10).tolist()
    man["val_class_counts"] = np.bincount(vy[vi], minlength=10).tolist()
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))
    for k, v in man["files"].items():
        print(f"{k:14s} {v['shape']} {v['sha256'][:16]}")


if __name__ == "__main__":
    main()
