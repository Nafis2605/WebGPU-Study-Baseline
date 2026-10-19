"""
Trained weights for the inference pillar: public/weights/<model>.trained.bin (+ manifest entry "trained").

Inference timing doesn't depend on weight values, but inference correctness (top-1 agreement, cosine)
is only meaningful on a model whose outputs aren't near-uniform, so every inference stage loads weights
trained once here. Each model starts from its study initial weights (<model>.init.bin) and is trained
natively (TF 2.10.1, GPU, TF32 off) for EPOCHS epochs over the fixed 5120-image subset in the order of
public/data/order.u32, with the study's own step (native/worker.py: tfjs_cce loss, Keras Adam with the
spec's hyperparameters, spec batch size). Op determinism is requested; if TensorFlow rejects it for an
op the model needs, training runs without it and the manifest records deterministic=false.
Validation loss/accuracy on the 1000-image subset is recorded as provenance (not a study result).

Runs in the tfgpu env:  C:\\tfjsbench\\tfgpu\\python.exe scripts/make_trained_weights.py [MODEL ...]
(PATH must include C:\\tfjsbench\\tfgpu\\Library\\bin; scripts/run_tfgpu.py sets it.)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
EPOCHS = 10


def write_entry(model_name, arrays, extra):
    wdir = ROOT / "public" / "weights"
    path = wdir / f"{model_name}.trained.bin"
    tensors, off, blob = [], 0, bytearray()
    for layer, name, a in arrays:
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        tensors.append({"layer": layer, "name": name, "shape": list(a.shape), "offset": off, "bytes": len(b),
                        "sha256": hashlib.sha256(b).hexdigest()})
        blob += b
        off += len(b)
    path.write_bytes(bytes(blob))
    entry = {"file": path.name, "bytes": len(blob), "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
             "tensors": tensors, **extra}
    man_p = wdir / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    man[model_name]["trained"] = entry
    man_p.write_text(json.dumps(dict(sorted(man.items())), indent=1), encoding="utf-8")
    return entry


def train_one(name, deterministic):
    import tensorflow as tf
    import keras_models as km
    import weights as wt
    from data import Data
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
    data = Data()
    spec = km.model_spec(name)
    opt_spec = km.load_spec()["optimizer"]
    model = km.build_model(spec)
    slots = km.weight_slots(model, spec)
    entry, arrays = wt.read_weights(name, "init")
    wt.assign(slots, entry, arrays)
    variables = [v for *_, v in slots]
    opt = tf.keras.optimizers.Adam(opt_spec["learning_rate"], opt_spec["beta1"], opt_spec["beta2"],
                                   epsilon=opt_spec["epsilon"])

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(km.tfjs_cce(y, model(x, training=True)))
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    @tf.function
    def evaluate(x, y):
        p = model(x, training=False)
        return (tf.reduce_sum(km.tfjs_cce(y, p)),
                tf.reduce_sum(tf.cast(tf.equal(tf.argmax(p, -1), tf.argmax(y, -1)), tf.float32)))

    B = spec["batch"]
    per = data.n_train // B
    t0 = time.perf_counter()
    hist = []
    for e in range(EPOCHS):
        losses = []
        for i in range(per):
            x, y, _ = data.train_batch(e * per + i, B)
            losses.append(float(step(tf.constant(x), tf.constant(y)).numpy()))
        ls = cs = 0.0
        for st in range(0, data.n_val, B):
            rows = np.arange(st, min(st + B, data.n_val))
            l, c = evaluate(tf.constant(data.images(data.val_x, rows)), tf.constant(data.one_hot(data.val_y, rows)))
            ls += float(l.numpy())
            cs += float(c.numpy())
        hist.append({"epoch": e, "train_loss_mean": float(np.mean(losses)), "val_loss": ls / data.n_val,
                     "val_acc": cs / data.n_val})
        print(f"  {name} epoch {e}: train {hist[-1]['train_loss_mean']:.4f} val {hist[-1]['val_loss']:.4f} "
              f"acc {hist[-1]['val_acc']:.3f}", flush=True)
    arrays = [(layer, wname, v.numpy()) for layer, wname, _, v in slots]
    if any(not np.isfinite(a).all() for *_, a in arrays):
        raise RuntimeError(f"{name}: non-finite trained weights")
    return arrays, {"generator": "scripts/make_trained_weights.py", "from": "init", "epochs": EPOCHS, "batch": B,
                    "device": "GPU:0 (TF32 off)", "deterministic": deterministic, "tensorflow": tf.__version__,
                    "train_seconds": time.perf_counter() - t0, "history": hist,
                    "final_val_acc": hist[-1]["val_acc"], "final_val_loss": hist[-1]["val_loss"]}


def main():
    import subprocess
    names = sys.argv[1:]
    if names == ["--child"] or (names and names[0] == "--child"):
        name, det = sys.argv[2], sys.argv[3] == "1"
        arrays, extra = train_one(name, det)
        e = write_entry(name, arrays, extra)
        print(f"WROTE {name} {e['sha256'][:16]} val_acc {extra['final_val_acc']:.3f}", flush=True)
        return
    spec = json.loads((ROOT / "public" / "spec" / "models.json").read_text(encoding="utf-8"))
    todo = [m["name"] for m in spec["models"] if not names or m["name"] in names]
    for name in todo:
        # One process per model (clean TF state; determinism is a process-wide switch).
        for det in ("1", "0"):
            p = subprocess.run([sys.executable, __file__, "--child", name, det], capture_output=True, text=True,
                               errors="replace")
            print(p.stdout.strip(), flush=True)
            if p.returncode == 0:
                break
            print(f"  {name}: deterministic={det} failed: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '?'}",
                  flush=True)
        else:
            raise SystemExit(f"{name}: training failed")


if __name__ == "__main__":
    main()
