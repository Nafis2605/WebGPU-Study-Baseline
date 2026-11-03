"""
Convert the 30 spec models with their trained weights to TFLite (float32) for the Android native baseline.

Same builders (native/keras_models.py) and the same sha256-checked weight bytes (native/weights.py) as the TF.js page and
the laptop's native runs; no quantization, no optimisation flags. Builtin ops only: a model that needs TF Select ops is
recorded as a conversion gap, never converted with a fallback. The batch dimension is static (one file per batch size): Keras RNN loops only lower to TFLite builtins with static\nshapes, and the GPU delegate needs static shapes anyway.

Runs once on the laptop in the tfgpu env (data preparation, like scripts/make_trained_weights.py):
    C:\\tfjsbench\\tfgpu\\python.exe native/tflite/convert.py [--filter L1 C1]
Output: public/tflite/<model>.b<batch>.trained.tflite and public/tflite/manifest.json (sha256, bytes, input/output shapes, status).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native"))
import keras_models as km  # noqa: E402
import weights as wt  # noqa: E402

OUT = ROOT / "public" / "tflite"
REF = OUT / "ref"


def convert(spec, batch):
    model = km.build_model(spec)
    slots = km.weight_slots(model, spec)
    entry, arrays = wt.read_weights(spec["name"], "trained")
    wt.assign(slots, entry, arrays)
    bad = wt.verify(slots, entry)
    if bad:
        raise RuntimeError(f"weights_mismatch: {bad[:5]}")
    fn = tf.function(lambda x: model(x, training=False))
    cf = fn.get_concrete_function(tf.TensorSpec([batch, *spec["input_shape"]], tf.float32))
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert(), entry["sha256"], model


def reference(model, spec, rows):
    """Correctness reference (laptop, data preparation): the Keras output of the same model on the first `rows` validation
    images, normalised as on every other path (x / 255 via the data LUT). Written as raw float32 next to the input."""
    import data as dt
    d = dt.Data()
    x = d.images(d.val_x, list(range(rows)))
    REF.mkdir(parents=True, exist_ok=True)
    xin = REF / f"val{rows}.in.f32"
    if not xin.exists():
        xin.write_bytes(x.astype("float32").tobytes())
    y = model(tf.constant(x), training=False).numpy().astype("float32")
    (REF / f"{spec['name']}.val{rows}.keras.f32").write_bytes(y.tobytes())
    return {"input": xin.name, "input_sha256": hashlib.sha256(xin.read_bytes()).hexdigest(),
            "output": f"{spec['name']}.val{rows}.keras.f32", "output_shape": list(y.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", nargs="*")
    ap.add_argument("--batches", nargs="*", type=int, default=[1])
    ap.add_argument("--reference-rows", type=int, default=32,
                    help="for this batch, also write the Keras output on the first N validation images (public/tflite/ref)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    manifest["_meta"] = {"tensorflow": tf.__version__, "ops": "TFLITE_BUILTINS", "dtype": "float32", "weights": "trained"}
    for spec in km.load_spec()["models"]:
        if a.filter and not any(spec["name"] == f or spec["name"].startswith(f) for f in a.filter):
            continue
        for batch in a.batches:
            key = f"{spec['name']}.b{batch}"
            t0 = time.time()
            try:
                blob, wsha, model = convert(spec, batch)
                f = OUT / f"{key}.trained.tflite"
                f.write_bytes(blob)
                interp = tf.lite.Interpreter(model_content=blob)
                ops = sorted({d["op_name"] for d in interp._get_ops_details()})
                io = {"input_name": interp.get_input_details()[0]["name"],
                      "output_name": interp.get_output_details()[0]["name"],
                      "output_shape": [int(v) for v in interp.get_output_details()[0]["shape"]]}
                manifest[key] = {"status": "ok", "model": spec["name"], "batch": batch, "file": f.name, "bytes": len(blob),
                                 "sha256": hashlib.sha256(blob).hexdigest(), "weights_sha256": wsha,
                                 "input_shape": [batch, *spec["input_shape"]], "ops": ops, **io,
                                 "seconds": round(time.time() - t0, 1)}
                if batch == a.reference_rows:
                    manifest[key]["reference"] = reference(model, spec, batch)
            except Exception as e:  # noqa: BLE001
                manifest[key] = {"status": "conversion_gap", "model": spec["name"], "batch": batch,
                                 "error": f"{type(e).__name__}: {str(e)[-400:]}"}
            print(key, manifest[key]["status"], manifest[key].get("ops", ""), flush=True)
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
