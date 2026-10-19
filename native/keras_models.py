"""
Keras (TF 2.10.1) builder for public/spec/models.json: the native counterpart of src/models/build.js,
op for op. Runs in the tfgpu conda env.

Every weight starts at zero and is then set from the shared weight file (native/weights.py), the
same bytes TF.js loads. Semantics that differ between the frameworks' defaults are pinned by the spec:
  LSTM / GRU recurrent_activation 'sigmoid' (TF.js default hardSigmoid, Keras sigmoid)
  GRU reset_after=False (the only variant TF.js implements; Keras then can't use the cuDNN kernel,
      recorded per launch as native_cudnn_rnn)
  layer_norm, encoder_block: the same elementwise ops / reshape+matmul sequence as the TF.js layers
  loss: tfjs_cce, TF.js's categoricalCrossentropy expression (Keras's own CCE switches to a logits
      formulation when its input comes from a softmax op, which TF.js doesn't do)
"""
import json
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "public" / "spec" / "models.json"
EPSILON = 1e-7   # Keras backend epsilon == TF.js float32 backend epsilon


def load_spec():
    return json.loads(SPEC.read_text(encoding="utf-8"))


def model_spec(name):
    return next(m for m in load_spec()["models"] if m["name"] == name)


def layer_norm(x, gamma, beta, eps):
    mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
    return (x - mean) * tf.math.rsqrt(variance + eps) * gamma + beta


def linear(x, w, b):
    lead = tf.shape(x)[:-1]
    y = tf.matmul(tf.reshape(x, [-1, x.shape[-1]]), w) + b
    return tf.reshape(y, tf.concat([lead, [w.shape[1]]], 0))


class Positional(tf.keras.layers.Layer):
    def __init__(self, cfg, **kw):
        super().__init__(name=cfg["name"], **kw)
        self.cfg = cfg

    def build(self, s):
        self.table = self.add_weight("table", shape=(self.cfg["length"], self.cfg["dim"]), initializer="zeros")

    def call(self, x):
        return x + self.table


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, cfg, **kw):
        super().__init__(name=cfg["name"], **kw)
        self.cfg = cfg

    def build(self, s):
        self.gamma = self.add_weight("gamma", shape=(s[-1],), initializer="zeros")
        self.beta = self.add_weight("beta", shape=(s[-1],), initializer="zeros")

    def call(self, x):
        return layer_norm(x, self.gamma, self.beta, self.cfg["ln_eps"])


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, cfg, **kw):
        super().__init__(name=cfg["name"], **kw)
        self.cfg = cfg

    def build(self, s):
        for w in self.cfg["weights"]:   # spec order
            setattr(self, w["name"], self.add_weight(w["name"], shape=tuple(w["shape"]), initializer="zeros"))

    def call(self, x):
        T, d = x.shape[1], x.shape[2]
        B = tf.shape(x)[0]
        H = self.cfg["heads"]
        dh = d // H
        eps = self.cfg["ln_eps"]
        h = layer_norm(x, self.ln1_gamma, self.ln1_beta, eps)

        def heads(t):
            return tf.transpose(tf.reshape(t, [B, T, H, dh]), [0, 2, 1, 3])
        q = heads(linear(h, self.wq, self.bq))
        k = heads(linear(h, self.wk, self.bk))
        v = heads(linear(h, self.wv, self.bv))
        scores = tf.matmul(q, k, transpose_b=True) * (1.0 / float(dh) ** 0.5)
        ctx = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        merged = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [B, T, d])
        x1 = x + linear(merged, self.wo, self.bo)
        h2 = layer_norm(x1, self.ln2_gamma, self.ln2_beta, eps)
        f = linear(tf.nn.relu(linear(h2, self.w1, self.b1)), self.w2, self.b2)
        return x1 + f


def make_layer(l):
    z = "zeros"
    t = l["type"]
    if t == "flatten":
        return tf.keras.layers.Flatten(name=l["name"])
    if t == "reshape":
        return tf.keras.layers.Reshape(l["target_shape"], name=l["name"])
    if t == "dense":
        return tf.keras.layers.Dense(l["units"], activation=l["activation"], kernel_initializer=z,
                                     bias_initializer=z, name=l["name"])
    if t == "conv2d":
        return tf.keras.layers.Conv2D(l["filters"], l["kernel_size"], padding=l["padding"], activation=l["activation"],
                                      kernel_initializer=z, bias_initializer=z, name=l["name"])
    if t == "maxpool2d":
        return tf.keras.layers.MaxPooling2D(l["pool_size"], name=l["name"])
    if t == "simple_rnn":
        return tf.keras.layers.SimpleRNN(l["units"], activation=l["activation"], return_sequences=l["return_sequences"],
                                         kernel_initializer=z, recurrent_initializer=z, bias_initializer=z,
                                         name=l["name"])
    if t == "lstm":
        return tf.keras.layers.LSTM(l["units"], activation=l["activation"],
                                    recurrent_activation=l["recurrent_activation"], unit_forget_bias=False,
                                    return_sequences=l["return_sequences"], kernel_initializer=z,
                                    recurrent_initializer=z, bias_initializer=z, name=l["name"])
    if t == "gru":
        return tf.keras.layers.GRU(l["units"], activation=l["activation"],
                                   recurrent_activation=l["recurrent_activation"], reset_after=l["reset_after"],
                                   return_sequences=l["return_sequences"], kernel_initializer=z,
                                   recurrent_initializer=z, bias_initializer=z, name=l["name"])
    if t == "positional":
        return Positional(l)
    if t == "encoder_block":
        return EncoderBlock(l)
    if t == "layer_norm":
        return LayerNorm(l)
    if t == "global_avg_pool_1d":
        return tf.keras.layers.GlobalAveragePooling1D(name=l["name"])
    raise ValueError(f"unknown layer type {t}")


def build_model(spec):
    model = tf.keras.Sequential(name=spec["name"])
    model.add(tf.keras.Input(shape=tuple(spec["input_shape"])))
    for l in spec["layers"]:
        model.add(make_layer(l))
    return model


def weight_slots(model, spec):
    """[(layer, name, shape, variable)] in spec order, checked against the spec."""
    slots = []
    for layer, l in zip(model.layers, spec["layers"]):
        want = l.get("weights", [])
        have = layer.weights
        if len(have) != len(want):
            raise ValueError(f"{spec['name']}/{l['name']}: {len(have)} weights, spec lists {len(want)}")
        for v, w in zip(have, want):
            if list(v.shape) != list(w["shape"]):
                raise ValueError(f"{spec['name']}/{l['name']}/{w['name']}: shape {list(v.shape)} != spec {w['shape']}")
            slots.append((l["name"], w["name"], list(w["shape"]), v))
    if len(model.layers) != len(spec["layers"]):
        raise ValueError(f"{spec['name']}: {len(model.layers)} layers, spec lists {len(spec['layers'])}")
    return slots


def tfjs_cce(y_true, y_pred):
    """TF.js tfjs-layers categoricalCrossentropy (fromLogits=false), per sample."""
    p = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    p = tf.clip_by_value(p, EPSILON, 1.0 - EPSILON)
    return -tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.math.log(p), axis=-1)


if __name__ == "__main__":
    # Same check as scripts/check_spec_js.mjs, for scripts/check_spec.py.
    import sys
    tf.config.set_visible_devices([], "GPU")
    only = sys.argv[1:]
    out = {}
    for m in load_spec()["models"]:
        if only and m["name"] not in only:
            continue
        try:
            model = build_model(m)
            slots = weight_slots(model, m)
            y = model(tf.zeros([2, *m["input_shape"]]))
            out[m["name"]] = {"ok": True, "weights": [[a, b, c] for a, b, c, _ in slots],
                              "params": int(model.count_params()), "output_shape": list(y.shape)}
        except Exception as e:  # noqa: BLE001
            out[m["name"]] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print("RESULT " + json.dumps(out))
