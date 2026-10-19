"""
Generate public/spec/models.json: the single architecture spec read by BOTH model builders
(src/models/build.js for TF.js, native/keras_models.py for Keras), so the two can't drift.
scripts/check_spec.py verifies that both builders produce exactly the weights listed here.

Families (CIFAR-10 input 32x32x3, 10 classes, softmax output):
  D1-D5  MLP        Flatten -> Dense ReLU hidden layers -> Dense10          (legacy, unchanged)
  C1-C5  CNN        Conv64-Pool-Conv128-Pool -> Flatten -> Dense head       (legacy, unchanged)
  R1-R5  SimpleRNN  32 rows x 96 features -> stacked SimpleRNN(ReLU)        (legacy, unchanged)
  L1-L5  LSTM       same layers x units grid as R1-R5
  G1-G5  GRU        same grid (reset_after=False: the only GRU variant TF.js implements)
  T1-T5  Transformer encoder on the same 32 row-tokens (pre-LN, attention from matMul)

Size grid (index k = 1..5): k layers of width 64 * 2^(k-1)  ->  64, 128x2, 256x3, 512x4, 1024x5.
Legacy batch sizes: 64 for D/C, 32 for R (kept for L/G/T).

Initialisers are named per weight; scripts/make_init_weights.py draws them (seeded, numpy) so both
runtimes start from the same bytes. Legacy initialisers are kept: glorot_uniform for D and R
(explicit in the legacy code), glorot_normal for C (TF.js's default, which the legacy C models used).
L/G/T use Keras defaults (glorot_uniform kernels, orthogonal recurrent kernels, LSTM unit forget bias).

Usage:  .venv/Scripts/python scripts/make_spec.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "spec" / "models.json"

INPUT_SHAPE = [32, 32, 3]
NUM_CLASSES = 10
SEQ = [32, 96]          # rows as timesteps / tokens: 32 x (32 pixels * 3 channels)
LN_EPS = 1e-5
GRID = {k: [64 * 2 ** (k - 1)] * k for k in range(1, 6)}   # k -> hidden widths


def dense(units, act, kinit, name):
    return {"type": "dense", "name": name, "units": units, "activation": act,
            "weights": [{"name": "kernel", "init": kinit}, {"name": "bias", "init": "zeros"}]}


def head(k, kinit):
    return [dense(u, "relu", kinit, f"dense_{i}") for i, u in enumerate(GRID[k])]


def out_layer(kinit):
    return dense(NUM_CLASSES, "softmax", kinit, "logits")


def mlp(k):
    return [{"type": "flatten", "name": "flatten"}, *head(k, "glorot_uniform"), out_layer("glorot_uniform")]


def cnn(k):
    conv = lambda f, n: {"type": "conv2d", "name": n, "filters": f, "kernel_size": [3, 3], "padding": "same",
                         "activation": "relu",
                         "weights": [{"name": "kernel", "init": "glorot_normal"}, {"name": "bias", "init": "zeros"}]}
    pool = lambda n: {"type": "maxpool2d", "name": n, "pool_size": [2, 2]}
    return [conv(64, "conv_0"), pool("pool_0"), conv(128, "conv_1"), pool("pool_1"), {"type": "flatten", "name": "flatten"},
            *head(k, "glorot_normal"), out_layer("glorot_normal")]


def rnn(k, cell):
    layers = [{"type": "reshape", "name": "rows", "target_shape": SEQ}]
    widths = GRID[k]
    for i, u in enumerate(widths):
        l = {"type": cell, "name": f"{cell}_{i}", "units": u, "return_sequences": i < len(widths) - 1}
        if cell == "simple_rnn":
            l.update(activation="relu",
                     weights=[{"name": "kernel", "init": "glorot_uniform"},
                              {"name": "recurrent_kernel", "init": "glorot_uniform"},
                              {"name": "bias", "init": "zeros"}])
        elif cell == "lstm":
            l.update(activation="tanh", recurrent_activation="sigmoid",
                     weights=[{"name": "kernel", "init": "glorot_uniform"},
                              {"name": "recurrent_kernel", "init": "orthogonal"},
                              {"name": "bias", "init": "lstm_unit_forget"}])
        else:
            l.update(activation="tanh", recurrent_activation="sigmoid", reset_after=False,
                     weights=[{"name": "kernel", "init": "glorot_uniform"},
                              {"name": "recurrent_kernel", "init": "orthogonal"},
                              {"name": "bias", "init": "zeros"}])
        layers.append(l)
    return [*layers, out_layer("glorot_uniform")]


def transformer(k):
    d = GRID[k][0]
    n_blocks = k
    heads = max(1, d // 64)
    layers = [{"type": "reshape", "name": "rows", "target_shape": SEQ},
              dense(d, "linear", "glorot_uniform", "token_proj"),
              {"type": "positional", "name": "pos", "length": SEQ[0], "dim": d,
               "weights": [{"name": "table", "init": "uniform_0.05"}]}]
    for b in range(n_blocks):
        w = [{"name": "ln1_gamma", "init": "ones"}, {"name": "ln1_beta", "init": "zeros"}]
        for p in ("q", "k", "v", "o"):
            w += [{"name": f"w{p}", "init": "glorot_uniform"}, {"name": f"b{p}", "init": "zeros"}]
        w += [{"name": "ln2_gamma", "init": "ones"}, {"name": "ln2_beta", "init": "zeros"},
              {"name": "w1", "init": "glorot_uniform"}, {"name": "b1", "init": "zeros"},
              {"name": "w2", "init": "glorot_uniform"}, {"name": "b2", "init": "zeros"}]
        layers.append({"type": "encoder_block", "name": f"block_{b}", "dim": d, "heads": heads, "ffn": 4 * d,
                       "ln_eps": LN_EPS, "weights": w})
    layers += [{"type": "layer_norm", "name": "final_ln", "ln_eps": LN_EPS,
                "weights": [{"name": "gamma", "init": "ones"}, {"name": "beta", "init": "zeros"}]},
               {"type": "global_avg_pool_1d", "name": "pool"},
               out_layer("glorot_uniform")]
    return layers


# ---- weight shapes (both builders must reproduce these exactly) ----

def infer_shapes(layers):
    """Walk the layer list, fill in each weight's shape and each layer's output shape."""
    shape = list(INPUT_SHAPE)
    for l in layers:
        t = l["type"]
        shapes = {}
        if t == "flatten":
            n = 1
            for s in shape:
                n *= s
            shape = [n]
        elif t == "reshape":
            shape = list(l["target_shape"])
        elif t == "dense":
            fin = shape[-1]
            shapes = {"kernel": [fin, l["units"]], "bias": [l["units"]]}
            shape = [*shape[:-1], l["units"]]
        elif t == "conv2d":
            kh, kw = l["kernel_size"]
            shapes = {"kernel": [kh, kw, shape[-1], l["filters"]], "bias": [l["filters"]]}
            shape = [shape[0], shape[1], l["filters"]]
        elif t == "maxpool2d":
            ph, pw = l["pool_size"]
            shape = [shape[0] // ph, shape[1] // pw, shape[2]]
        elif t in ("simple_rnn", "lstm", "gru"):
            g = {"simple_rnn": 1, "lstm": 4, "gru": 3}[t]
            u = l["units"]
            shapes = {"kernel": [shape[-1], g * u], "recurrent_kernel": [u, g * u], "bias": [g * u]}
            shape = [shape[0], u] if l["return_sequences"] else [u]
        elif t == "positional":
            shapes = {"table": [l["length"], l["dim"]]}
        elif t == "encoder_block":
            d, f = l["dim"], l["ffn"]
            for n in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "bq", "bk", "bv", "bo", "b2"):
                shapes[n] = [d]
            for n in ("wq", "wk", "wv", "wo"):
                shapes[n] = [d, d]
            shapes.update(w1=[d, f], b1=[f], w2=[f, d])
        elif t == "layer_norm":
            shapes = {"gamma": [shape[-1]], "beta": [shape[-1]]}
        elif t == "global_avg_pool_1d":
            shape = [shape[-1]]
        else:
            raise ValueError(t)
        for w in l.get("weights", []):
            w["shape"] = shapes[w["name"]]
        l["output_shape"] = list(shape)
    return layers


def model(name, family, k, batch, layers):
    layers = infer_shapes(layers)
    params = 0
    for l in layers:
        for w in l.get("weights", []):
            n = 1
            for s in w["shape"]:
                n *= s
            params += n
    assert layers[-1]["output_shape"] == [NUM_CLASSES], name
    return {"name": name, "family": family, "size": k, "batch": batch, "input_shape": INPUT_SHAPE,
            "num_classes": NUM_CLASSES, "params": params, "layers": layers}


def main():
    models = []
    for k in range(1, 6):
        models.append(model(f"D{k}", "mlp", k, 64, mlp(k)))
    for k in range(1, 6):
        models.append(model(f"C{k}", "cnn", k, 64, cnn(k)))
    for fam, cell, p in (("simple_rnn", "simple_rnn", "R"), ("lstm", "lstm", "L"), ("gru", "gru", "G")):
        for k in range(1, 6):
            models.append(model(f"{p}{k}", fam, k, 32, rnn(k, cell)))
    for k in range(1, 6):
        models.append(model(f"T{k}", "transformer", k, 32, transformer(k)))
    spec = {
        "schema": 1,
        "dataset": "cifar10",
        "input_shape": INPUT_SHAPE,
        "num_classes": NUM_CLASSES,
        # epsilon pinned for every backend and natively: TF.js's default (tf.backend().epsilon()) is 1e-4 on
        # WebGL 4.22.0 desktop (floatPrecision() returns 16 when WEBGL_RENDER_FLOAT32_ENABLED), 1e-7 elsewhere.
        "optimizer": {"type": "adam", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7},
        "loss": "categorical_crossentropy_tfjs",
        "loss_epsilon": 1e-7,
        "models": models,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    for m in models:
        print(f"{m['name']:3s} {m['family']:11s} batch {m['batch']:3d} params {m['params']:>11,d}")


if __name__ == "__main__":
    main()
