"""
One native TensorFlow launch for the custom-model study (run by native/run_native.py in a fresh process,
conda env `tfgpu`: TF 2.10.1, Keras 2.10.0, CUDA 11.2, cuDNN 8.1). Reads a job JSON (argv[1]); prints one
"RESULT <json>" line in the browser harness's experiment schema (benchkit.results builds the same CSVs).
Structure ported from the sibling TF.js study's native/worker.py.

Same model spec (native/keras_models.py), same weight bytes (native/weights.py), same data and batch order
(native/data.py), same phase labels as the page (src/infer-modes.js, src/train-modes.js).

Configs (job["config"]):
  tf-cuda           TensorFlow defaults on the GPU (TF32 tensor-core math on Ampere), graph mode (tf.function)
  tf-cuda-fp32      the same with TF32 disabled (full FP32 math like WebGPU / WebGL): primary parity config
  tf-cuda-eager     FP32, no tf.function: every op dispatched from Python one at a time, like TF.js's eager
                    execution (why-axis: how much of the gap is graph compilation)
  tf-cuda-resident  inference input / training batch placed on the GPU before the timed region
  tf-cpu            CPU only (CUDA_VISIBLE_DEVICES=-1 set by the driver), 8 intra-op threads (counterpart of wasm-8t)

Timing (same definitions as the page):
  inference  dur_ms = input array -> tensor -> model -> output as a host array; exec_ms = until the call returned;
             read_ms = .numpy() (waits for the device)
  training   prep (host batch gather + normalise) outside; t0 -> tensors -> train_step (forward, loss, gradients,
             optimizer update; returns the loss with a control dependency on the update) -> t1 = exec_ms ->
             loss.numpy() -> t2 = dur_ms
Phases timed separately at launch (cold start): backend_init_ms (import TF + device init), fetch_ms (weight
file read + sha256), build_ms, weights_assign_ms, trace_ms (tf.function tracing of the step / inference
function, done on the first call; recorded as part of the first run like TF.js's shader compilation, and
also measured on its own with get_concrete_function before the first run when job["separate_trace"]).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
CONFIGS = ("tf-cuda", "tf-cuda-fp32", "tf-cuda-eager", "tf-cuda-resident", "tf-cpu")
INFER_MODES = ("latency", "batch", "correctness", "shapes")
TRAIN_MODES = ("train", "train_batch", "train_long", "train_fit", "train_phases", "train_correctness", "lifecycle")


def now_ms():
    return time.time() * 1e3


# ---- bit-exact ports of the page's sketch PRNG (src/common.js) ----

def fnv1a32(s):
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def mulberry32_signs(seed, n):
    """+-1 signs of n successive mulberry32 draws (rand < 0.5 -> -1), vectorised but bit-exact."""
    import numpy as np
    a = (seed + 0x6D2B79F5 * np.arange(1, n + 1, dtype=np.uint64)) & 0xFFFFFFFF
    t = a.astype(np.uint64)
    t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
    t = t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    r = (t ^ (t >> 14)) & 0xFFFFFFFF
    return np.where(r < 2 ** 31, -1.0, 1.0)


def sketch(values, name, k=4):
    import numpy as np
    v = np.asarray(values, np.float64).ravel()
    return [float(mulberry32_signs(fnv1a32(f"{name}/{j}"), v.size) @ v) for j in range(k)]


def encode(name, arr):
    import base64
    import numpy as np
    arr = np.ascontiguousarray(arr, np.float32)
    return {"name": name, "type": "float32", "dims": list(arr.shape), "b64": base64.b64encode(arr.tobytes()).decode()}


class TfjsAdam:
    """TF.js 4.22.0 Adam (tfjs-core adam_optimizer.js), op for op, for the correctness stage:
    m = m*b1 + g*(1-b1); v = v*b2 + g^2*(1-b2); w += -lr * (m/(1-accB1)) / (sqrt(v/(1-accB2)) + eps);
    accB1 *= b1; accB2 *= b2 (accB* start at b1, b2)."""

    def __init__(self, tf, lr, b1, b2, eps):
        self.tf, self.lr, self.b1, self.b2, self.eps = tf, lr, b1, b2, eps
        self.acc1 = tf.Variable(b1, dtype=tf.float32, trainable=False)
        self.acc2 = tf.Variable(b2, dtype=tf.float32, trainable=False)
        self.m, self.v = {}, {}

    def apply_gradients(self, pairs):
        tf = self.tf
        om1, om2 = 1.0 - self.acc1, 1.0 - self.acc2
        ops = []
        for g, w in pairs:
            k = w.ref()
            if k not in self.m:
                self.m[k] = tf.Variable(tf.zeros_like(w), trainable=False)
                self.v[k] = tf.Variable(tf.zeros_like(w), trainable=False)
            m = self.m[k] * self.b1 + g * (1 - self.b1)
            v = self.v[k] * self.b2 + tf.square(g) * (1 - self.b2)
            self.m[k].assign(m)
            self.v[k].assign(v)
            ops.append(w.assign((m / om1) / (tf.sqrt(v / om2) + self.eps) * (-self.lr) + w))
        self.acc1.assign(self.acc1 * self.b1)
        self.acc2.assign(self.acc2 * self.b2)
        return ops


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = job["config"]
    assert cfg in CONFIGS, cfg
    mode = job["mode"]
    pillar = "train" if mode in TRAIN_MODES else "infer"
    res = {"schema": 3, "model": job["model"], "backend": "native", "config_label": cfg, "mode": mode, "pillar": pillar,
           "status": "failed", "failure": None, "runs": [], "capped": [],
           "config": {"warmup": job.get("warmup"), "measure": job.get("measure"), "seed": None, "strict": False,
                      "threads_requested": job.get("threads"), "flags": {"tf32": cfg in ("tf-cuda", "tf-cuda-resident")},
                      "exec": "eager" if cfg == "tf-cuda-eager" else "tf.function", "precompile": False,
                      "instrumented": False, "weights": job.get("weights"),
                      "train_batch": None, "infer_batch": job.get("infer_batch", 1),
                      "sync_policy": "loss", "data_io": "resident" if cfg == "tf-cuda-resident" else "host",
                      "max_measure_s": job.get("max_measure_s"), "min_measure": job.get("min_measure")}}
    stage = "import"
    try:
        import numpy as np
        t0 = time.perf_counter()
        import tensorflow as tf
        if cfg == "tf-cpu":
            tf.config.threading.set_intra_op_parallelism_threads(job.get("threads") or 8)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        if cfg in ("tf-cuda-fp32", "tf-cuda-eager"):
            tf.config.experimental.enable_tensor_float_32_execution(False)
        if job.get("determinism"):
            tf.config.experimental.enable_op_determinism()
        gpus = tf.config.list_physical_devices("GPU")
        bi = tf.sysconfig.get_build_info()
        if cfg.startswith("tf-cuda"):
            if not gpus:
                raise RuntimeError("backend_unavailable: no GPU visible to TensorFlow")
            for gp in gpus:
                tf.config.experimental.set_memory_growth(gp, True)
            dev = "/GPU:0"
        else:
            if gpus:
                raise RuntimeError("backend_mismatch: tf-cpu launch sees a GPU")
            dev = "/CPU:0"
        # Device init happens on first use: force it here so it's part of backend_init_ms.
        with tf.device(dev):
            _ = (tf.constant([1.0]) + 1.0).numpy()
        res["backend_init_ms"] = (time.perf_counter() - t0) * 1e3
        res["tfjs"] = {"tfjs_version": f"tensorflow {tf.__version__}", "keras_version": tf.keras.__version__,
                       "cuda_build": bi.get("cuda_version"), "cudnn_build": bi.get("cudnn_version"),
                       "gpus": [str(g) for g in gpus], "backend_epsilon": tf.keras.backend.epsilon(),
                       "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled())
                       if cfg.startswith("tf-cuda") else None,
                       "op_determinism": bool(job.get("determinism"))}

        import keras_models as km
        import weights as wt
        from data import Data

        stage = "data_load"
        d0 = time.perf_counter()
        data = Data()
        res["data_fetch_ms"] = (time.perf_counter() - d0) * 1e3

        stage = "fetch"
        spec = km.model_spec(job["model"])
        res.update(family=spec["family"], size=spec["size"], params=spec["params"])
        kind = job.get("weights") or ("init" if pillar == "train" else "trained")
        res["config"]["weights"] = kind
        f0 = time.perf_counter()
        entry, arrays = wt.read_weights(job["model"], kind)
        res["fetch_ms"] = (time.perf_counter() - f0) * 1e3
        res["model_bytes"] = entry["bytes"]
        res["weights_sha256"] = entry["sha256"]

        stage = "session_create"
        s0 = time.perf_counter()
        with tf.device(dev):
            model = km.build_model(spec)
            slots = km.weight_slots(model, spec)
        s1 = time.perf_counter()
        wt.assign(slots, entry, arrays)
        s2 = time.perf_counter()
        res["build_ms"], res["weights_assign_ms"] = (s1 - s0) * 1e3, (s2 - s1) * 1e3
        res["session_create_ms"] = (s2 - s0) * 1e3
        if job.get("verify_weights"):
            bad = wt.verify(slots, entry)
            res["weights_readback"] = {"ok": not bad, "mismatches": bad[:5]}
            if bad:
                raise RuntimeError(f"weights_mismatch: read back differs for {bad[:5]}")
        variables = [v for *_, v in slots]
        eager = cfg == "tf-cuda-eager"
        resident = cfg == "tf-cuda-resident"
        opt_spec = json.loads(km.SPEC.read_text(encoding="utf-8"))["optimizer"]

        def wrap(fn):
            return fn if eager else tf.function(fn)

        # ---------- inference ----------
        def make_infer(B, rotate=True):
            pool = [data.val_batch(i, B)[0] for i in range(8 if rotate else 1)]
            fn = wrap(lambda x: model(x, training=False))
            state = {"k": 0, "x_res": None}
            if resident:
                with tf.device(dev):
                    state["x_res"] = tf.identity(tf.constant(pool[0]))

            def run():
                arr = pool[state["k"] % len(pool)]
                state["k"] += 1
                s = time.perf_counter()
                if resident:
                    out = fn(state["x_res"])
                else:
                    with tf.device(dev):
                        out = fn(tf.constant(arr))
                e = time.perf_counter()
                a = out.numpy()
                r = time.perf_counter()
                return {"t_start_epoch_ms": now_ms() - (r - s) * 1e3, "dur_ms": (r - s) * 1e3, "exec_ms": (e - s) * 1e3,
                        "read_ms": (r - e) * 1e3, "batch": B}, a
            return run, pool

        # ---------- training ----------
        def make_train(B, optimizer_impl="keras"):
            if optimizer_impl == "keras":
                opt = tf.keras.optimizers.Adam(opt_spec["learning_rate"], opt_spec["beta1"], opt_spec["beta2"],
                                               epsilon=opt_spec["epsilon"])
                apply = lambda pairs: [opt.apply_gradients(pairs)]  # noqa: E731
            else:
                opt = TfjsAdam(tf, opt_spec["learning_rate"], opt_spec["beta1"], opt_spec["beta2"], opt_spec["epsilon"])
                apply = opt.apply_gradients

            def step_fn(x, y):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(km.tfjs_cce(y, model(x, training=True)))
                grads = tape.gradient(loss, variables)
                upd = apply(list(zip(grads, variables)))
                with tf.control_dependencies(upd):
                    return tf.identity(loss)

            def fwd_fn(x, y):
                return tf.reduce_mean(km.tfjs_cce(y, model(x, training=True)))

            def fwdbwd_fn(x, y):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(km.tfjs_cce(y, model(x, training=True)))
                grads = tape.gradient(loss, variables)
                with tf.control_dependencies(grads):
                    return tf.identity(loss)

            fns = {"step": wrap(step_fn), "fwd": wrap(fwd_fn), "fwdbwd": wrap(fwdbwd_fn)}
            state = {"k": 0}

            def run(part="step"):
                p0 = time.perf_counter()
                xb, yb, epoch = data.train_batch(state["k"], B)
                step = state["k"]
                state["k"] += 1
                prep = (time.perf_counter() - p0) * 1e3
                if resident:
                    with tf.device(dev):
                        xr, yr = tf.identity(tf.constant(xb)), tf.identity(tf.constant(yb))
                    _ = xr.numpy().ravel()[:1]
                s = time.perf_counter()
                if resident:
                    loss = fns[part](xr, yr)
                else:
                    with tf.device(dev):
                        loss = fns[part](tf.constant(xb), tf.constant(yb))
                e = time.perf_counter()
                lv = float(loss.numpy())
                r = time.perf_counter()
                return {"t_start_epoch_ms": now_ms() - (r - s) * 1e3, "dur_ms": (r - s) * 1e3, "exec_ms": (e - s) * 1e3,
                        "read_ms": (r - e) * 1e3, "step": step, "epoch": int(epoch), "batch": B, "prep_ms": prep,
                        "loss": lv, "nonfinite": not np.isfinite(lv)}
            return run

        def capped_loop(run, phase, count, cap_s, cap_min, extra=None, part=None):
            # same rule as the page's runnerRuns: count runs and at least min_measure_s seconds (at most 50x
            # count), or stop at the time cap once cap_min runs are done
            t_start = time.perf_counter()
            min_s = job.get("min_measure_s") or 0
            i = 0
            while True:
                el = time.perf_counter() - t_start
                if i >= count and (el >= min_s or i >= 50 * count):
                    break
                if cap_s and i >= min(cap_min, count) and el > cap_s:
                    if i < count:
                        res["capped"].append({"phase": phase, "runs": i, **(extra or {})})
                    break
                rec = run(part) if part else run()
                rec = rec[0] if isinstance(rec, tuple) else rec
                res["runs"].append({"phase": phase, "i": i, **(extra or {}), **rec})
                i += 1

        def warm_measure(run, prefix, warm, meas, cap_s, cap_min, extra=None, part=None, measure_name=None):
            for i in range(warm):
                rec = run(part) if part else run()
                rec = rec[0] if isinstance(rec, tuple) else rec
                res["runs"].append({"phase": f"{prefix}_warmup", "i": i, **(extra or {}), **rec})
            capped_loop(run, measure_name or prefix, meas, cap_s, cap_min, extra, part)

        stage = "run"
        if mode in TRAIN_MODES:
            B = job.get("train_batch") or spec["batch"]
            res["config"]["train_batch"] = B
        if mode == "latency":
            run, _ = make_infer(job.get("infer_batch", 1))
            for i in range(job["warmup"]):
                res["runs"].append({"phase": "warmup", "i": i, **run()[0]})
            capped_loop(run, "measure", job["measure"], job.get("max_measure_s"), job.get("min_measure", 30))
        elif mode == "batch":
            vals = sorted(job.get("batch_values") or [1, 2, 4, 8, 16, 32, 64, 128, 256])
            res["config"]["batch_values"] = vals
            for b in vals:
                run, _ = make_infer(b)
                warm_measure(run, "batch", job["warmup"], job["measure"], job.get("max_measure_s"),
                             job.get("min_measure", 30), {"batch": b})
        elif mode == "correctness":
            run, pool = make_infer(job.get("correctness_batch", 32), rotate=False)
            for i in range(job["warmup"]):
                res["runs"].append({"phase": "warmup", "i": i, **run()[0]})
            _, a = run()
            res["io"] = {"inputs": [encode("images", pool[0])], "outputs": [encode("probs", a)]}
        elif mode == "train":
            run = make_train(B)
            warm_measure(run, "train", job["train_warmup"], job["train_measure"], job.get("train_max_s"),
                         job.get("train_min_measure", 20), measure_name="train_measure")
            if job.get("infer_after_train", True):
                for tag, b in (("b1", 1), ("bB", B)):
                    irun, _ = make_infer(b)
                    warm_measure(irun, f"infer_{tag}", job["infer_warmup"], job["infer_measure"], job.get("infer_max_s"),
                                 job.get("infer_min_measure", 20), {"batch": b})
        elif mode == "train_phases":
            run = make_train(B)
            for part in ("fwd", "fwdbwd", "step"):
                warm_measure(run, f"phase_{part}", job["train_warmup"], job["train_measure"], job.get("phase_max_s", 10),
                             job.get("train_min_measure", 20), part=part)
        elif mode == "train_batch":
            vals = sorted(job.get("train_batch_values") or [16, 32, 64, 128, 256])
            res["config"]["train_batch_values"] = vals
            for b in vals:
                run = make_train(b)
                warm_measure(run, "train_batch", job.get("batch_warmup", 5), job["train_measure"],
                             job.get("batch_max_s", 10), job.get("batch_min_measure", 10), {"batch": b})
        elif mode == "train_long":
            run = make_train(B)
            E = job.get("long_epochs", 3)
            per = data.n_train // B
            res["long"] = {"epochs": E, "steps_per_epoch": per, "val": []}
            evalf = wrap(lambda x, y: (tf.reduce_sum(km.tfjs_cce(y, model(x, training=False))),
                                       tf.reduce_sum(tf.cast(tf.equal(tf.argmax(model(x, training=False), -1),
                                                                      tf.argmax(y, -1)), tf.float32))))
            for e in range(E):
                for i in range(per):
                    res["runs"].append({"phase": "train_long", "i": e * per + i, **run()})
                ls, cs = 0.0, 0.0
                for st in range(0, data.n_val, B):
                    rows = np.arange(st, min(st + B, data.n_val))
                    with tf.device(dev):
                        l, c = evalf(tf.constant(data.images(data.val_x, rows)), tf.constant(data.one_hot(data.val_y, rows)))
                    ls += float(l.numpy())
                    cs += float(c.numpy())
                res["long"]["val"].append({"epoch": e, "val_loss": ls / data.n_val, "val_acc": cs / data.n_val})
        elif mode == "train_fit":
            rows = data.order[0, :data.n_train]
            x_all, y_all = data.images(data.train_x, rows), data.one_hot(data.train_y, rows)
            c0 = time.perf_counter()
            model.compile(optimizer=tf.keras.optimizers.Adam(opt_spec["learning_rate"], opt_spec["beta1"], opt_spec["beta2"],
                                                             epsilon=opt_spec["epsilon"]),
                          loss=km.tfjs_cce, run_eagerly=eager)
            res["compile_ms"] = (time.perf_counter() - c0) * 1e3
            E = job.get("fit_epochs", 2)
            steps = data.n_train // B
            for phase, epochs in (("fit_warmup", 1), ("fit_measure", E)):
                s = time.perf_counter()
                with tf.device(dev):
                    h = model.fit(x_all, y_all, batch_size=B, epochs=epochs, shuffle=False, verbose=0)
                dur = (time.perf_counter() - s) * 1e3
                res["runs"].append({"phase": phase, "i": 0, "t_start_epoch_ms": now_ms() - dur, "dur_ms": dur,
                                    "epochs": epochs, "steps": steps * epochs, "per_step_ms": dur / (steps * epochs),
                                    "batch": B, "loss": h.history["loss"][-1], "loss_history": json.dumps(h.history["loss"])})
        elif mode == "train_correctness":
            impl = job.get("optimizer_impl", "keras")
            res["config"]["optimizer_impl"] = impl
            run = make_train(B, impl)
            full = spec["params"] <= job.get("dump_params_max", 3_000_000)
            xb, yb, _ = data.train_batch(0, B)
            with tf.device(dev):
                with tf.GradientTape() as tape:
                    loss0 = tf.reduce_mean(km.tfjs_cce(tf.constant(yb), model(tf.constant(xb), training=True)))
                g0 = tape.gradient(loss0, variables)
            corr = {"step0_loss": float(loss0.numpy()), "full_dumps": full, "grads": [], "weights_after": []}
            io = {"grads": [], "weights_after": []}
            for (layer, name, _, _), g in zip(slots, g0):
                n = f"{layer}/{name}"
                ga = g.numpy()
                corr["grads"].append({"name": n, "sketch": sketch(ga, n), "sumsq": float(np.sum(ga.astype(np.float64) ** 2))})
                if full:
                    io["grads"].append(encode(n, ga))
            K = job.get("correctness_steps", 20)
            for i in range(K):
                res["runs"].append({"phase": "correctness_step", "i": i, **run()})
            for layer, name, _, v in slots:
                n = f"{layer}/{name}"
                wa = v.numpy()
                corr["weights_after"].append({"name": n, "sketch": sketch(wa, n), "sumsq": float(np.sum(wa.astype(np.float64) ** 2))})
                if full:
                    io["weights_after"].append(encode(n, wa))
            corr["steps"] = K
            res["train_correctness"] = corr
            if full:
                res["io"] = io
        elif mode == "lifecycle":
            order = job.get("lifecycle_order", "infer_first")
            res["config"]["lifecycle_order"] = order
            trun = make_train(B)
            irun, _ = make_infer(job.get("infer_batch", 1))
            di = lambda p: warm_measure(irun, p, job["infer_warmup"], job["infer_measure"], job.get("infer_max_s"),  # noqa: E731
                                        job.get("infer_min_measure", 20))
            dt = lambda p: warm_measure(trun, p, job["train_warmup"], job["train_measure"], job.get("train_max_s"),  # noqa: E731
                                        job.get("train_min_measure", 20))
            seq = [("i", "lc_infer1"), ("t", "lc_train"), ("i", "lc_infer2")] if order == "infer_first" else \
                  [("t", "lc_train1"), ("i", "lc_infer"), ("t", "lc_train2")]
            for kind_, p in seq:
                (di if kind_ == "i" else dt)(p)
        else:
            raise RuntimeError(f"config_unsupported: mode {mode} has no native implementation")

        # cuDNN use of the recurrent layers. Keras decides per layer (_could_use_gpu_kernel: tanh /
        # sigmoid, no recurrent dropout, not unrolled, bias, and for GRU reset_after=True); when eligible,
        # the runtime choice is read from a standalone layer of the same config run with return_runtime
        # (untimed, after the runs): 2 = cuDNN kernel, 1 = generic kernel on CPU, 0 = not reported.
        rnn = [(l, layer) for l, layer in zip(spec["layers"], model.layers) if l["type"] in ("lstm", "gru")]
        if rnn and cfg.startswith("tf-cuda"):
            l, layer = rnn[0]
            eligible = bool(getattr(layer, "_could_use_gpu_kernel", False))
            info = {"layer": l["type"], "cudnn_eligible": eligible}
            if eligible:
                cls = tf.keras.layers.LSTM if l["type"] == "lstm" else tf.keras.layers.GRU
                kw = {"reset_after": l["reset_after"]} if l["type"] == "gru" else {}
                probe = cls(l["units"], activation=l["activation"], recurrent_activation=l["recurrent_activation"],
                            return_runtime=True, **kw)
                with tf.device(dev):
                    _, runtime = (wrap(lambda x: probe(x)))(tf.zeros([2, 32, 96]))
                info["runtime"] = int(runtime.numpy())
            info["kernel"] = ("cudnn" if info.get("runtime") == 2 else "generic") if eligible else "generic (not eligible)"
            res["native_cudnn_rnn"] = info
        if cfg.startswith("tf-cuda"):
            mi = tf.config.experimental.get_memory_info("GPU:0")
            res["native_gpu_memory"] = {"current_bytes": mi.get("current"), "peak_bytes": mi.get("peak")}
        nonfinite = any(r.get("nonfinite") for r in res["runs"])
        res["status"] = "nonfinite_loss" if nonfinite else "ok"
        res["verification"] = {"verified": True, "detail": f"TensorFlow {tf.__version__} on {dev}"}
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        head = msg.split(": ", 2)
        cat = head[1] if len(head) > 2 and head[1].replace("_", "").isalnum() and "_" in head[1] else \
            ("op_unsupported_on_backend" if ("No OpKernel" in msg or "NotFoundError" in msg) else
             ("out_of_memory" if "ResourceExhausted" in msg or "OOM" in msg else "native_failure"))
        res["failure"] = {"stage": stage, "category": cat, "message": msg[:2000]}
    res["process_ms"] = (time.perf_counter() - T_PROCESS) * 1e3
    if job.get("result_file"):   # under nsys the worker's stdout is not forwarded
        Path(job["result_file"]).write_text(json.dumps(res), encoding="utf-8")
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
