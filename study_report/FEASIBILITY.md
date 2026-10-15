# Feasibility (Phase 0)

What was checked before the harness was trusted, with the evidence. Probe numbers decide design questions;
none of them is a study result (probes run unlocked, some while other jobs used the GPU).

## Decided by source / registry inspection

| # | Question | Finding | Evidence | Consequence |
|---|---|---|---|---|
| F1 | Can ONNX Runtime Web train? | No WebGPU / WebGL training in any release. 1.30.0 (the ONNX study's pin) has no training API; `./training` (WASM only, `ort.training.wasm.min.mjs`) existed up to 1.19.2, removed in 1.20.0 | `node_modules/onnxruntime-web` 1.30.0 dist + `onnxruntime-common`; `npm view onnxruntime-web@1.19.2 / @1.20.0 exports` | ONNX Runtime Web is out of scope |
| F2 | TF.js LSTM / GRU defaults vs Keras | recurrentActivation `hardSigmoid` (Keras: `sigmoid`); GRU throws on `resetAfter: true` | `tfjs-layers/src/layers/recurrent.ts:1325,1337,1645` (4.22 source) | spec pins `sigmoid`; GRU `reset_after=False` on both sides |
| F3 | Keras cuDNN eligibility | LSTM (tanh/sigmoid, no recurrent dropout, not unrolled, bias) uses cuDNN; GRU needs `reset_after=True`, so the parity GRU runs Keras's generic kernel | `keras/layers/rnn/{lstm,gru}.py` `_could_use_gpu_kernel`; per-launch record `native_cudnn_rnn` | reported per model; GRU native numbers are "same equations", not "best native" |
| F4 | MultiHeadAttention / Einsum | tfjs-layers 4.22 doesn't export MultiHeadAttention; Einsum has no gradient (and no WASM kernel) | `tfjs-layers` dist exports; `tfjs-core/src/gradients` | attention built from matMul / softmax / transpose (both runtimes) |
| F5 | WASM training kernel coverage | `Conv2DBackpropFilter`, `UnsortedSegmentSum`, `Einsum` unregistered on wasm | kernel registries 4.22.0; confirmed in a launch: C1 wasm `Kernel 'Conv2DBackpropFilter' not registered for backend 'wasm'` | C x wasm training = recorded coverage gap (`KNOWN_GAPS`); D/R/L/G/T train on all three backends |
| F6 | WebGL default epsilon | `floatPrecision()` returns 16 whenever `WEBGL_RENDER_FLOAT32_ENABLED` is true, so `backend.epsilon()` = 1e-4 on desktop WebGL (1e-7 elsewhere) | `tfjs-backend-webgl/dist/backend_webgl.js:846-868`; measured `backend_epsilon` 1e-4 (webgl) / 1e-7 (wasm, webgpu) | Adam and loss-clip epsilon pinned to 1e-7 in the spec for every backend; the TF.js default is recorded per launch; `train_fit` keeps the library default (its loss clip uses 1e-4 on WebGL) |
| F7 | `model.fit` readbacks / yields | no per-batch readback without `onBatch*` callbacks; `yieldEvery: 'auto'` awaits requestAnimationFrame every 125 ms; `shuffle` uses Math.random | `tfjs-layers/src/engine/training.ts`, `base_callbacks.ts` | primary training loop is a custom step; `train_fit` uses `yieldEvery: 'never'`, no callbacks, pre-ordered data |
| F8 | Phase attribution without perturbation | `engine.gradients` runs forward in `tidy('forward')`, backprop in `tidy('backward')` | `tfjs-core/dist/engine.js:868,877` | kernels tagged from the scope stack; optimizer in `tidy('optimizer')` |

## Verified by gates

| Gate | Result |
|---|---|
| `scripts/check_spec.py`: both builders reproduce every spec weight (name, order, shape), parameter counts, output shape | PASS, 30/30 models; D/C/R parameter counts equal the legacy code's |
| `scripts/check_prng.py`: Python port of the sketch PRNG == src/prng.js | PASS, 100 000 draws x 4 seeds bit-identical |
| Weight files deterministic | identical sha256 on regeneration (init); trained weights generated once, sha256 in the manifest |
| Data: CIFAR-10 binary release | md5 `c32a1d4ab5d03f1284b67883e8d87530` (published); subsets regenerate bit-identically; 512 / 100 images per class; normalisation LUT sha256 checked by page and worker |
| Chrome for Testing | exe / dll sha256 equal the sibling lock (`09cac61e...`, `6dd41d3d...`) |
| Python venv | `pip freeze` identical to the sibling's locked venv |

## Probes (P0.x)

Filled in from `probes/out/` as each probe completes.

| Probe | Question | Status |
|---|---|---|
| P0.1 | every model x backend builds, loads weights bit-exactly, trains, runs inference; status ok | 90 launches (probes/out/p01; CPU shared with two other jobs, so timings are upper bounds): 83 ok; weight read-back bit-exact on 83/83; 0 leaked tensors, 0 Math.random; forwarded kernels per GPU step = 6 on every model; 5 failures = C x wasm known gap; 2 = L5, G5 on wasm-8t exceeded the 1 h launch timeout. Contended wasm step times: T5 96 s, R5 21 s, T4 19 s, L4 12 s |
| P0.2 | memory fit (WASM 4 GB heap, WebGPU limits) for the largest models | WebGPU / WebGL: every model trains. WASM, uncontended (probes/out/p02_wasm): R5 15 s/step, T4 20 s, T5 100 s; **L5 and G5 never complete a step**: the renderer reaches 4.49 GB private memory within 36 s (the 4 GB WASM memory ceiling) and then runs one core at 100 % with flat memory for 15 min, no error (probes/out/p02_l5diag, chrome_memory_samples.csv). TF.js's WASM backend hangs instead of failing when the heap is exhausted. -> KNOWN_GAPS (L5, G5) x wasm training, skipped in the training stages; inference on them is unaffected |
| P0.3 | equation parity TF.js cpu vs native tf-cpu on identical bytes; tolerances | PASS (probes/out/p03_cmp, sizes 1-2 of every family, reference = native tf-cpu, op determinism, TF.js's Adam formula). D/R/L/G/T: step-0 loss rel <= 1e-7, full-gradient rel L2 <= 2.7e-7, loss trajectory rel <= 1.5e-6 over 20 steps, weights rel L2 <= 1e-3 of the distance moved. C: gradient rel L2 2e-5 (C1) / 4e-7 (C2), consistent with float32 summation-order error over ~65k-term weight-gradient reductions; the loss deviation then grows ~2x per step (1e-7 -> 2e-3 by step 20): amplification, not an equation difference. Keras's Adam (vs TF.js's formula, same gradients) departs by > 1 % within 2-17 steps on D2/R2/C2: its bias correction folds into an effective epsilon ~30x larger early in training, so the reference uses TF.js's formula. Gradient sketches agree with the full-gradient errors once normalised by the global norm; variables whose exact gradient is zero (attention key bias: softmax ignores a per-query constant) are flagged null_grad_vars. **Tolerances fixed here** (CPU vs CPU equation gate): step-0 loss rel <= 1e-6, gradient rel L2 / sketch <= 1e-4; loss trajectories are reported (first divergence step > 1e-2), not gated |
| P0.6 | `await loss.data()` waits for the whole step | **WebGPU PASS**: a fence right after the loss read costs 0.09-0.12 ms, the same as a fence on an idle queue (0.05-0.09 ms), on D1/D3/C3/L1/L3/T1/T3; weights change every step (probes/out/syncSemantics.jsonl). **WebGL**: the comparison is invalid in that probe because fence completion itself changes regime: the first ~11 fences on an idle queue take ~5 ms, then 250-280 ms (TF.js's setTimeout-polled fence) or ~130-650 ms (raw fence polled via MessageChannel), while setTimeout(0) and MessageChannel stay fast (probes/fence_probe.py: GPU fence status reports late once the GPU has been idle). In a training loop (GPU always busy) this doesn't occur: webgl-tfence read_ms after the fence = 5.2 ms (D1, T1), the same as without the fence (probes/out/p06_fence). Since TF.js's WebGL read (backend_webgl.js read()) awaits the same fence, every WebGL data() has a ~5 ms floor from setTimeout clamping of the fence poll (D1 read 5.4 ms vs WebGPU 3.0 ms). Consequences: WebGL step / inference timings are valid; gpu_done_ms on WebGL is quantised to the poll period; low-rate open-loop inference on WebGL (load mode) may hit the idle-GPU regime: checked in the load stage, reported, not assumed |
| P0.7 | training pre-compilation | PASS (probes/out/trainPrecompile.jsonl, p07_train_precompile): a compile-only pass of one full step (throwaway optimizer, weights rewritten afterwards) leaves 0 pipelines to create in later WebGPU steps (WebGL: 1-2 programs); losses are identical step for step with and without it; T1 WebGPU first step 914 ms cold vs 23 ms after a 154 ms pass. Enabled for training (`warmup_precompile_train`) |
| P0.8 | tf.profile kernel list joins with the phase tags | PASS on D1/T1 webgpu, L1/C1 webgl, G1 wasm after logging in completion order (tf.profile records nested kernels post-order) |
| P0.10 | boost-probe workload reaches >= 1700 MHz | dev runs: WebGPU training launches reached 1777 MHz |
| P0.12 | no Math.random in measured work | 0 calls in every dev launch (checked per launch) |
| P0.14 | `fit` yields / reads | PASS for the configuration used: with yieldEvery 'never' and no callbacks, fit() makes 0 sync reads and 1 async read (history) on WebGL / WebGPU; WASM makes 25 readSync per 8 steps (heap copies, no GPU sync). The probe's rAF counter cannot observe 'auto' yields (tf.nextFrame captures requestAnimationFrame at import), so 'auto' is not characterised here |
| P0.15 | per-launch time, projection | from P0.1 and the preflight |

## Design changes forced by dev runs (before any study data)

* **Native priming launch.** The first TensorFlow process after an idle period was ~3x slower in steady
  state (L1 tf-cuda-fp32 9.5 ms vs 3.0 ms with a primed file cache), not only in its first step. The native
  driver now runs one unrecorded launch before a stage's plan; the browser stages are preceded by the boost
  probe launch.
* **Minimum measured duration.** 100 steps of small models last < 1 s, under the NVML energy counter's
  ~1 s resolution; every measured phase now also runs for >= 2 s (bounded by its time cap).
* **steps_to_steady definition.** A 5-step window median masked a single 100x cold first step and a fixed
  10 % band rejected noisy WebGL; the band is now the launch's own steady P90 with a per-record condition
  at the start (analysis/warmup.py docstring).
* **Warm-up floor and steps_to_steady horizon (preflight).** See VERIFICATION.md "Revisions forced by the
  preflight": every timed warm-up runs >= 1 s after the cold first record (browser and native), and the
  steps_to_steady window condition covers the next 10 records instead of the rest of the series.
* **Cap and batch trims not applied.** With the measured per-pair times, a 20 s instead of 30 s cap changes
  the measured window only where 200 records (inference) / 100 steps (training) take > 20 s, saving ~7 min
  of the main stages; fewer batch values (inference {1,4,16,64,256}, training {16,64,256}) would save
  ~25-30 min. Validating either would cost more run time than it saves, so both are kept at full size.
* **GPU traces for every model (requested before the final run).** The Nsight matrix (`gpu_trace`) now covers all
  30 models instead of the 6-model subset: WebGPU and WebGL (inference at batch 1 and training steps), native
  tf-cuda-fp32 (inference, training) and tf-cuda-eager (training), 2 repeats each: 420 captures (~36 GB).
