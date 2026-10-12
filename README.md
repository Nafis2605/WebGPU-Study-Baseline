# In-browser warm-up, inference and training of custom models: TF.js (WASM / WebGL / WebGPU) vs native TensorFlow (CUDA)

Branch `training-study`. A locked, verified benchmark harness for 30 code-defined models trained and run on
CIFAR-10 in Chrome with TensorFlow.js 4.22.0, compared with native TensorFlow 2.10.1 / Keras on the same
GPU. It is the sibling of the TF.js pretrained-model inference study (`TFJS-WebGPU-Study`), whose harness,
counters, launch discipline and statistics it reuses; the pre-study page is kept in `legacy/` (see
`study_report/LEGACY_ISSUES.md` for why its numbers aren't used).

## Questions (pillars)

| Pillar | What is measured | Launches |
|---|---|---|
| Warm-up | backend init, weight fetch, model build, first / second run or step, warm-up curve, steps to steady state, time to first result; why-axes: pre-compilation, GPU shader disk cache, cross-phase transfer (inference <-> training) | each pillar's own cold launches |
| Inference | latency distribution, throughput, batch scaling, shape reuse, API counters, kernels, memory, data movement, energy, responsiveness, open-loop load, correctness, Nsight | trained weights |
| Training | step time (custom loop and `model.fit`), forward / backward / optimizer split, sync policies, data I/O, batch scaling, memory and leaks, energy per step / sample, correctness of losses, gradients and weights, Nsight | initial weights |

ONNX Runtime Web is not included: no release offers training on WebGPU or WebGL (1.30.0 has no training
API; the WASM-only `onnxruntime-web/training` existed up to 1.19.2 and was removed in 1.20.0).

## Models (public/spec/models.json, one spec for both runtimes)

| Family | Sizes k = 1..5 | Notes |
|---|---|---|
| D (MLP) | Dense ReLU widths 64, 128x2, 256x3, 512x4, 1024x5 | legacy architecture, batch 64 |
| C (CNN) | Conv64-Pool-Conv128-Pool + the D heads | legacy, batch 64; training on WASM is a TF.js coverage gap (Conv2DBackpropFilter) |
| R (SimpleRNN) | 32 rows x 96 features, ReLU, widths as D | legacy, batch 32 |
| L (LSTM) / G (GRU) | same grid as R | sigmoid recurrent activation pinned; GRU reset_after=False (TF.js's only variant) |
| T (Transformer) | k pre-LN encoder blocks, d = 64..1024, heads d/64, FFN 4d | attention from matMul (Einsum has no gradient in TF.js) |

Parameter counts of D/C/R equal the legacy code's. Both builders are checked against the spec
(`scripts/check_spec.py`).

## Locked environment

Same machine and pins as the sibling study: RTX 3060 (driver 610.60), Windows 11, Chrome for Testing
157.0.8081.0 (hash-checked, no fallback), Node 24.18.0, TF.js 4.22.0 incl. tfjs-layers (exact npm pins, one
deduplicated tfjs-core), Python 3.12.10 venv (sibling's exact pip lock), native TF 2.10.1 + CUDA 11.2 +
cuDNN 8.1 (`C:\tfjsbench\tfgpu`, conda + pip locks), Nsight Systems 2026.5.1. `env.lock.json` also hashes
the spec, the data and weight manifests and every data / weight file; it is verified per stage and per launch.

## Setup

```
npm ci                                       # exact package-lock; copies the TF.js WASM binaries
python -m venv .venv && .venv\Scripts\pip install --no-deps -r requirements.lock
# Chrome for Testing 157.0.8081.0 win64 unpacked into tools/chrome-for-testing/chrome-win64/
.venv\Scripts\python scripts\make_spec.py
.venv\Scripts\python scripts\make_init_weights.py
# CIFAR-10 binary release into downloads/cifar10/ (md5 c32a1d4ab5d03f1284b67883e8d87530), then:
.venv\Scripts\python -I scripts\prepare_cifar10.py
C:\tfjsbench\tfgpu\python.exe scripts\make_trained_weights.py   # PATH incl. C:\tfjsbench\tfgpu\Library\bin
.venv\Scripts\python scripts\check_spec.py && .venv\Scripts\python scripts\check_prng.py
npx vite build && npx vite preview           # serves dist/ + public/ on :4185 with COOP/COEP
.venv\Scripts\python -m benchkit.envlock freeze   # on a committed, clean tree
.venv\Scripts\python preflight.py                 # GO / NO-GO + duration projection
.venv\Scripts\python run_study.py --output-dir benchmark_results\final --resume
```

## Measurement rules

* One fresh browser (empty profile) or native process per (trial, model, config); shuffled within a trial;
  GPU cooled to 60 C, idle and boost guards per stage, low-clock launches re-run.
* No backend fallback: `tf.setBackend()` must succeed and match, the graphics-API counter signature must
  match the backend, strict configs must forward no kernel; every launch also checks for tensor leaks,
  Math.random use and non-finite losses.
* Timed regions: inference = host array -> tensor -> `model.predict` -> `data()`; training step = host
  batch -> tensors -> `variableGrads` (forward + backward) -> `applyGradients` -> `await loss.data()`.
  Host batch assembly is outside, reported as `prep_ms`. Native uses the same definitions.
* Training uses short measured windows (15 warm-up + up to 100 steps, >= 2 s) instead of epochs; the
  `train_long` stage tests (TOST, +-5 %) that they represent steps deep into training.
* Numerics pinned for every backend: Adam / loss epsilon 1e-7 (TF.js's WebGL default would be 1e-4).

## Layout

`src/` page harness, `run_benchmark.py` browser driver, `native/` native driver + worker, `benchkit/` shared
Python (lock, Chrome, telemetry, results), `run_study.py` stage orchestrator, `preflight.py`, `nsight/`
Nsight Systems captures, `analysis/` statistics, `probes/` Phase-0 feasibility probes, `scripts/` artifact
generation and gates, `study_report/` feasibility, verification and legacy notes.
