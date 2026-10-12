# Legacy harness: why its numbers aren't used

The pre-study code (branches `main` / `Detailed-Profiling`, now under `legacy/` on `training-study`; the
`Native` branch's Python scripts are not carried over) produced one JSON per page load. Each issue below
was checked against the code (line numbers in `legacy/`) and, where stated, against the TF.js 4.22.0
sources the CDN's `@latest` resolves to. Nothing from these runs is reused in the study.

## Measurement validity

| # | Issue | Evidence | Effect |
|---|---|---|---|
| 1 | Kernel times always 0 / empty | `DNN/D1.js:35-54` read `kernel.kernelMs`; tfjs-core 4.22.0 names the field `kernelTimeMs` (`node_modules/@tensorflow/tfjs-core/dist/engine.js:554`) | `totalKernelTime` = 0 and `topKernels` = [] in every file |
| 2 | Inference never synchronised | `DNN/D1.js:69-74`: 100× `model.predict()` timed without `data()` / `dataSync()` | WebGL / WebGPU numbers are dispatch time; WASM numbers are compute time: not comparable |
| 3 | Warm-up is one unsynchronised predict | `DNN/D1.js:16-20` (also times `nextTestBatch(1)`) | No training warm-up at all; shader / pipeline compilation lands in the first `fit` step |
| 4 | WASM thread count never applied | `DNN/D1.js:108-109`: `setThreadsCount(8)` after `setBackend('wasm')`; the API "must be called before calling `tf.setBackend('wasm')`" (`tfjs-backend-wasm/dist/backend_wasm.js:421`) | WASM runs used TF.js's default thread count, not the intended 8 |
| 5 | Backend fallback undetected | `setBackend()`'s result is ignored; `logDetails.backend = tf.getBackend()` | A failed WebGPU init would be recorded under whatever backend was active |
| 6 | Training time includes rendering | `DNN/D1.js:187,219`: tfjs-vis `fitCallbacks` (onBatchEnd + onEpochEnd) | Per-batch loss readback (a GPU sync every step) plus chart rendering inside `trainingTime`; cost scales with step count |
| 7 | Inconsistent training budgets | `RNN/R4.js:244` and R5: 1 epoch; all others 10; batch 64 (D, C) vs 32 (R) | Training times aren't comparable within the RNN family |
| 8 | Data load is mostly network | `data_cifar10.js`: 139 MB sprite from GCS per page load, plus `showExamples()` rendering | "data loading time" measures the network |
| 9 | Unpinned libraries | `index.html:11,28,31`: `tfjs@latest`, unversioned wasm backend, `tfjs-backend-webgpu@latest` | Results can't be reproduced once a new release ships |
| 10 | No determinism / no repetition | `createShuffledIndices` (Math.random), `fit({shuffle: true})`, unseeded initialisers, one run per page load | No variance, no CIs, different data subset each run |
| 11 | Leaks | warm-up output, 100 inference outputs, `testBatch`, profile result never disposed | GC / pool pressure during the measured loop |
| 12 | `GPUpeakBytes` is not GPU memory | `profile.peakBytes` = TF.js engine tensor bytes incl. weights and leaked tensors | Mislabelled metric |
| 13 | CNN training can't run on WASM | `CNN/C5.js:97` (comment): `Kernel 'Conv2DBackpropFilter' not registered for backend 'wasm'` | A real TF.js coverage gap; the study records it as a failure, never works around it |
| 14 | Output file doesn't identify the run | file name = page name + timestamp | Model and backend must be inferred from the JSON body |

## Native branch (`origin/Native`)

* Docker files call `benchmark_native.py`, which doesn't exist on any branch.
* `benchmark_numba_cuda.py` defines a Numba CUDA kernel that is never called and labels a TF device
  "GPU:0 (Numba CUDA)" without checking (native-Windows TF >= 2.11 has no GPU support).
* The PyTorch port differs from the TF.js models: default PyTorch initialisers, two RNN biases, no softmax
  layer, Adam eps 1e-8 vs 1e-7, 10 epochs for R4/R5 (TF.js: 1), `drop_last=True`; it records no accuracy.

## What replaces it

One locked harness (`run_benchmark.py`, `native/run_native.py`) ported from the sibling TF.js inference
study: fresh browser per launch, explicit backend with API-counter verification, synchronised timing,
shared weight files and data subsets for TF.js and native TensorFlow, and the same 15 legacy architectures
(parameter counts match the legacy code exactly) plus LSTM, GRU and Transformer families.
