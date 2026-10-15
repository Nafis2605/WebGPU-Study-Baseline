# TF.js custom-model study (warmup / inference / training): working notes

Branch `training-study`. Sibling of `..\TFJS\TFJS-WebGPU-Study` (TF.js pretrained-model inference study;
harness ported from it, never modified) and `..\ONNX-WebGPU-Study`. Same machine, Chrome build, driver,
native TF env, Nsight version, launch discipline, counters and statistics, so results are comparable.

## Hard rules
- No fabricated or estimated metrics. A value that can't be measured is left empty with a reason.
- No backend fallback. `tf.setBackend()` must resolve true and match; the API-counter signature must
  match the backend (`verifyBackend`); strict configs need forwarded_kernels == 0. Also checked per
  launch: no Math.random during the verification run, no tensor leak, finite losses.
- TF.js and native start from identical bytes: one spec (`public/spec/models.json`) drives both
  builders (`src/models/build.js`, `native/keras_models.py`); weights (`public/weights/`) and data
  (`public/data/`) are sha256-checked on both sides. Gates: `scripts/check_spec.py`, `scripts/check_prng.py`.
- Cold-start numbers of each pillar come only from that pillar's own fresh-browser launches.
- Known coverage gap (genuine): CNN training on wasm (`Conv2DBackpropFilter` unregistered). Recorded,
  never worked around.
- Never commit without asking. Never edit measurement code during a locked run.

## Layout
- `scripts/make_spec.py` -> spec; `make_init_weights.py` (numpy, seeded); `make_trained_weights.py`
  (tfgpu, 10 epochs) -> inference weights; `prepare_cifar10.py` -> fixed 5120/1000 subsets + order.u32.
- `src/`: `tfjs-setup.js` (applyConfig, gpuFence), `instrument.js` (sibling counters), `backend-hooks.js`
  (+ forward/backward/optimizer phase tags from the engine scope stack, Math.random count, kernel log),
  `layers-runner.js` (inference run; sibling runner interface), `train-runner.js` (custom-loop step:
  variableGrads + applyGradients in tidy('optimizer') + await loss.data()), `infer-modes.js`,
  `train-modes.js`, `bench.js` (runExperiment), `profile.js` (sibling + phase join), `probes.js`.
- `run_benchmark.py` (browser driver, CONFIGS/MODES), `native/run_native.py` + `native/worker.py`
  (tfgpu, one process per launch), `benchkit/` (sibling: chrome, envlock, clockguard, telemetry, results).
- `probes/run_probes.py` Phase-0 probes; `study_report/FEASIBILITY.md`, `VERIFICATION.md`, `LEGACY_ISSUES.md`.
- `legacy/` the pre-study page, untouched (see LEGACY_ISSUES.md).

## Commands
```
npm ci; npx vite build                       # bundle (runs use dist/, served by `npx vite preview` on :4185)
.venv/Scripts/python scripts/check_spec.py    # builders == spec
.venv/Scripts/python -m benchkit.envlock verify full
.venv/Scripts/python run_benchmark.py --mode train --config webgpu --filter L1 --trials 1
.venv/Scripts/python native/run_native.py --mode train --config tf-cuda-fp32 --filter L1 --trials 1
```
Rebuild (`npx vite build`) after editing `src/`. `TRAINBENCH_ALLOW_DIRTY=1` only for development runs.

## Gotchas
- Heredoc'd Python in the Bash tool turns backslash escapes into real characters: write edit scripts
  with the Write tool (scratchpad) instead.
- tfjs-layers: LSTM/GRU recurrentActivation defaults to hardSigmoid (spec pins sigmoid); GRU has no
  reset_after=true (Keras GRU then can't use cuDNN: native_cudnn_rnn records it).
- TF.js Adam creates scalar tensors per variable per step (CPU-forwarded, counted by the hooks).
- WebGPU dataSync goes through a canvas path (sibling finding): runners use data().
- `preflight.py` has no argparse: any invocation (even `--help`) starts a preflight. Read its docstring instead.
- Timed warm-ups run >= W records and >= 1 s after the first record (`--warmup-min-s`, page and native):
  fast models get hundreds of warm-up records, so steps_to_steady uses a 10-record horizon (analysis/warmup.py).
