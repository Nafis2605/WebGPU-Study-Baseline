# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `probes/run_probes.py` Phase-0 probes, `browser_caps.py`, `fit_device_diag.py`; `benchkit/firefox.py` (Firefox launcher over
  WebDriver BiDi), `benchkit/layout.py` (where results and reports go), `analysis/browser_compare.py` (Firefox vs Chrome),
  `scripts/index_results.py` (results catalogue).
- `legacy/` the pre-study page, untouched.
- Local, git-ignored, never pushed: `results/<device>/<browser>-<version>/<run>/` (measurements; `results/INDEX.json` lists
  every run), `reports/` (`methodology/` feasibility, verification, metrics map, legacy notes; `<device>/<browser>/` analyses,
  tables, figures, report scripts; `<device>/compare/` margins and the Firefox verification plan), `device.local.json`
  (this machine's id), `tools/` (pinned browsers), `ACM_Web_2027_WebGPU/` (paper; its `.bib` is read-only).
- Only `README.md` and `CLAUDE.md` are tracked markdown; other notes go under `reports/`.

## Platform
Commands and paths below are Windows (`.venv/Scripts`, `C:\tfjsbench\tfgpu`, WMI launch, NVML/`nvidia-smi` guards, PowerShell
host info). macOS (Apple silicon, branch `macos-port`): paths in `benchkit/paths.py` (`.venv/bin`, `~/tfjsbench/tfmac`),
Chrome for Testing mac-arm64 in `tools/chrome-for-testing/chrome-mac-arm64/`, NVML replaced by `benchkit/macgpu.py` (ioreg
utilisation + memory, power-state guard: AC, Low Power Mode off, no thermal warning; no clocks/temperature/power without
root), native `tf-cuda-fp32` -> `tf-metal`, other tf-cuda configs and `gpu_trace` (Nsight) unavailable. Mac vs Windows
differs in OS and hardware at once: report differences as platform effects. Models, stack and measurement rules: README.md.

## Architecture (big picture)
- Chain: `run_study.py` (flights, `STAGES`, `--resume`, per-stage `stage_done.json`) -> `run_benchmark.py` or
  `native/run_native.py` (one fresh process per trial x model x config) -> page `src/main.js` -> `bench.js` `runExperiment`
  -> mode in `infer-modes.js` / `train-modes.js`. `preflight.py` runs the whole pipeline at minimal counts for a GO / NO-GO.
- Guards around every stage and launch: `benchkit/envlock` (`env.lock.json`, clean committed tree), GPU idle / boost (NVML),
  re-run of low-clock launches, and a per-launch verification block in `experiments.jsonl`.
- Output: `results/<device>/<browser>-<ver>/<run>/<stage>/<ts>/`; `experiments.jsonl` is the source of truth, `analysis/`
  reads it. Device id comes from `device.local.json`.
- No unit tests or linter. "Tests" are the gates (`check_spec.py`, `check_prng.py`, `analysis/check_stage.py`, `preflight.py`)
  and one-launch runs (`--filter <model> --trials 1`). Modes and configs: `MODES` / `CONFIGS` in `run_benchmark.py`.

## Commands
```
npm ci; npx vite build                       # bundle (runs use dist/, served by `npx vite preview` on :4185)
.venv/Scripts/python scripts/check_spec.py    # builders == spec
.venv/Scripts/python -m benchkit.envlock verify full
.venv/Scripts/python run_benchmark.py --mode train --config webgpu --filter L1 --trials 1
.venv/Scripts/python native/run_native.py --mode train --config tf-cuda-fp32 --filter L1 --trials 1
.venv/Scripts/python run_study.py --browser firefox --flight verify --resume   # -> results/<device>/firefox-<ver>/verify
.venv/Scripts/python scripts/index_results.py                                  # refresh results/INDEX.json
```
`npm run bench:serve` = build + preview on :4185 (COOP/COEP headers required; `public/` served uncached);
`npm run build` also emits unminified `dist-diag` (:4186, `--mode diag`) for kernel names. `postinstall` copies the TF.js WASM
binaries into `public/tfjs-wasm/`.
Rebuild (`npx vite build`) after editing `src/`. `TRAINBENCH_ALLOW_DIRTY=1` only for development runs.
Launch multi-hour runs outside the app's process tree (WMI `Win32_Process.Create`); a resumed run keeps its first harness
record in `run.json` and lists later launches under `resumes`.

## Gotchas
- `.claude/` is git-ignored, so `.claude/skills/writing-style/SKILL.md` may be absent in a checkout; ask the user for it.
- `public/data/*.u8|u32` and `public/weights/*.bin` are git-ignored; regenerate with the scripts, hashes live in the manifests.
- Heredoc'd Python in the Bash tool turns backslash escapes into real characters: write edit scripts
  with the Write tool (scratchpad) instead.
- PowerShell: `mv` is an alias of `Move-Item` and wins over a function named `Mv`; name helper functions differently.
- tfjs-layers: LSTM/GRU recurrentActivation defaults to hardSigmoid (spec pins sigmoid); GRU has no
  reset_after=true (Keras GRU then can't use cuDNN: native_cudnn_rnn records it).
- TF.js Adam creates scalar tensors per variable per step (CPU-forwarded, counted by the hooks).
- WebGPU dataSync goes through a canvas path (sibling finding): runners use data().
- `preflight.py` has no argparse: any invocation (even `--help`) starts a preflight. Read its docstring instead.
- Timed warm-ups run >= W records and >= 1 s after the first record (`--warmup-min-s`, page and native):
  fast models get hundreds of warm-up records, so steps_to_steady uses a 10-record horizon (analysis/warmup.py).

## Writing and bibliography (always active)
- Before drafting or revising any paper, report, abstract, caption, slide, email or documentation text, load the
  `writing-style` skill (`.claude/skills/writing-style/SKILL.md`, from `Desktop/Permanent Anti-AI-Slop Writing
  Standard.md`) and apply it. It stays active unless the user overrides a specific rule for a specific task.
- Never change, manipulate, reformat or add to any `.bib` file (e.g. `ACM_Web_2027_WebGPU/refs.bib`). Read-only. Cite
  only keys that already exist; for a missing source write `[citation needed]` in the text and tell the user. Report
  an apparent error in an entry instead of fixing it.
- Every number in a report or paper comes from a measured result, a computed table or a cited source; label each
  mechanism as measured, source-backed or hypothesis.
