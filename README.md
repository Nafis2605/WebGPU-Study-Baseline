# In-browser warm-up, inference and training of custom models: TensorFlow.js on WASM, WebGL and WebGPU

A benchmark harness for 30 code-defined models (MLP, CNN, SimpleRNN, LSTM, GRU, Transformer; five sizes each) that are
built, run and trained on CIFAR-10 inside a browser with TensorFlow.js 4.22.0, and compared with native TensorFlow 2.10.1
on the same GPU. One launch is one fresh browser process with an empty profile; the harness records timings, graphics-API
counters, kernel logs, memory, GPU telemetry and numerical agreement, and refuses to report a run whose backend, tensors
or numbers fail a check. The same code drives Chrome and Firefox, and results are stored per device and browser so runs
from different machines never overlap. `legacy/` keeps the earlier page-based experiment that this harness replaces.

## What it measures

| Pillar | Quantities | Weights |
|---|---|---|
| Warm-up | backend init, weight fetch, model build, first and second run or step, steps to steady state, effect of pre-compilation, shader disk cache and inference/training order | each pillar's own cold launches |
| Inference | latency distribution, batch scaling, shape reuse, API counters, kernels, memory, data movement, responsiveness, open-loop load, correctness | trained (10 epochs, native TF) |
| Training | step time (custom loop and `model.fit`), forward / backward / optimizer split, sync policy, data I/O, batch scaling, memory, loss / gradient / weight agreement with native | initial (seeded) |

Backends (`run_benchmark.py` `CONFIGS`): `wasm-1t|4t|8t`, `webgl`, `webgpu`, `cpu` (TF.js reference), plus variants that
isolate one factor (`*-strict` no CPU forwarding, `webgpu-sb1|4|64|512` submit batch, `*-precompile`, `*-diskcache`,
`*-gpuio`, `*-fence`, `*-sync`, `*-tfence`, `*-resident`). Native configs: `tf-cuda`, `tf-cuda-fp32`, `tf-cuda-eager`,
`tf-cuda-resident`, `tf-cpu`.

## Models (`public/spec/models.json`, one spec for both runtimes)

| Family | Parameters (size 1 to 5) | Batch | Architecture |
|---|---|---|---|
| D (MLP) | 197,322 to 7,355,402 | 64 | Dense ReLU, widths 64, 128x2, 256x3, 512x4, 1024x5 |
| C (CNN) | 600,650 to 12,673,930 | 64 | Conv64-Pool-Conv128-Pool plus the D heads |
| R (SimpleRNN) | 10,954 to 9,550,858 | 32 | 32 rows x 96 features, ReLU, widths as D |
| L (LSTM) / G (GRU) | 41,866 to 38,172,682 / 31,562 to 28,632,074 | 32 | as R; sigmoid recurrent activation pinned; GRU `reset_after=false` (the only TF.js variant) |
| T (Transformer) | 59,018 to 63,125,514 | 32 | k pre-LN encoder blocks, d = 64..1024, d/64 heads, FFN 4d; attention from matMul (Einsum has no gradient in TF.js) |

Input 32x32x3 CIFAR-10, 10 classes, class-balanced subsets of 5,120 training and 1,000 validation images in a fixed order.
Adam (lr 1e-3, beta 0.9 / 0.999, epsilon 1e-7), categorical cross-entropy with epsilon 1e-7 on every backend (the TF.js
WebGL default of 1e-4 is overridden). Both builders (`src/models/build.js`, `native/keras_models.py`) are checked against the
spec by `scripts/check_spec.py`; weights and data are sha256-checked on both sides.

## Stack

| Layer | Pinned version |
|---|---|
| Browser runtime | TensorFlow.js 4.22.0 (`tfjs-core`, `-layers`, `-backend-cpu|wasm|webgl|webgpu`, exact npm pins, one deduplicated core) |
| Bundler / server | Vite 6.4.3, Node 24.18.0, npm 11.16.0; page served with COOP/COEP headers (cross-origin isolation for WASM threads and fine timers) |
| Browser drivers | Chrome for Testing 157.0.8081.0 via Playwright 1.63.0 (DevTools protocol); Firefox 158.0 via geckodriver 0.37.1 and Selenium 4.51.0 (WebDriver BiDi) |
| Python | 3.12.10 venv (`requirements.lock`): pandas, numpy, scipy, matplotlib, psutil, nvidia-ml-py |
| Native reference | TensorFlow 2.10.1, CUDA 11.2.2, cuDNN 8.1.0 in a conda environment (`native/requirements-tfgpu.lock`, `native/conda-tfgpu-explicit.txt`, `benchkit/paths.py:TFGPU`) |
| Telemetry / traces | NVML (clocks, power, temperature, PCIe), Nsight Systems 2026.5.1 (optional) |

## Requirements on the device

Tested configuration (not a stated minimum): Windows 11, Intel Core i7-12700K (20 logical cores), 31.7 GB RAM, NVIDIA
GeForce RTX 3060 12 GB, driver 610.60.

- A browser build with WebGPU, WebGL 2 and WASM SIMD / threads, pinned in `tools/` (see Setup); no auto-update.
- Node 24.x and Python 3.12 with the locked packages.
- For the full harness as written: an NVIDIA GPU with `nvidia-smi` (clock, temperature and idle/boost guards read NVML) and
  Windows (PowerShell calls for host info). Native stages additionally need the conda TensorFlow environment; GPU traces need
  Nsight Systems. The browser stages can run without the native and Nsight parts.
- Disk: about 63 GB for one complete Chrome flight (mostly telemetry CSVs), 1.6 GB for the generated weights, 163 MB for CIFAR-10.
- Time: the complete Chrome flight (43 stages, native included) took 30.4 h of stage time and the Firefox flight (26 browser stages) 26.4 h on the tested machine; a reduced
  verification flight (`--flight verify`) took 7.2 h.
- Other operating systems and mobile browsers are not wired up yet. The results layout and manifests do not depend on the
  device; the drivers and guards listed above do.

## Repository layout

```
src/ public/ index.html vite.config.js    page harness (modes, backends, counters), model spec, weight and data manifests
run_benchmark.py run_readback_micro.py    browser drivers (one launch per experiment)
run_study.py preflight.py                 stage orchestrator (flights) and GO / NO-GO check
native/ nsight/                           native TensorFlow driver and worker; Nsight Systems captures
benchkit/                                 shared Python: layout, environment lock, Chrome / Firefox launchers, telemetry, results
analysis/                                 statistics and cross-stage / cross-browser comparison
probes/ scripts/                          feasibility and capability probes; spec, weight and data generation and gates
legacy/                                   the earlier page-based experiment
results/   (local, git-ignored)           measurement data, one folder per device / browser / run
reports/   (local, git-ignored)           write-ups, tables, figures and notes; never pushed
```

## Setup

```
npm ci                                                       # exact lock; copies the TF.js WASM binaries
python -m venv .venv
.venv\Scripts\pip install --no-deps -r requirements.lock     # on macOS / Linux: .venv/bin/pip
# Pinned browsers (never auto-updated), unpacked into tools/:
#   tools/chrome-for-testing/chrome-win64/chrome.exe         Chrome for Testing 157.0.8081.0 (win64)
#   tools/firefox-158.0/firefox.exe                          copy of a Firefox 158.0 install, plus distribution/policies.json:
#       {"policies":{"DisableAppUpdate":true,"DisableTelemetry":true,"DisableFirefoxStudies":true,
#        "DontCheckDefaultBrowser":true,"DisableDefaultBrowserAgent":true}}
#   tools/geckodriver-0.37.1/geckodriver.exe                 mozilla/geckodriver v0.37.1 release
.venv\Scripts\python scripts\make_spec.py                    # spec from the model definitions
.venv\Scripts\python scripts\make_init_weights.py            # seeded initial weights (numpy)
# CIFAR-10 binary release (md5 c32a1d4ab5d03f1284b67883e8d87530) into downloads/cifar10/, then:
.venv\Scripts\python -I scripts\prepare_cifar10.py           # fixed 5,120 / 1,000 subsets and order
C:\tfjsbench\tfgpu\python.exe scripts\make_trained_weights.py   # inference weights, 10 epochs (native env; its Library\bin on PATH)
.venv\Scripts\python scripts\check_spec.py
.venv\Scripts\python scripts\check_prng.py                   # builders and PRNG agree with the spec
npx vite build; npx vite preview                             # serves dist/ and public/ on :4185 (keep running)
.venv\Scripts\python -m benchkit.envlock freeze              # on a committed, clean tree: records driver, browsers, packages, hashes
.venv\Scripts\python preflight.py                            # GO / NO-GO and a duration projection
```

## Running

```
.venv\Scripts\python run_benchmark.py --mode train --config webgpu --filter L1 --trials 1    # one experiment
.venv\Scripts\python native\run_native.py --mode train --config tf-cuda-fp32 --filter L1 --trials 1
.venv\Scripts\python run_study.py --resume                                   # complete flight, current browser (chrome)
.venv\Scripts\python run_study.py --browser firefox --resume                 # complete browser flight in Firefox
.venv\Scripts\python run_study.py --browser firefox --flight verify --resume # reduced replication flight
.venv\Scripts\python run_study.py --stages train native_train --resume       # chosen stages
```

`run_benchmark.py --mode` selects the question (`latency`, `shapes`, `batch`, `profile`, `correctness`, `load`,
`responsiveness`, `train`, `train_pipelined`, `train_profile`, `train_correctness`, `train_batch`, `train_long`, `train_fit`,
`train_phases`, `train_responsiveness`, `lifecycle`); `run_study.py` lists the stage commands, caps and trial counts in
`STAGES` and the stage groups in its docstring. A multi-hour flight should be launched outside the interactive shell so that
an application update cannot stop it; `--resume` re-runs only unfinished stages. `TRAINBENCH_ALLOW_DIRTY=1` skips the
clean-tree check of the environment lock and is for development only.

### Results layout

```
results/INDEX.json, INDEX.csv                         catalogue of all runs (python scripts\index_results.py)
results/<device>/device.json                          OS, CPU, RAM, GPU, driver, power plan
results/<device>/<browser>-<version>/browser.json     pinned build, flags or prefs
results/<device>/<browser>-<version>/<run>/run.json   kind, dates, harness commit and patch hash, stage statuses
    <run> = full | verify | diagnostics | adhoc/<timestamp> | _dev/<name>
    <run>/<stage>/<timestamp>/  env.json  experiments.jsonl  sessions.csv  runs.csv  kernels.csv  telemetry CSVs  logs/  io/
    <run>/analysis/             cross-stage tables;  <run>/env.lock.json  the lock the run was verified against
results/<device>/native/<run>/  results/<device>/probes/  results/<device>/_ops/
```

The device id comes from `device.local.json` (created on first run, e.g. `win11-rtx3060`; edit it to rename) or `--device-id`.
To add another device, check out the repository there, install the pinned browser into `tools/`, and run the commands above;
its folder `results/<device>/` can be copied next to the others without collisions. A new browser needs a launcher in
`benchkit/` that yields a page with `evaluate`, plus an entry in `benchkit/layout.py:browser_id`.

## Another Windows device

Steps that the Setup section does not cover, found on an RTX 4090 laptop with an Intel iGPU (device `win11-rtx4090`):

1. Tools: Git, Node 24.18.0 (`winget install OpenJS.NodeJS.LTS --version 24.18.0`), Python 3.12.10, and Chrome for Testing
   157.0.8081.0 unpacked into `tools/chrome-for-testing/`. For the native stages install Miniconda and create the
   environment from conda-forge only (this avoids the Anaconda channel terms prompt):
   `conda create -p C:\tfjsbench\tfgpu --file native\conda-tfgpu-explicit.txt --override-channels -c conda-forge`, then
   `pip install --no-deps -r` a copy of `native\requirements-tfgpu.lock` without its `pip @ file:///...` line.
2. Weights and data: copy `public/weights/*.bin` and `public/data/*.u8|*.u32` from the first machine. They are checked against
   the tracked manifests; regenerating them on another GPU gives different bytes.
3. Hybrid GPU: set the Windows graphics preference of `tools\chrome-for-testing\chrome-win64\chrome.exe` to High performance
   (Settings, System, Display, Graphics). `benchkit/gpucheck.py` fails any WebGL or WebGPU launch whose adapter is not
   NVIDIA (`TRAINBENCH_REQUIRE_GPU_VENDOR=any` turns it off).
4. Nsight Systems: install the version in `benchkit/paths.py` (found automatically) and, in NVIDIA Control Panel,
   Developer, Manage GPU Performance Counters, allow access for all users; otherwise every capture fails with
   `ERR_NVGPUCTRPERM`.
5. Guards: measure the new GPU's idle clock, boost peak and temperatures, then copy `trainbench.env.example` to
   `trainbench.env` and set `TRAINBENCH_IDLE_MAX_MHZ`, `_BOOST_MHZ`, `_COOL_TO_C`, `_COOL_TIMEOUT_S`. The first machine's values
   (1000 MHz, 1700 MHz, 60 C, 180 s) cannot be met on a laptop that idles at 1455 MHz and 70 C.
6. Long runs on a hybrid-GPU laptop: start `run_study.py` from a one-shot interactive Scheduled Task, not from WMI.

   ```
   Register-ScheduledTask -TaskName StudyRun -Action (New-ScheduledTaskAction -Execute cmd.exe -Argument '/c C:\path\run.cmd') `
     -Principal (New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited)
   Start-ScheduledTask StudyRun      # delete it afterwards: Unregister-ScheduledTask StudyRun -Confirm:$false
   ```

   `run.cmd` starts `npx vite preview` and then `run_study.py`, with Node and Git on `PATH`. Check first that Chrome, started the
   same way, reports the NVIDIA adapter.
7. Telemetry: NVML may fail single queries. The sampler keeps the other fields and records the failures in
   `telemetry.field_errors`. If the energy counter and the sampled power disagree, `analysis/gpu_telemetry.py` sets
   `energy_counter_ok` to False; quote the sampled-power columns then and name the source.

## Analysing results

`results/INDEX.csv` gives one row per run (device, browser, run, launches, failures, harness commit, dirty flag, path).
`experiments.jsonl` is the raw source of truth (one JSON record per launch, with the verification block); `sessions.csv` has
one row per launch (status, failure, known gap, init and first-run times, API counters), `runs.csv` one row per timed record
(`phase`, `dur_ms`, `exec_ms`, `read_ms`, ...). Scripts in `analysis/`: `summarize.py` (per-model statistics), `warmup.py`,
`cost.py`, `memory.py`, `data_movement.py`, `responsiveness.py`, `load.py`, `train_correctness.py`, `native_compare.py`
(browser vs native), `browser_compare.py` (browser vs browser against pre-registered margins). `analysis/check_stage.py` re-checks
a stage's launch guards.

## Measurement rules

- One fresh browser or native process per (trial, model, config), shuffled within a trial (seed 1); GPU cooled to 60 C; idle
  and boost guards per stage; launches whose SM clock never exceeds 1,000 MHz are re-run.
- No fallback: `tf.setBackend()` must succeed and match, the API-counter signature must match the backend, strict configs
  forward no kernel to the CPU; every launch also checks for tensor leaks, `Math.random` use and non-finite losses.
- Timed regions: inference = host array to tensor, `model.predict`, `data()`; training step = host batch to tensors,
  `variableGrads`, `applyGradients`, `await loss.data()`. Native uses the same definitions.
- Training uses short windows (15 warm-up steps, up to 100 measured, at least 2 s); the `train_long` stage tests that they
  represent steps deep into training (TOST, +-5 %).
- A value that cannot be measured is left empty with a reason. Recorded coverage gaps are never worked around: CNN training
  on WASM (`Conv2DBackpropFilter` unregistered), and L5 / G5 training on WASM (4 GB heap exhausted).
- `env.lock.json` pins the GPU driver, browsers and drivers, Node and npm packages, Python and native environments, Nsight, the
  spec, data and weight hashes, and the git commit. It is verified before every stage and launch; a mismatch stops the run.
  Each run stores its lock and `run.json` (harness commit, and a hash of uncommitted changes when the tree was dirty).

## License and citation

No license file is included yet. Citation: [to be added with the paper].
