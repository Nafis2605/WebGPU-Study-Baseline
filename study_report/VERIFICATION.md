# Verification log

One entry per metric group: the check, decided before the study data, and its evidence. Groups are
verified one at a time on D1, L1 and T1 (one MLP, one recurrent, one attention model) before the next is
trusted. Status: PASS / FAIL (with what changed) / PENDING.

## 0. Pre-registered definitions (fixed before any study data)

| Item | Definition |
|---|---|
| Inference record | host Float32Array -> `tf.tensor` -> `model.predict(x, {batchSize: B})` -> `await out.data()`; `exec_ms` until predict returned, `read_ms` = data() |
| Training step | host batch prepared outside (`prep_ms`); timed: tensors -> `tf.variableGrads` (forward + loss + backward) -> `applyGradients` in `tidy('optimizer')` -> `await loss.data()` |
| Native step | same order; `tf.function` GradientTape step returning the loss under a control dependency on the update; `.numpy()` sync |
| Warm-up | >= W records and >= 1 s after the first (cold) record (`--warmup-min-s`, browser and native), at most `--warmup-max-s` 30 s once >= 3 records exist; every warm-up record is kept (cold-start curve) |
| Measured window | then N records, at least 2 s, at most the time cap (>= min records); capped phases flagged, never extrapolated |
| Steady state | median s of the measured records; band = max(their P90, 1.02 s) |
| steps_to_steady | first k with t_k <= band and the forward 5-record (full-window) medians at positions k .. k+5 <= band (the next 10 records stay in the band) |
| Warm-up sufficient | k <= number of warm-up records; assessed with >= 20 measured records (else not assessable) |
| measured_drift | median(first 20 % of measured records) / median(rest), 90 % bootstrap CI; flagged when the CI lies above 1.02; diagnostic, reported next to the latency |
| Shortened training | TOST on median(steps of epochs 2-3) / median(short window): 90 % bootstrap CI inside [0.95, 1.05] |
| Training correctness reference | native tf-cpu, op determinism, TF.js's Adam formula (`tf-cpu+tfjsadam`), identical weights and batches |
| Energy | NVML counter over a measured window >= 1 s, idle baseline subtracted separately |
| Status ok | backend counters match, no Math.random in the verification run, no leaked tensor, finite losses, strict: no forwarded kernel |

### Revisions forced by the preflight (2026-10-12, before any study data)

The preflight (smoke runs of every stage) and full-setting re-runs of the pairs it flagged showed:

* **steps_to_steady.** The earlier condition ("every forward window from k to the end of the series") fails
  on long steady series: isolated bursts of >= 3 records above the P90 band occur every few hundred
  records, so 700-record launches of D1 / D3 on WebGPU reported k = 168-707 for series steady from record
  1-2. The window condition now covers the next 10 records. On the same data k = 1-8 (C3 wasm 30-36).
* **Warm-up length.** 10 warm-up records of a 0.3-3 ms inference last 3-30 ms. The first 0.2-0.3 s of the
  measured window was then 3-10 % slower than the rest (C3 wasm 9 %, G2 webgl 5-6 %, D3 webgpu 3.5 %; per-
  decile medians flat afterwards), i.e. residual warm-up. Every timed warm-up now also lasts >= 1 s after
  the first record. Training warm-ups (15 steps incl. compilation) already exceeded 1 s on 71 / 83 pairs;
  the training re-runs showed k <= W and drift within 1.3 %.
* **Smoke band.** A P90 band of 5-10 measured records is about their maximum, so the smoke runs flagged
  9 launches by exactly one record (k = W + 1). Sufficiency is now assessed only with >= 20 measured records.
* **Minimum band width.** On a very quiet series (native L4 tf-cpu training, ~1 % spread) the P90 band lay
  1.2 % above the median, so warm-up steps 1 % slower than the measured window counted as "not steady"
  (k = W + 1 while the series was flat from step 2). The band is now at least 2 % above the median (the
  same margin as the drift flag).
* **Batch order past 10 epochs.** order.u32 holds 10 epochs (800 steps at B = 64, 200 at B = 256). Fast
  pairs exceed that at full settings (native D1: ~1.6 ms per step, 2 s minimum window), which the smoke
  runs (no minimum window) never reached; such launches failed with "beyond the epochs in order.u32".
  Steps past 10 epochs now reuse the orders cyclically (order[epoch mod 10]), identically in src/data.js and
  native/data.py (rows compared at steps 0 - 33333, B = 16 - 256: identical).

Validation of these revisions (development runs on the uncommitted tree, not study data):
full settings, 7 inference models x 3 browser backends + 2 native configs and 6 training models x the same,
2 trials, plus the native D1 / D3 training pairs after the order fix (126 timing launches): warm-up
sufficient in 126 / 126 (steps_to_steady median 3, p95 27); the measured-window drift of the pairs that
showed it fell from 1.03-1.09 to 0.95-1.02 (G2 webgl 1.05-1.06 -> 1.01); the remaining drift flags (4) are
sub-0.1 ms WASM inferences at the 5 us timer resolution. Smoke of the 17 warm-up-using stages over all 30
models: 276 / 276 assessable launches sufficient (17 slow pairs not assessable: < 20 records in 10 s);
after the order fix the failed stages re-ran 109 / 109 ok; every analysis ran without a traceback.

## 1. Environment and inputs

| Check | Status | Evidence |
|---|---|---|
| Lock collects the sibling's exact Chrome / venv / tfgpu hashes | PASS | `python -m benchkit.envlock show`: chrome exe `09cac61e...`, pip freeze `96bf7eee...`, tfgpu pip `4ecf98ad...`, conda `b73825dc...` = sibling `env.lock.json` |
| Single tfjs-core, layers shares the engine | PASS | `npm ls` (deduped); preflight static gate |
| Builders == spec | PASS | `scripts/check_spec.py` 30/30 |
| Weights round-trip bit-exactly | PENDING (all models) | `--verify-weights` in P0.1: per-tensor sha256 read back from the backend |
| Data bytes identical in page and worker | PASS | both check every file's sha256 and the LUT sha256 against `public/data/manifest.json` |

## 2-n. Metric groups

Filled in as each group is verified (init / warm-up, latency and step time, API counters, hooks and phase
split, profile kernels, memory and leaks, data movement, energy, responsiveness, load, correctness,
native, Nsight, features).
