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
| Measured window | warm-up W, then N records, at least 2 s, at most the time cap (>= min records); capped phases flagged, never extrapolated |
| Steady state | median of the measured records; band = their P90 |
| steps_to_steady | first k with t_k <= band and every forward 5-record (full-window) median from k on <= band |
| Shortened training | TOST on median(steps of epochs 2-3) / median(short window): 90 % bootstrap CI inside [0.95, 1.05] |
| Training correctness reference | native tf-cpu, op determinism, TF.js's Adam formula (`tf-cpu+tfjsadam`), identical weights and batches |
| Energy | NVML counter over a measured window >= 1 s, idle baseline subtracted separately |
| Status ok | backend counters match, no Math.random in the verification run, no leaked tensor, finite losses, strict: no forwarded kernel |

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
