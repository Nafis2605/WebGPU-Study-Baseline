# Metric coverage

Every metric of the pretrained-model inference studies' list, mapped to where this study records it.
"same" = identical name and definition in the inference pillar; "+train" = also recorded per training step
(training phases in runs.csv: train_warmup / train_measure / infer_b1 / infer_bB / ...). Columns are in the
stage folders' sessions.csv / runs.csv / kernels.csv unless an analysis file is named.

| Group | Metrics | Inference | Training | Where |
|---|---|---|---|---|
| Per launch | backend_init_ms, fetch_ms, model_bytes, session_create_ms, precompile_ms, precompile_programs_linked, precompile_pipelines_created, first_run_ms, second_run_ms, n_measure, measure_capped, capped_phases, status, failure_stage / category / message, backend_verified, verification_detail, env_lock_sha256, gpu_sm_clock_max_mhz, low_clock_reruns, webgpu_submits / dispatches / dispatches_per_submit / bind_groups_created, webgl_draws, kernels_run, forwarded_kernels, forwarded_by_kernel, sync_reads, sync_read_ms, sync_reads_by_kernel, async_reads, webgpu_timestamp_query, webgl_timer_reliable, page_memory_*, profile_phases, profile_time_skipped, browser_version, tfjs_version, flags_effective, wasm_threads | same | same + first_step_ms, second_step_ms, build_ms, weights_assign_ms, weights_verify_ms, data_fetch_ms, compile_ms, kernels_by_phase, forwarded_by_phase, kernel_cpu_ms_by_phase, tensors_leaked, math_random_calls, backend_epsilon, num_tensors_first/last_step | sessions.csv; session_create_ms = build_ms + weights_assign_ms (fetch_ms = weight file); time_to_first_result_ms in analysis/warmup.py |
| Per inference | dur_ms, exec_ms, read_ms, t_start_epoch_ms, phase, batch, shape | same | +train: step, epoch, prep_ms, loss, nonfinite, num_tensors, gpu_done_ms (fence policy), window_steps (pipelined) | runs.csv |
| WebGPU / WebGL API counters, TF.js hooks per record | all listed | same (`--instrument-runs`) | +train, plus hooks_kernels_/kernel_cpu_ms_/forwarded_<forward/backward/optimizer/other> | runs.csv of instrumented stages |
| Profile summary, per kernel | all listed | same (profile) | +train (train_profile): kernels carry train_phase; analysis/train_phase.py kernel_phase.csv | analysis/profile_summary.py --mode, kernels.csv |
| Latency statistics | mean..p99, iqr, cov, tail ratio, p99_reliable, throughput, p50/p99 CIs, trial medians, between-trial CoV, ICC, drift, drift_ratio, drifting_trials, speedup + CI | same | same on step times (--series train), + per_sample_p50_ms, samples_per_s; browser vs native pairs | analysis/summarize.py |
| Batch | p50, p95, per_sample_p50, throughput_samples_per_s, amortisation | same (1..256) | same over training batch 16..256 | summarize.py *_batch_summary.csv |
| Shapes | first_seen / revisit medians and ratios, shape memory | same (batch dimension: the only free input dimension) | n/a: training shapes are fixed (one batch size per launch; the batch sweep covers new shapes) | analysis/init_reuse.py |
| GPU telemetry (samples, per window) | all listed | same | + energy_per_sample_mj, per training step, per post-training inference window | analysis/gpu_telemetry.py |
| System | cpu_percent, Chrome rss / private / cpu_user / cpu_system / processes, cpu_ms_per_inference, cores_busy | same | same per step | telemetry CSVs; analysis/cost.py |
| Memory | all listed | same (profile) | same (train_profile) + optimizer_state_mb_computed (labelled computed), tensor growth / leaks | analysis/memory.py |
| Data movement | all listed | same (profile + gpuio / fence / sync configs) | upload / readback bytes per step (instrumented), sync policies (tfence, pipelined), resident data I/O | analysis/data_movement.py, train_phase.py hook_phase.csv |
| Readback micro | all listed | same | n/a (one 4-byte loss read per step, measured directly as read_ms) | run_readback_micro.py |
| Correctness | max_abs_err .. inputs_identical | same (vs TF.js cpu and native tf-cpu) | redefined: step-0 loss, gradient sketches / full gradients, loss trajectory, first divergence step, weights after K steps (vs native deterministic tf-cpu with TF.js's Adam formula) | benchkit/correctness.py; analysis/train_correctness.py |
| Responsiveness | all listed | same | same while training (train_responsiveness) | analysis/responsiveness.py |
| Load | all listed | same | n/a: training is a closed-loop job, it has no request arrivals | analysis/load.py |
| Nsight GPU traces | all listed | same (subset) | per training step (workloads per step, busy fraction, idle gaps); native CUDA kernels per step | nsight/, analysis/nsys_summary.py |
| Native TensorFlow | native_tf_cuda_init / load_build / first_run / ttfr, gpu memory peak, tf32, placement, p50..p99, per_sample, amortisation, norm_max_err, cosine, top1 | same (Keras on the study weights) | + native_cudnn_rnn, trace / first_step, eager vs graph | native/run_native.py; analysis/native_compare.py |
| Model features | macs, gmacs, mac_kernels, weight_bytes, kernels | same | + training MACs per phase incl. Conv2DBackpropFilter / Input | analysis/features.py |

Not measurable on this host (left empty with a reason): CPU energy of WASM runs (no RAPL access);
per-process GPU memory (WDDM). Per-dispatch GPU time of WebGL kernels uses the harness's draw timers
(tf.profile's WebGL times are unreliable per kernel, sibling finding).
