"""
Run the complete study on the locked environment: every browser stage (run_benchmark.py), the native
TensorFlow stages (native/run_native.py), the readback micro-benchmark, the Nsight Systems traces, then the
analyses. One output folder, one environment. Port of the sibling TF.js study's run_study.py.

Pillars and their stages (cold-start numbers of a pillar come only from that pillar's own launches):
  inference  infer_latency, infer_native, infer_native_why, infer_correctness, infer_native_correctness,
             infer_batch, infer_native_batch, infer_shapes, infer_why, infer_instrumented, infer_profile,
             infer_responsiveness, infer_load, readback_micro
  warm-up    warmup_precompile, warmup_diskcache, warmup_lifecycle_if, warmup_lifecycle_tf
  training   train, native_train, native_train_why, train_instrumented, train_profile, train_correctness,
             train_correctness_cpu, native_train_correctness*, train_long, native_train_long, train_batch,
             native_train_batch, train_why, train_sync_io, train_pipelined, train_phases, native_train_phases,
             train_fit, native_train_fit, train_responsiveness
  traces     gpu_trace (nsight/capture_matrix.py)

Before every stage (as in the sibling study):
  - the environment lock is verified in full (driver, browser build, packages, artifact hashes, clean
    committed code); any mismatch stops the study
  - the GPU must idle normally (idle SM clock <= 1000 MHz) and must boost: a short WebGPU training run
    has to reach >= BOOST_MHZ (the boost probe; retried for up to 15 min). It also warms the browser's
    binaries in the OS file cache, so a stage's first launch isn't slower than the rest (the native
    driver does the same with its own unrecorded priming launch).
Inside every stage, GPU launches whose SM clock never exceeds 1000 MHz are re-run automatically.

Each stage writes <study>/<stage>/ (+ <stage>.log) and, when finished, <stage>/stage_done.json.
--resume skips finished stages. Cross-stage analyses write <study>/analysis/.

Usage:
    python run_study.py --resume                                  # -> results/<device>/chrome-<ver>/full
    python run_study.py --browser firefox --stages train --resume  # -> results/<device>/firefox-<ver>/full
    python run_study.py --browser firefox --flight verify --resume # -> results/<device>/firefox-<ver>/verify
    python run_study.py --output-dir <study folder> --analyze-only
(<device> comes from device.local.json or --device-id; folder naming is documented in benchkit/layout.py.)

--browser firefox runs the browser stages in the pinned Firefox (benchkit/firefox.py). Native stages don't
depend on the browser and are refused there: the comparison uses the Chrome flight's native results. The boost
probe stays a Chrome WebGPU D5 run, identical to the Chrome flight's, so it doubles as a drift sentinel; each
Firefox stage then starts with one unrecorded Firefox launch that warms Firefox's binaries in the OS file cache.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchkit import envlock  # noqa: E402
from benchkit.clockguard import ClockSampler, idle_check  # noqa: E402
from benchkit.paths import resolve_result_dir  # noqa: E402
from benchkit import layout  # noqa: E402

PRIMARY = ["wasm-8t", "webgl", "webgpu"]
SUBSET = ["D3", "C3", "R3", "L3", "G3", "T3"]
SIZES_135 = [f"{f}{k}" for f in "DCRLGT" for k in (1, 3, 5)]
SIZES_12 = [f"{f}{k}" for f in "DCRLGT" for k in (1, 2)]
COOL_TO_C = "60"   # the RTX 3060 idles at 53-57 C (fans stop below ~60 C); 60 = back to idle (sibling)
BOOST_MHZ = 1700
IS_MAC = sys.platform == "darwin"
# macOS (Apple silicon): no CUDA. tf-cuda-fp32 (FP32 GPU parity config) -> tf-metal, tf-cpu stays tf-cpu
# (tensorflow-macos); tf-cuda (TF32), tf-cuda-eager and tf-cuda-resident have no counterpart and are dropped,
# which the stage summary records.
MAC_NATIVE = {"tf-cuda-fp32": "tf-metal", "tf-cpu": "tf-cpu"}
INFER_CAP = ["--warmup", "10", "--measure", "200", "--max-measure-s", "30", "--min-measure", "30"]
TRAIN_CAP = ["--train-warmup", "15", "--train-measure", "100", "--train-max-s", "30", "--train-min-measure", "20"]
POST_INFER = ["--infer-warmup", "5", "--infer-measure", "50", "--infer-max-s", "10"]
F = lambda *m: ["--filter", *m]   # noqa: E731
RB, RN = "run_benchmark.py", "native/run_native.py"

# stage -> (script, arguments, per-stage analyses). "{dir}" = the stage's result folder,
# "{io}" = the infer_correctness stage's io/ folder (TF.js cpu reference outputs).
STAGES = {
    # ---------------- inference ----------------
    "infer_latency": (RB, ["--mode", "latency", "--config", *PRIMARY, *INFER_CAP, "--trials", "3"],
                      [["analysis/summarize.py", "{dir}", "--series", "infer"], ["analysis/warmup.py", "{dir}"]]),
    "infer_native": (RN, ["--mode", "latency", "--config", "tf-cuda-fp32", "tf-cpu", *INFER_CAP, "--trials", "3"], []),
    "infer_native_why": (RN, ["--mode", "latency", "--config", "tf-cuda", "tf-cuda-resident", "tf-cuda-eager", *F(*SUBSET),
                              *INFER_CAP, "--trials", "2"], []),
    "infer_correctness": (RB, ["--mode", "correctness", "--config", "cpu", *PRIMARY, "--warmup", "2", "--trials", "1",
                               "--no-telemetry"], []),
    "infer_native_correctness": (RN, ["--mode", "correctness", "--config", "tf-cuda-fp32", "tf-cuda", "tf-cpu",
                                      "--io-dir", "{io}", "--warmup", "2", "--trials", "1", "--no-telemetry"], []),
    "infer_batch": (RB, ["--mode", "batch", *F(*SUBSET), "--config", *PRIMARY, "--warmup", "5", "--measure", "30",
                         "--max-measure-s", "10", "--min-measure", "10", "--min-measure-s", "1", "--trials", "2"], []),
    "infer_native_batch": (RN, ["--mode", "batch", *F(*SUBSET), "--config", "tf-cuda-fp32", "tf-cpu", "--warmup", "5",
                                "--measure", "30", "--max-measure-s", "10", "--min-measure", "10", "--min-measure-s", "1",
                                "--trials", "2"], []),
    "infer_shapes": (RB, ["--mode", "shapes", *F(*SUBSET), "--config", *PRIMARY, "--warmup", "5", "--shape-rounds", "3",
                          "--shape-repeats", "3", "--trials", "2"], []),
    "infer_why": (RB, ["--mode", "latency", *F(*SUBSET), "--config", "webgl-strict", "webgpu-strict", "webgpu-sb1",
                       "webgpu-sb64", "wasm-1t", "wasm-4t", "webgl-gpuio", "webgpu-gpuio", "webgl-fence", "webgpu-fence",
                       "webgl-sync", "webgpu-sync", *INFER_CAP, "--trials", "2"], []),
    "infer_instrumented": (RB, ["--mode", "latency", "--config", "webgl", "webgpu", "--instrument-runs", "--warmup", "10",
                                "--measure", "30", "--max-measure-s", "30", "--min-measure-s", "0", "--trials", "1"], []),
    "infer_profile": (RB, ["--mode", "profile", "--config", *PRIMARY, "--warmup", "10", "--measure", "20", "--trials", "1",
                           "--max-measure-s", "30"], []),
    "infer_responsiveness": (RB, ["--mode", "responsiveness", "--animate", *F(*SUBSET), "--config", *PRIMARY,
                                  "--warmup", "5", "--page-baseline-s", "3", "--load-s", "8", "--trials", "2",
                                  "--no-telemetry"], []),
    "infer_load": (RB, ["--mode", "load", "--with-animation", *F(*SUBSET), "--config", *PRIMARY, "--warmup", "10",
                        "--rates", "5", "20", "50", "100", "200", "--load-s", "6", "--max-queue", "16", "--drain-s", "4",
                        "--trials", "2"], []),
    "readback_micro": ("run_readback_micro.py", ["--trials", "3", "--reps", "20", "--backends", "webgl", "webgpu"], []),
    # ---------------- warm-up why-axes ----------------
    "warmup_precompile": (RB, ["--mode", "latency", *F(*SUBSET), "--config", "webgl-precompile", "webgpu-precompile",
                               "--warmup", "10", "--measure", "50", "--max-measure-s", "10", "--trials", "2"], []),
    "warmup_precompile_train": (RB, ["--mode", "train", *F(*SUBSET), "--config", "webgl-precompile", "webgpu-precompile",
                                     *TRAIN_CAP, "--no-infer-after-train", "--trials", "2"], []),
    "warmup_diskcache": (RB, ["--mode", "latency", *F(*SUBSET), "--config", "webgl-diskcache", "webgpu-diskcache",
                              "--warmup", "10", "--measure", "50", "--max-measure-s", "10", "--trials", "2"], []),
    "warmup_lifecycle_if": (RB, ["--mode", "lifecycle", "--lifecycle-order", "infer_first", *F(*SUBSET), "--config",
                                 *PRIMARY, "--train-warmup", "10", "--train-measure", "30", "--train-max-s", "10",
                                 *POST_INFER, "--trials", "2"], []),
    "warmup_lifecycle_tf": (RB, ["--mode", "lifecycle", "--lifecycle-order", "train_first", *F(*SUBSET), "--config",
                                 *PRIMARY, "--train-warmup", "10", "--train-measure", "30", "--train-max-s", "10",
                                 *POST_INFER, "--trials", "2"], []),
    # ---------------- training ----------------
    "train": (RB, ["--mode", "train", "--config", *PRIMARY, *TRAIN_CAP, *POST_INFER, "--trials", "3", "--skip-known-gaps"],
              [["analysis/summarize.py", "{dir}", "--series", "train"], ["analysis/warmup.py", "{dir}"]]),
    "native_train": (RN, ["--mode", "train", "--config", "tf-cuda-fp32", "tf-cpu", *TRAIN_CAP, *POST_INFER,
                          "--trials", "3"], []),
    "native_train_why": (RN, ["--mode", "train", *F(*SIZES_135), "--config", "tf-cuda", "tf-cuda-eager", *TRAIN_CAP,
                              *POST_INFER, "--trials", "2"], []),
    "train_instrumented": (RB, ["--mode", "train", "--config", *PRIMARY, "--instrument-runs", "--train-warmup", "10",
                                "--train-measure", "10", "--train-max-s", "30", "--min-measure-s", "0",
                                "--no-infer-after-train", "--trials", "1", "--skip-known-gaps"], []),
    "train_profile": (RB, ["--mode", "train_profile", "--config", *PRIMARY, "--train-warmup", "10", "--profile-steps", "5",
                           "--train-max-s", "30", "--trials", "1", "--skip-known-gaps"], []),
    # Coverage record of the CNN x wasm training gap (fails in seconds with the kernel error). The L5 / G5 x wasm
    # gap (4 GB heap exhausted, then a hang) is not re-run: evidence in results/<device>/probes/p02_l5diag (FEASIBILITY.md).
    "train_coverage": (RB, ["--mode", "train", "--config", "wasm-8t", "--filter", "C1", "C5", "--trials", "1",
                            "--no-telemetry", "--no-infer-after-train"], []),
    "train_correctness": (RB, ["--mode", "train_correctness", "--config", *PRIMARY, "--correctness-steps", "20",
                               "--trials", "1", "--no-telemetry", "--skip-known-gaps"], []),
    "train_correctness_cpu": (RB, ["--mode", "train_correctness", *F(*SIZES_12), "--config", "cpu",
                                   "--correctness-steps", "20", "--trials", "1", "--no-telemetry"], []),
    "native_train_correctness_ref": (RN, ["--mode", "train_correctness", "--config", "tf-cpu", "--determinism",
                                          "--optimizer-impl", "tfjs", "--trials", "1", "--no-telemetry"], []),
    "native_train_correctness_keras": (RN, ["--mode", "train_correctness", "--config", "tf-cpu", "tf-cuda-fp32",
                                            "--optimizer-impl", "keras", "--trials", "1", "--no-telemetry"], []),
    "native_train_correctness_gpu": (RN, ["--mode", "train_correctness", "--config", "tf-cuda-fp32", "tf-cuda",
                                          "--optimizer-impl", "tfjs", "--trials", "1", "--no-telemetry"], []),
    "train_long": (RB, ["--mode", "train_long", *F(*SUBSET), "--config", *PRIMARY, "--long-epochs", "3", "--trials", "2",
                        "--skip-known-gaps"], []),
    "native_train_long": (RN, ["--mode", "train_long", *F(*SUBSET), "--config", "tf-cuda-fp32", "--long-epochs", "3",
                               "--trials", "2"], []),
    "train_batch": (RB, ["--mode", "train_batch", *F(*SUBSET), "--config", *PRIMARY, "--train-measure", "30",
                         "--batch-max-s", "10", "--min-measure-s", "1", "--trials", "2", "--skip-known-gaps"], []),
    "native_train_batch": (RN, ["--mode", "train_batch", *F(*SUBSET), "--config", "tf-cuda-fp32", "tf-cpu",
                                "--train-measure", "30", "--batch-max-s", "10", "--min-measure-s", "1", "--trials", "2"], []),
    "train_why": (RB, ["--mode", "train", *F(*SUBSET), "--config", "webgl-strict", "webgpu-strict", "webgpu-sb1",
                       "webgpu-sb64", "wasm-1t", "wasm-4t", *TRAIN_CAP, "--no-infer-after-train", "--trials", "2",
                       "--skip-known-gaps"], []),
    "train_sync_io": (RB, ["--mode", "train", *F(*SUBSET), "--config", "webgl-tfence", "webgpu-tfence", "webgl-resident",
                           "webgpu-resident", *TRAIN_CAP, "--no-infer-after-train", "--trials", "2"], []),
    "train_pipelined": (RB, ["--mode", "train_pipelined", *F(*SUBSET), "--config", *PRIMARY, *TRAIN_CAP,
                             "--pipeline-window", "10", "--trials", "2", "--skip-known-gaps"], []),
    "train_phases": (RB, ["--mode", "train_phases", *F(*SUBSET), "--config", *PRIMARY, "--train-warmup", "10",
                          "--train-measure", "50", "--phase-max-s", "10", "--trials", "2", "--skip-known-gaps"], []),
    "native_train_phases": (RN, ["--mode", "train_phases", *F(*SUBSET), "--config", "tf-cuda-fp32", "tf-cuda-eager",
                                 "--train-warmup", "10", "--train-measure", "50", "--phase-max-s", "10", "--trials", "2"], []),
    "train_fit": (RB, ["--mode", "train_fit", *F(*SUBSET), "--config", *PRIMARY, "--fit-epochs", "2", "--trials", "2",
                       "--skip-known-gaps"], []),
    "native_train_fit": (RN, ["--mode", "train_fit", *F(*SUBSET), "--config", "tf-cuda-fp32", "tf-cuda-eager",
                              "--fit-epochs", "2", "--trials", "2"], []),
    "train_responsiveness": (RB, ["--mode", "train_responsiveness", "--animate", *F(*SUBSET), "--config", *PRIMARY,
                                  "--train-warmup", "5", "--page-baseline-s", "3", "--load-s", "8", "--trials", "2",
                                  "--no-telemetry", "--skip-known-gaps"], []),
    # ---------------- GPU traces ----------------
    "gpu_trace": ("nsight/capture_matrix.py", [], []),
}
NEEDS_IO = {"infer_native_correctness"}
COOLED = {RB, RN}
DIR_IS_RESULT = {"nsight/capture_matrix.py"}


def newest_subdir(path):
    subdirs = [p for p in Path(path).iterdir() if p.is_dir() and (p / "env.json").exists()]
    return max(subdirs, key=lambda p: p.stat().st_mtime) if subdirs else None


def stage_result(study, stage):
    done = study / stage / "stage_done.json"
    if done.exists():
        return resolve_result_dir(study, stage, json.loads(done.read_text(encoding="utf-8")).get("result_dir"))
    return None


def run_logged(cmd, log_path):
    """Run cmd, echo its output and append it to log_path. Returns the exit code."""
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n### {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", cwd=ROOT)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return p.wait()


def boost_probe(study, timeout_s=900):
    """The GPU must boost: a short WebGPU D5 training run (one launch, ~10 s of steps) must reach >= BOOST_MHZ.
    (The sibling's probe workload, WebGPU ResNet-50 inference, isn't in this study; probe P0.10 chose this one.)"""
    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        clock = ClockSampler().start()
        cmd = [PY, RB, "--mode", "train", "--config", "webgpu", "--filter", "D5", "--trials", "1", "--train-warmup", "10",
               "--train-measure", "300", "--train-max-s", "10", "--no-infer-after-train", "--no-telemetry",
               "--low-clock-retries", "0", "--output-dir", str(study / "_guards" / "boost_probe")]
        rc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode
        mx = clock.stop()
        if IS_MAC:
            # The Apple GPU clock needs root (powermetrics): the probe still runs (it warms the browser binaries
            # in the OS file cache and is the drift sentinel) and the power-state guard stands in for the boost check.
            if rc == 0:
                return True, f"boost probe (macOS: GPU clock not readable, run as warm-up + drift sentinel), attempt {attempt}"
            if time.time() - t0 > timeout_s:
                return False, f"boost probe failed: rc {rc} after {attempt} attempts"
            time.sleep(60)
            continue
        if rc == 0 and mx is not None and mx >= BOOST_MHZ:
            return True, f"boost probe: SM max {mx} MHz (attempt {attempt})"
        if time.time() - t0 > timeout_s:
            return False, f"boost probe failed: SM max {mx} MHz after {attempt} attempts (rc {rc})"
        time.sleep(60)


BROWSER_SCRIPTS = {RB, "run_readback_micro.py"}

# --flight verify: the cross-browser verification flight (reports/<device>/compare/VERIFICATION_PLAN.md).
# Same stage commands, caps, trial counts and discipline as the Chrome flight; only the model list of the primary
# timing stages shrinks to the six-model SUBSET (one per family). The why-axis stages already use SUBSET, and the
# correctness stages keep all 30 models. Appended after the stage's own arguments, so --filter replaces theirs.
VERIFY_STAGES = ["infer_latency", "infer_correctness", "infer_instrumented", "infer_profile", "infer_why",
                 "readback_micro", "warmup_lifecycle_if", "train", "train_correctness", "train_instrumented",
                 "train_profile", "train_why"]
VERIFY_SUBSET_STAGES = {"infer_latency", "infer_instrumented", "infer_profile", "train", "train_instrumented",
                        "train_profile"}


def browser_priming(study, browser):
    """One unrecorded launch of the replication browser (a short WebGPU D1 latency run), so a stage's first
    measured launch doesn't pay for loading the browser's binaries from disk (the boost probe does this for Chrome)."""
    cmd = [PY, RB, "--browser", browser, "--mode", "latency", "--config", "webgpu", "--filter", "D1", "--trials", "1",
           "--warmup", "2", "--measure", "5", "--min-measure-s", "0", "--no-telemetry", "--low-clock-retries", "0",
           "--output-dir", str(study / "_guards" / "browser_priming")]
    rc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode
    return f"{browser} priming launch: exit {rc}"


def verify_analyses(study, browser="firefox", reference=None):
    """Verification-flight analyses: training numerics against the Chrome flight's tf-cpu reference, then the
    pre-registered Firefox-vs-Chrome comparison (analysis/browser_compare.py). A Chrome flight on another device
    (reference = that device's Chrome full flight) gets only the numerics check against the reference's tf-cpu
    trajectory; its cross-device comparison is a separate, pre-registered analysis, not browser_compare."""
    chrome = Path(reference) if reference else layout.run_dir("chrome", "full")
    if not chrome.is_absolute():
        chrome = ROOT / chrome
    out = study / "analysis"
    out.mkdir(exist_ok=True)
    jobs = []
    tc, ref = stage_result(study, "train_correctness"), stage_result(chrome, "native_train_correctness_ref")
    if tc and ref:
        jobs.append(("train_correctness", ["analysis/train_correctness.py", str(tc), str(ref), "--reference",
                                           "tf-cpu+tfjsadam", "--out", str(out / "train_correctness")]))
    if browser != "chrome":
        jobs.append(("browser_compare", ["analysis/browser_compare.py", "compare", "--chrome", str(chrome),
                                         "--firefox", str(study), "--out", str(out / "browser_compare")]))
    return [(name, run_logged([PY, *cmd], study / "analysis.log")) for name, cmd in jobs]


def cross_stage_analyses(study):
    """Analyses that combine stages; each writes into <study>/analysis/<name>/."""
    have = {s: str(p) for s in STAGES if (p := stage_result(study, s))}
    out = study / "analysis"
    out.mkdir(exist_ok=True)

    def dirs(*stages):
        return [have[s] for s in stages if s in have]

    jobs = []
    for series, stages in (("infer", ("infer_latency", "infer_native", "infer_native_why", "infer_why", "warmup_precompile",
                                      "warmup_diskcache")),
                           ("train", ("train", "native_train", "native_train_why", "train_why", "train_sync_io")),
                           ("infer_b1", ("train", "native_train")), ("infer_bB", ("train", "native_train"))):
        if dirs(*stages):
            jobs.append(["analysis/summarize.py", *dirs(*stages), "--series", series, "--out", str(out / "latency")])
    if dirs("infer_latency", "train", "native_train", "infer_native", "warmup_precompile", "warmup_diskcache"):
        jobs.append(["analysis/warmup.py", *dirs("infer_latency", "train", "native_train", "infer_native",
                                                  "warmup_precompile", "warmup_diskcache"), "--out", str(out / "warmup")])
    corr = dirs("train_correctness", "train_correctness_cpu", "native_train_correctness_ref",
                "native_train_correctness_keras", "native_train_correctness_gpu")
    if corr:
        jobs.append(["analysis/train_correctness.py", *corr, "--reference", "tf-cpu+tfjsadam",
                     "--out", str(out / "train_correctness")])
    for s in ("infer_latency", "infer_native", "train", "native_train", "train_long", "train_batch", "infer_batch"):
        if s in have:
            jobs.append(["analysis/gpu_telemetry.py", have[s]])
    if dirs("infer_latency", "warmup_precompile", "infer_instrumented", "infer_shapes"):
        jobs.append(["analysis/init_reuse.py", *dirs("infer_latency", "warmup_precompile", "infer_instrumented", "infer_shapes"),
                     "--out", str(out / "init_reuse")])
    if "infer_profile" in have:
        jobs.append(["analysis/profile_summary.py", have["infer_profile"], "--latency", *dirs("infer_latency"),
                     "--out", str(out / "profile")])
    if "train_profile" in have:
        jobs.append(["analysis/profile_summary.py", have["train_profile"], "--mode", "train_profile",
                     "--latency", *dirs("train"), "--out", str(out / "train_profile")])
    if dirs("infer_profile", "train_profile"):
        jobs.append(["analysis/features.py", *dirs("infer_profile", "train_profile"), "--out", str(out / "features")])
    if dirs("train_profile", "train_instrumented", "train_phases", "native_train_phases"):
        jobs.append(["analysis/train_phase.py", "--profile", *dirs("train_profile"), "--instrumented",
                     *dirs("train_instrumented"), "--phases", *dirs("train_phases", "native_train_phases"),
                     "--out", str(out / "train_phase")])
    if dirs("train_long", "native_train_long"):
        jobs.append(["analysis/equivalence.py", "--long", *dirs("train_long", "native_train_long"),
                     "--stage", *dirs("train", "native_train"), "--out", str(out / "equivalence")])
    dm = (["--profile", have["infer_profile"]] if "infer_profile" in have else []) + \
         (["--latency", *dirs("infer_latency", "infer_why")] if "infer_why" in have else []) + \
         (["--micro", have["readback_micro"]] if "readback_micro" in have else [])
    if dm:
        jobs.append(["analysis/data_movement.py", *dm, "--out", str(out / "data_movement")])
    if dirs("infer_latency", "infer_why"):
        jobs.append(["analysis/cost.py", *dirs("infer_latency", "infer_why"), "--out", str(out / "cost")])
    mem = [have[s] for s in STAGES if s in have and s not in ("readback_micro", "gpu_trace")]
    if mem:
        jobs.append(["analysis/memory.py", *mem, "--out", str(out / "memory")])
    for s in ("infer_responsiveness", "train_responsiveness"):
        if s in have:
            jobs.append(["analysis/responsiveness.py", have[s], "--out", str(out / s)])
    if "infer_load" in have:
        jobs.append(["analysis/load.py", have["infer_load"], "--slo-ms", "100", "--max-miss", "0.01",
                     "--out", str(out / "load")])
    if "infer_native" in have and "infer_latency" in have:
        jobs.append(["analysis/native_compare.py", "--study", str(study), "--out", str(out / "native")])
    if "gpu_trace" in have:
        jobs.append(["analysis/nsys_summary.py", str(Path(have["gpu_trace"]) / "nsys"),
                     str(Path(have["gpu_trace"]) / "nsys_native"), "--out", str(out / "gpu_trace")])
    results = []
    for j in jobs:
        rc = run_logged([PY, *j], study / "analysis.log")
        results.append((j[0], rc))
    (out / "sources.json").write_text(json.dumps(have, indent=1), encoding="utf-8")
    return results


def keep_awake():
    """Per-process request: no sleep, display on, while this process runs (released at exit)."""
    if sys.platform == "win32":
        import ctypes
        ok = ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        print(f"keep-awake request: {'set' if ok else 'FAILED'}")
    elif IS_MAC:
        # caffeinate holds display / idle / disk / system sleep off until this process exits
        subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        print("keep-awake request: caffeinate set")


def mac_native_args(bench_args):
    """A native stage's arguments with its --config list mapped by MAC_NATIVE. Returns (args, dropped configs)."""
    out, dropped, in_cfg = [], [], False
    for a in bench_args:
        if a.startswith("--"):
            in_cfg = a == "--config"
            out.append(a)
        elif in_cfg:
            (out.append(MAC_NATIVE[a]) if a in MAC_NATIVE else dropped.append(a))
        else:
            out.append(a)
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=None, choices=list(STAGES))
    ap.add_argument("--flight", default=None, choices=("verify",),
                    help="verify: the Firefox verification flight (stage list and model subset above)")
    ap.add_argument("--resume", action="store_true", help="skip stages that already have stage_done.json")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--no-boost-probe", action="store_true", help="development only")
    ap.add_argument("--browser", default="chrome", choices=("chrome", "firefox"))
    ap.add_argument("--run-name", default=None, help="folder name under results/<device>/<browser>/ (default: verify for "
                    "--flight verify, else full)")
    ap.add_argument("--device-id", default=None, help="result folder of this machine (default: device.local.json)")
    ap.add_argument("--output-dir", default=None, help="explicit study folder (overrides --run-name and --device-id)")
    ap.add_argument("--reference", default=None,
                    help="--flight verify --browser chrome on another device: the Chrome full flight to compare with "
                         "(e.g. results/win11-rtx3060/chrome-157.0.8081.0/full)")
    ap.add_argument("--stage-args", default="{}",
                    help="JSON {stage: [extra args]} appended to that stage's command (preflight uses this)")
    args = ap.parse_args()

    if args.device_id:
        os.environ["TRAINBENCH_DEVICE"] = args.device_id
    run_name = args.run_name or ("verify" if args.flight == "verify" else "full")
    study = Path(args.output_dir) if args.output_dir else layout.run_dir(args.browser, run_name)
    if not study.is_absolute():
        study = ROOT / study
    study.mkdir(parents=True, exist_ok=True)
    if not args.output_dir:
        layout.write_device_manifest()
        layout.write_browser_manifest(args.browser)
    keep_awake()
    lock = envlock.verify(scope="full")
    print(f"environment lock verified (sha256 {lock['sha256'][:16]})", flush=True)
    # The lock that governs this study, kept with its results (env.lock.json itself is not committed: it
    # records the commit it was frozen at).
    (study / "env.lock.json").write_text(envlock.LOCK.read_text(encoding="utf-8"), encoding="utf-8")
    layout.update_run_manifest(study, kind="flight", run=study.name, flight=args.flight, browser=layout.browser_id(args.browser),
                               device=layout.device_id(), harness=layout.harness_state(), command=sys.argv,
                               env_lock_sha256=lock["sha256"])
    stage_args = json.loads(args.stage_args)
    if args.flight == "verify":
        if args.browser == "chrome" and not args.reference:
            ap.error("--flight verify with --browser chrome is the cross-device replication flight: give --reference")
        args.stages = args.stages or VERIFY_STAGES
        for st in VERIFY_SUBSET_STAGES:
            stage_args[st] = [*stage_args.get(st, []), *F(*SUBSET)]
    args.stages = args.stages or list(STAGES)
    summary = []
    if not args.analyze_only:
        for stage in args.stages:
            script, bench_args, analyses = STAGES[stage]
            if args.browser != "chrome":
                if script not in BROWSER_SCRIPTS:
                    summary.append((stage, f"not run: not a browser stage ({args.browser} replication)", 0, None))
                    print(f"\n=== {stage}: not a browser stage; skipped for --browser {args.browser}", flush=True)
                    continue
                bench_args = [*bench_args, "--browser", args.browser]
            stage_dir = study / stage
            if args.resume and (stage_dir / "stage_done.json").exists():
                print(f"\n=== {stage}: already done, skipped (--resume)")
                continue
            if IS_MAC and script == "nsight/capture_matrix.py":
                summary.append((stage, "not available: Nsight Systems does not exist on macOS", 0, None))
                print(f"\n=== {stage}: Nsight Systems does not exist on macOS; skipped", flush=True)
                continue
            if IS_MAC and script == RN:
                bench_args, dropped = mac_native_args(bench_args)
                if "--config" not in bench_args or bench_args[bench_args.index("--config") + 1].startswith("--"):
                    summary.append((stage, f"not available on macOS: {', '.join(dropped)} need CUDA", 0, None))
                    print(f"\n=== {stage}: no macOS counterpart of {dropped}; skipped", flush=True)
                    continue
                if dropped:
                    print(f"\n=== {stage}: macOS, configs without counterpart dropped: {dropped}", flush=True)
            if stage in NEEDS_IO:
                corr = stage_result(study, "infer_correctness")
                if corr is None:
                    summary.append((stage, "not run: needs the infer_correctness stage first", 0, None))
                    print(f"\n=== {stage}: needs the infer_correctness stage (reference outputs); skipped", flush=True)
                    continue
                bench_args = [a.replace("{io}", str(corr / "io")) for a in bench_args]
            envlock.verify(scope="full")
            ok, msg = idle_check()
            print(f"\n=== {stage}: GPU idle check: {msg}", flush=True)
            if not ok:
                summary.append((stage, f"not run: {msg}" if IS_MAC else "not run: GPU clocks locked", 0, None))
                print(f"Guard failed ({msg}). Stopping." if IS_MAC else
                      "GPU clocks look locked (idle SM clock stays high). Stopping.", flush=True)
                break
            if not args.no_boost_probe:
                ok, msg = boost_probe(study)
                print(f"=== {stage}: {msg}", flush=True)
                if not ok:
                    summary.append((stage, "not run: GPU does not boost", 0, None))
                    break
            if args.browser != "chrome":
                print(f"=== {stage}: {browser_priming(study, args.browser)}", flush=True)
            common = ["--cool-to-c", COOL_TO_C] if script in COOLED else []
            stage_dir.mkdir(parents=True, exist_ok=True)
            out_args = ["--out", str(stage_dir)] if script in DIR_IS_RESULT else ["--output-dir", str(stage_dir)]
            cmd = [PY, script, *bench_args, *common, *out_args, *stage_args.get(stage, [])]
            print(f"=== {stage}: {' '.join(cmd[1:])}", flush=True)
            t0 = time.time()
            rc = run_logged(cmd, study / f"{stage}.log")
            result_dir = stage_dir if script in DIR_IS_RESULT else newest_subdir(stage_dir)
            status = "ok" if rc == 0 and result_dir else f"failed (exit {rc})"
            checks = [["analysis/check_stage.py", "{dir}"]] if result_dir and (result_dir / "sessions.csv").exists() else []
            for a in (checks + analyses) if result_dir else []:
                acmd = [PY] + [x.replace("{dir}", str(result_dir)) for x in a]
                if run_logged(acmd, study / f"{stage}.log") != 0:
                    status += f"; analysis {a[0]} failed"
            secs = time.time() - t0
            if rc == 0 and result_dir:
                (stage_dir / "stage_done.json").write_text(json.dumps(
                    {"result_dir": str(result_dir), "status": status, "minutes": round(secs / 60, 1),
                     "env_lock_sha256": lock["sha256"], "command": cmd[1:]}, indent=1), encoding="utf-8")
            summary.append((stage, status, secs, result_dir))
            if rc != 0:
                print(f"=== {stage} failed (exit {rc}); stopping so it can be fixed and resumed.", flush=True)
                break

    if args.flight == "verify" and not args.no_analyze:
        for name, rc in verify_analyses(study, args.browser, args.reference):
            summary.append((f"analysis:{name}", "ok" if rc == 0 else f"failed (exit {rc})", 0, study / "analysis"))
    elif not args.no_analyze:
        print("\n=== Cross-stage analyses", flush=True)
        for name, rc in cross_stage_analyses(study):
            summary.append((f"analysis:{Path(name).stem}", "ok" if rc == 0 else f"failed (exit {rc})", 0, study / "analysis"))

    print("\n=== Study summary")
    for stage, status, secs, d in summary:
        print(f"{stage:32s} {status:45s} {secs / 60:6.1f} min  {d}")
    layout.update_run_manifest(study, finished_at=datetime.now().isoformat(timespec="seconds"),
                               stage_status={stage: status for stage, status, _, _ in summary})


if __name__ == "__main__":
    main()
