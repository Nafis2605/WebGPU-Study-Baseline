"""
Browser driver for the custom-model study (warmup / inference / training). Port of the sibling TF.js
inference study's driver (TFJS-WebGPU-Study/run_benchmark.py); same launch discipline and outputs.

Each (trial, model, config) experiment runs in a brand-new Chrome for Testing process with an empty
profile, so no shader, pipeline or HTTP cache carries over (except the *-diskcache warm-up configs,
which deliberately reuse one profile across a priming launch and the measured launch). The page
(src/main.js) exposes window.__bench; results come back as structured JSON.

The backend is chosen explicitly with tf.setBackend() and never falls back: the experiment fails if it
doesn't initialise. A run is "ok" only when graphics-API counters confirm the requested backend did the
work, no Math.random call happened, the verification run leaked no tensor, and (strict configs) no
kernel was forwarded to TF.js's CPU implementation.

Pillars (--mode):
  inference  latency, shapes, batch, profile, correctness, load, responsiveness   (trained weights)
  training   train, train_pipelined, train_profile, train_correctness, train_batch, train_long,
             train_fit, train_phases, train_responsiveness                          (initial weights)
  warm-up    lifecycle (cross-phase transfer), plus the *-precompile / *-diskcache configs
Cold-start metrics of each pillar come from that pillar's own cold launches.

Output (benchmark_results/<timestamp>/): env.json, experiments.jsonl (source of truth), sessions.csv,
runs.csv, kernels.csv (profile modes), telemetry CSVs, io/ (correctness), logs/.

Usage:
    npm run bench:serve        # production build + preview server (port 4185), in another terminal
    python run_benchmark.py --mode latency --config wasm-8t webgl webgpu --filter D1 L1 --trials 1
    python run_benchmark.py --mode train --config webgpu --filter T1 --train-measure 50
"""
import argparse
import asyncio
import json
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from playwright.async_api import async_playwright

from benchkit.chrome import CHROME_FLAGS, chrome_page, find_chrome_dev
from benchkit.correctness import compare_all
from benchkit.devtools import DiagCollector
from benchkit.envinfo import host_info
from benchkit.results import ResultWriter
from benchkit.paths import SPEC
from benchkit import envlock

# Named experiment configurations: TF.js backend + runtime options. Env flags are set before
# tf.setBackend() (src/tfjs-setup.js applyConfig).
CONFIGS = {
    "wasm-1t": {"backend": "wasm", "threads": 1},
    "wasm-4t": {"backend": "wasm", "threads": 4},
    "wasm-8t": {"backend": "wasm", "threads": 8},     # = TF.js's own default here (half the logical cores, max 8)
    "webgl": {"backend": "webgl"},                    # TF.js defaults (packing on, CPU forwarding on)
    "webgpu": {"backend": "webgpu"},                  # TF.js defaults (15 dispatches/submit, CPU forwarding on)
    # Why-axes
    "webgl-strict": {"backend": "webgl", "strict": True},      # CPU forwarding off
    "webgpu-strict": {"backend": "webgpu", "strict": True},    # CPU forwarding off
    **{f"webgpu-sb{n}": {"backend": "webgpu", "flags": {"WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE": n}} for n in (1, 4, 64, 512)},
    # Warm-up: parallel shader/pipeline pre-compilation before the first run (inference modes).
    "webgl-precompile": {"backend": "webgl", "precompile": True},
    "webgpu-precompile": {"backend": "webgpu", "precompile": True},
    # Warm-up: returning user (GPU shader disk cache kept from a priming launch on the same profile).
    "webgl-diskcache": {"backend": "webgl", "diskcache": True},
    "webgpu-diskcache": {"backend": "webgpu", "diskcache": True},
    # Inference data movement: GPU-resident input; fence before readback; dataSync() readback.
    "webgl-gpuio": {"backend": "webgl", "io": "gpu"},
    "webgpu-gpuio": {"backend": "webgpu", "io": "gpu"},
    "webgl-fence": {"backend": "webgl", "fence": True},
    "webgpu-fence": {"backend": "webgpu", "fence": True},
    "webgl-sync": {"backend": "webgl", "readback": "sync"},
    "webgpu-sync": {"backend": "webgpu", "readback": "sync"},
    # Training sync / I/O: fence before the loss read; batch uploaded before the step.
    "webgl-tfence": {"backend": "webgl", "syncPolicy": "fence"},
    "webgpu-tfence": {"backend": "webgpu", "syncPolicy": "fence"},
    "webgl-resident": {"backend": "webgl", "dataIo": "resident"},
    "webgpu-resident": {"backend": "webgpu", "dataIo": "resident"},
    # Correctness reference: TF.js's pure-JS cpu backend.
    "cpu": {"backend": "cpu"},
}
PRIMARY = ["wasm-8t", "webgl", "webgpu"]
INFER_MODES = ("latency", "shapes", "batch", "profile", "correctness", "load", "responsiveness")
TRAIN_MODES = ("train", "train_pipelined", "train_profile", "train_correctness", "train_batch", "train_long",
               "train_fit", "train_phases", "train_responsiveness", "lifecycle")
MODES = INFER_MODES + TRAIN_MODES
# Known coverage gaps (genuine TF.js limitations, verified in probes): recorded, never worked around.
# Keys are (family or model name, TF.js backend, pillar).
KNOWN_GAPS = {
    ("cnn", "wasm", "train"): "Conv2DBackpropFilter is not registered for the wasm backend",
    # Training L5 / G5 on wasm fills the 4 GB WASM memory and then hangs (one core busy, no progress, no
    # error) instead of failing: probes/out/p02_l5diag (renderer private memory 4.49 GB within 36 s, flat for 15 min).
    ("L5", "wasm", "train"): "wasm heap exhausted (4 GB limit): the step hangs without an error",
    ("G5", "wasm", "train"): "wasm heap exhausted (4 GB limit): the step hangs without an error",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:4185/", help="benchmark page; start it with: npm run bench:serve")
    p.add_argument("--config", nargs="+", default=PRIMARY, choices=list(CONFIGS))
    p.add_argument("--exclude", nargs="+", default=[], metavar="MODEL:CONFIG",
                   help="skip these (model name, config) pairs, e.g. C1:wasm-8t")
    p.add_argument("--skip-known-gaps", action="store_true",
                   help="skip pairs listed in KNOWN_GAPS (they are still run, and recorded as failures, without it)")
    p.add_argument("--filter", nargs="+", default=None, help="keep models whose name is one of these (or starts with one)")
    p.add_argument("--mode", default="latency", choices=MODES)
    p.add_argument("--weights", default=None, choices=("init", "trained"),
                   help="weight file (default: trained for inference modes, init for training modes)")
    p.add_argument("--verify-weights", action="store_true", help="read every weight back and check its sha256 (gate)")
    # Inference
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measure", type=int, default=200)
    p.add_argument("--max-measure-s", type=float, default=None)
    p.add_argument("--min-measure", type=int, default=30)
    p.add_argument("--min-measure-s", type=float, default=2.0,
                   help="every measured phase also runs for at least this long (GPU energy windows need >= 1 s)")
    p.add_argument("--infer-batch", type=int, default=1, help="base batch of the inference modes")
    p.add_argument("--batch-values", type=int, nargs="+", default=None, help="batch mode values (default 1..256)")
    p.add_argument("--shape-values", type=int, nargs="+", default=None, help="shapes mode batch values (default 1..64)")
    p.add_argument("--shape-rounds", type=int, default=3)
    p.add_argument("--shape-repeats", type=int, default=3)
    p.add_argument("--correctness-batch", type=int, default=32)
    # Training
    p.add_argument("--train-warmup", type=int, default=15)
    p.add_argument("--train-measure", type=int, default=100)
    p.add_argument("--train-max-s", type=float, default=30.0)
    p.add_argument("--train-min-measure", type=int, default=10)
    p.add_argument("--warmup-max-s", type=float, default=30.0,
                   help="training warm-up stops after this long once >= 3 steps are done (flagged)")
    p.add_argument("--train-batch", type=int, default=None, help="override the spec's training batch")
    p.add_argument("--train-batch-values", type=int, nargs="+", default=None)
    p.add_argument("--batch-warmup", type=int, default=5)
    p.add_argument("--batch-max-s", type=float, default=10.0)
    p.add_argument("--batch-min-measure", type=int, default=10)
    p.add_argument("--infer-warmup", type=int, default=5, help="post-training / lifecycle inference warm-up")
    p.add_argument("--infer-measure", type=int, default=50)
    p.add_argument("--infer-max-s", type=float, default=10.0)
    p.add_argument("--infer-min-measure", type=int, default=20)
    p.add_argument("--no-infer-after-train", action="store_true")
    p.add_argument("--pipeline-window", type=int, default=10)
    p.add_argument("--long-epochs", type=int, default=3)
    p.add_argument("--fit-epochs", type=int, default=2)
    p.add_argument("--phase-max-s", type=float, default=10.0)
    p.add_argument("--profile-steps", type=int, default=5)
    p.add_argument("--correctness-steps", type=int, default=20)
    p.add_argument("--lifecycle-order", default="infer_first", choices=("infer_first", "train_first"))
    # Responsiveness / load (sibling options)
    p.add_argument("--animate", action="store_true")
    p.add_argument("--page-baseline-s", type=float, default=5.0)
    p.add_argument("--load-s", type=float, default=10.0)
    p.add_argument("--rates", type=float, nargs="+", default=[1, 2, 5, 10, 20, 50])
    p.add_argument("--max-queue", type=int, default=16)
    p.add_argument("--drain-s", type=float, default=5.0)
    p.add_argument("--with-animation", action="store_true")
    p.add_argument("--click-interval-s", type=float, default=0.25)
    p.add_argument("--pacing", default="yield", choices=("yield", "back_to_back"))
    # Launch discipline
    p.add_argument("--instrument-runs", action="store_true",
                   help="per-run / per-step counter deltas for the whole launch; timings are then not latency data")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--trace", action="store_true", help="Chrome trace with GPU categories per experiment")
    p.add_argument("--session-timeout-s", type=int, default=600)
    p.add_argument("--experiment-timeout-s", type=int, default=3600)
    p.add_argument("--cool-to-c", type=float, default=None)
    p.add_argument("--cooldown-s", type=float, default=5.0)
    p.add_argument("--low-clock-retries", type=int, default=2)
    p.add_argument("--verify-full", action="store_true")
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--telemetry-interval-s", type=float, default=0.05)
    p.add_argument("--no-pcie", action="store_true")
    p.add_argument("--baseline-s", type=float, default=3.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--exact-output-dir", action="store_true", help="write into --output-dir itself (no timestamp folder)")
    return p.parse_args(argv)


def load_models(filters):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    models = [{**m, "optimizer": spec["optimizer"]} for m in spec["models"]]
    if filters:
        models = [m for m in models if any(m["name"] == f or m["name"].startswith(f) for f in filters)]
    return models


def check_server(url):
    try:
        with urlopen(url, timeout=10) as r:
            coop = r.headers.get("Cross-Origin-Opener-Policy")
            coep = r.headers.get("Cross-Origin-Embedder-Policy")
    except Exception as e:
        sys.exit(f"Server not reachable at {url} ({e}). Start it with: npm run bench:serve")
    if coop != "same-origin" or coep != "require-corp":
        sys.exit(f"page is not cross-origin isolated (COOP={coop}, COEP={coep}): WASM threads and 5 µs timers need it")


def known_gap(model, config, mode):
    backend = CONFIGS[config]["backend"]
    pillar = "train" if mode in TRAIN_MODES else "infer"
    return KNOWN_GAPS.get((model["family"], backend, pillar)) or KNOWN_GAPS.get((model["name"], backend, pillar))


async def click_while_running(page, experiment, interval_s):
    """Responsiveness modes: click the page's interaction target with real input events until the
    experiment finishes (sibling driver)."""
    clicks = []
    while not experiment.done():
        if await page.evaluate("() => !!window.__bench.interactionTarget"):
            break
        await asyncio.sleep(0.05)
    target = page.locator("#interactBtn")
    await target.scroll_into_view_if_needed()
    box = await target.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    if not (0 <= y <= page.viewport_size["height"] and 0 <= x <= page.viewport_size["width"]):
        raise RuntimeError(f"interaction target at ({x:.0f}, {y:.0f}) is outside the viewport")
    while not experiment.done():
        t0 = time.time() * 1000.0
        await page.mouse.click(x, y)
        clicks.append({"t_send_epoch_ms": t0, "roundtrip_ms": time.time() * 1000.0 - t0})
        await asyncio.sleep(interval_s)
    return clicks


async def wait_for_gpu_temp(limit_c, timeout_s=180):
    """Hold the next launch until the GPU has cooled to limit_c (returns seconds waited)."""
    import pynvml as nv
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    t0 = time.time()
    while nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU) > limit_c and time.time() - t0 < timeout_s:
        await asyncio.sleep(2)
    return time.time() - t0


from benchkit.clockguard import LOW_CLOCK_MHZ, ClockSampler  # noqa: E402


def page_cfg(args, model, config, trial):
    """The camelCase experiment config passed to window.__bench.runExperiment."""
    c = {k: v for k, v in CONFIGS[config].items() if k != "diskcache"}
    return {
        **c, "configLabel": config, "model": model, "mode": args.mode, "seed": args.seed, "trial": trial,
        "weights": args.weights, "verifyWeights": args.verify_weights,
        "warmup": args.warmup, "measure": args.measure, "maxMeasureS": args.max_measure_s, "minMeasure": args.min_measure, "minMeasureS": args.min_measure_s,
        "inferBatch": args.infer_batch, "batchValues": args.batch_values, "shapeValues": args.shape_values,
        "shapeRounds": args.shape_rounds, "shapeRepeats": args.shape_repeats, "correctnessBatch": args.correctness_batch,
        "trainWarmup": args.train_warmup, "trainMeasure": args.train_measure, "trainMaxS": args.train_max_s,
        "trainMinMeasure": args.train_min_measure, "warmupMaxS": args.warmup_max_s, "trainBatch": args.train_batch,
        "trainBatchValues": args.train_batch_values, "batchWarmup": args.batch_warmup, "batchMaxS": args.batch_max_s,
        "batchMinMeasure": args.batch_min_measure,
        "inferWarmup": args.infer_warmup, "inferMeasure": args.infer_measure, "inferMaxS": args.infer_max_s,
        "inferMinMeasure": args.infer_min_measure, "inferAfterTrain": not args.no_infer_after_train,
        "pipelineWindow": args.pipeline_window, "longEpochs": args.long_epochs, "fitEpochs": args.fit_epochs,
        "phaseMaxS": args.phase_max_s, "profileSteps": args.profile_steps, "correctnessSteps": args.correctness_steps,
        "lifecycleOrder": args.lifecycle_order,
        "instrumentRuns": args.instrument_runs,
        "animate": args.animate, "baselineSeconds": args.page_baseline_s, "loadSeconds": args.load_s,
        "pacing": args.pacing, "rates": args.rates, "maxQueue": args.max_queue, "drainSeconds": args.drain_s,
        "withAnimation": args.with_animation,
        "sessionTimeoutMs": args.session_timeout_s * 1000,
    }


async def prime_disk_cache(pw, args, chrome, cfg, user_data_dir, log_path):
    """*-diskcache configs: one untimed launch on the shared profile that runs the same experiment,
    so the GPU shader disk cache holds this model's shaders; the measured launch reuses the profile."""
    flags = [f for f in CHROME_FLAGS if f != "--disable-gpu-shader-disk-cache"]
    async with chrome_page(pw, chrome, args.url, args.headless, log_path=log_path, user_data_dir=user_data_dir,
                           flags=flags) as (page, _):
        r = await asyncio.wait_for(page.evaluate("cfg => window.__bench.runExperiment(cfg)", cfg),
                                   timeout=args.experiment_timeout_s)
    return {"status": r.get("status"), "first_run_ms": (r.get("runs") or [{}])[0].get("dur_ms")}


async def run_one(pw, args, chrome, model, config, trial, writer, telemetry=None, attempt=0):
    backend = CONFIGS[config]["backend"]
    diskcache = CONFIGS[config].get("diskcache", False)
    tag = f"t{trial}_{model['name']}_{config}"
    log_path = writer.out_dir / "logs" / f"{tag}.log"
    cfg = page_cfg(args, model, config, trial)
    diag = DiagCollector(writer.out_dir / "logs" / tag, cpu_profile=False, trace=True) if args.trace else None
    started = datetime.now().isoformat(timespec="seconds")
    lock = envlock.verify(scope="launch")
    shared_profile = tempfile.mkdtemp(prefix="trainbench-diskcache-") if diskcache else None
    prime = None
    if diskcache:
        try:
            prime = await prime_disk_cache(pw, args, chrome, cfg, shared_profile, writer.out_dir / "logs" / f"{tag}.prime.log")
        except Exception as e:
            prime = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    clock = ClockSampler().start() if backend in ("webgl", "webgpu") else None
    cooled_s = await wait_for_gpu_temp(args.cool_to_c) if args.cool_to_c is not None else None
    if telemetry:
        telemetry.start()
        telemetry.mark("baseline_start")
        await asyncio.sleep(args.baseline_s)
        telemetry.mark("baseline_end")
    try:
        extra = ["--enable-blink-features=ForceEagerMeasureMemory"] if args.mode in ("profile", "train_profile") else []
        flags = [f for f in CHROME_FLAGS if not (diskcache and f == "--disable-gpu-shader-disk-cache")]
        on_launch = telemetry.watch_chrome if telemetry else None
        async with chrome_page(pw, chrome, args.url, args.headless, extra_flags=extra, log_path=log_path,
                               on_launch=on_launch, cdp_setup=diag.setup if diag else None,
                               user_data_dir=shared_profile, flags=flags) as (page, version):
            page_env = await page.evaluate("() => window.__bench.probeEnv()")
            if telemetry:
                telemetry.mark("experiment_start")
            experiment = asyncio.ensure_future(page.evaluate("cfg => window.__bench.runExperiment(cfg)", cfg))
            clicks = (await click_while_running(page, experiment, args.click_interval_s)
                      if args.mode in ("responsiveness", "train_responsiveness") else None)
            result = await asyncio.wait_for(experiment, timeout=args.experiment_timeout_s)
            if clicks is not None:
                result["python_clicks"] = clicks
            result["browser_version"] = version
            result["page_env"] = page_env
    except Exception as e:
        result = {
            "schema": 3, "model": model["name"], "family": model["family"], "size": model["size"],
            "params": model["params"], "backend": backend, "config_label": config, "mode": args.mode,
            "pillar": "train" if args.mode in TRAIN_MODES else "infer", "status": "failed",
            "failure": {"stage": "browser", "category": "browser_failure", "message": f"{type(e).__name__}: {e}"},
            "runs": [],
        }
    finally:
        if shared_profile:
            shutil.rmtree(shared_profile, ignore_errors=True)
    result.update({"trial": trial, "started_at": started, "cooled_s": cooled_s, "low_clock_reruns": attempt,
                   "env_lock_sha256": lock["sha256"], "known_gap": known_gap(model, config, args.mode)})
    if prime is not None:
        result["diskcache_prime"] = prime
    if result.get("io"):
        io_dir = writer.out_dir / "io"
        io_dir.mkdir(exist_ok=True)
        (io_dir / f"{tag}.json").write_text(json.dumps(result.pop("io")), encoding="utf-8")
        result["io_file"] = f"io/{tag}.json"
    if diag:
        result["devtools"] = diag.status
    # Low-clock guard (sibling): a GPU launch whose SM clock never exceeded 1000 MHz ran in a stuck
    # low-power state; it is re-run instead of recorded; a final failure is recorded as gpu_low_clock.
    if clock is not None:
        sm_max = clock.stop()
        result["gpu_sm_clock_max_mhz"] = sm_max
        if result.get("status") == "ok" and sm_max is not None and sm_max <= LOW_CLOCK_MHZ:
            if attempt < args.low_clock_retries:
                if telemetry:
                    telemetry.stop()
                return {"retry_low_clock": True, "sm_max": sm_max}
            result["status"] = "failed"
            result["failure"] = {"stage": "guard", "category": "gpu_low_clock",
                                 "message": f"SM clock never exceeded {LOW_CLOCK_MHZ} MHz (max {sm_max})"}
    if telemetry:
        telemetry.mark("experiment_end")
        data = telemetry.stop()
        key = {"trial": trial, "model": model["name"], "backend": config}
        writer.append_rows("gpu_samples.csv", [{**key, **s} for s in data["samples"]])
        writer.append_rows("gpu_util_samples.csv", [{**key, **s} for s in data["util"]])
        writer.append_rows("gpu_pcie_samples.csv", [{**key, **s} for s in data["pcie"]])
        writer.append_rows("gpu_marks.csv", [{**key, **s} for s in data["marks"]])
        writer.append_rows("chrome_memory_samples.csv", [{**key, **s} for s in data["chrome"]])
        writer.append_rows("system_cpu_samples.csv", [{**key, **s} for s in data["system_cpu"]])
        result["telemetry"] = {"samples": len(data["samples"]), "util_samples": len(data["util"]),
                               "pcie_samples": len(data["pcie"]), "chrome_samples": len(data["chrome"]),
                               "errors": data["errors"][:20]}
    writer.append(result)
    return result


TIMED_PHASES = ("measure", "shape", "batch", "load", "train_measure", "train_batch", "train_long", "phase_step",
                "train_pipelined", "lc_train", "lc_train1")


async def main(argv=None):
    args = parse_args(argv)
    chrome = find_chrome_dev()
    if not chrome:
        sys.exit("Pinned Chrome for Testing not found (tools/chrome-for-testing); see README setup.")
    check_server(args.url)
    models = load_models(args.filter)
    if not models:
        sys.exit("No models matched.")
    lock = envlock.verify(scope="full" if args.verify_full else "launch")
    out_dir = Path(args.output_dir) if args.exact_output_dir else Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ResultWriter(out_dir)
    env = {"created_at": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "chrome_executable": chrome,
           "chrome_flags": CHROME_FLAGS, "host": host_info(), "env_lock_sha256": lock["sha256"], "page_env": None}
    writer.write_json("env.json", env)

    def excluded(model, config):
        if args.skip_known_gaps and known_gap(model, config, args.mode):
            return True
        return any(model["name"] == sub and cfg == config for sub, cfg in (e.split(":", 1) for e in args.exclude))

    plan = [(t, m, c) for t in range(1, args.trials + 1) for m in models for c in args.config if not excluded(m, c)]
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        # Shuffle within each trial: trials stay sequential, order effects don't line up with a fixed order.
        plan = [x for t in range(1, args.trials + 1)
                for x in rng.sample([p for p in plan if p[0] == t], k=sum(p[0] == t for p in plan))]
    telemetry = None
    if not args.no_telemetry:
        from benchkit.telemetry import NvmlTelemetry
        telemetry = NvmlTelemetry(interval_s=args.telemetry_interval_s, pcie=not args.no_pcie)

    print(f"Output: {out_dir}\nChrome: {chrome}\n{len(plan)} experiments", flush=True)
    async with async_playwright() as pw:
        for n, (trial, model, config) in enumerate(plan, 1):
            t0 = time.time()
            print(f"[{n}/{len(plan)}] trial {trial} {model['name']} / {config} / {args.mode} ...", flush=True)
            attempt = 0
            while True:
                r = await run_one(pw, args, chrome, model, config, trial, writer, telemetry, attempt)
                if not r.get("retry_low_clock"):
                    break
                attempt += 1
                print(f"    low-clock launch (SM max {r['sm_max']} MHz): re-run {attempt}", flush=True)
                await asyncio.sleep(args.cooldown_s)
            measured = [x["dur_ms"] for x in r.get("runs", []) if x["phase"] in TIMED_PHASES]
            detail = (r.get("verification") or {}).get("detail") or (r.get("failure") or {}).get("message", "")
            median = sorted(measured)[len(measured) // 2] if measured else None
            print(f"    {r['status']}" + (f", median {median:.3f} ms over {len(measured)} runs" if median is not None else "")
                  + (" [capped]" if r.get("capped") else "") + f" ({time.time() - t0:.0f}s) {detail[:160]}", flush=True)
            if env["page_env"] is None and r.get("page_env"):
                env["page_env"] = r["page_env"]
                env["browser_version"] = r.get("browser_version")
                writer.write_json("env.json", env)
            if n < len(plan):
                await asyncio.sleep(args.cooldown_s)

    if args.mode == "correctness":
        rows = compare_all(writer.out_dir, {m["name"]: {"logits_outputs": ["probs"]} for m in models})
        writer.append_rows("correctness.csv", rows)
        print(f"Correctness: {len(rows)} output comparisons -> correctness.csv")
    n_sessions, n_runs = writer.export_csvs()
    print(f"Done: {n_sessions} experiments, {n_runs} run records -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    asyncio.run(main())
