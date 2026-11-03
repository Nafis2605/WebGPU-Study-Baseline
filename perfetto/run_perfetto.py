"""
Perfetto GPU trace of the TF.js page on the phone (Android target): the counterpart of nsight/run_nsys.py. Diagnostic: it
explains the study's results and never replaces its measurements.

Same page and phase as the laptop captures: Chrome (study flags, fresh processes) opens nsight/nsight.html served by Termux,
which runs one model on one study config in a closed loop and waits (wait=1). The Perfetto session is started on the phone,
the page is released, and the session is stopped after --start-at-run warm-up runs + --measure-runs runs (or --window-s).
The Mali counter producer (/system/bin/gpu_counter_producer) is started first; the trace config is
perfetto/android_gpu_trace.pbtxt (GPU counters, render stages, GPU/CPU frequency, power rails, battery, thermal, Chrome
track events). The page's per-run timestamps come over CDP, as on the laptop.

Output: <out>/<model>__<config>__<task>__closed__run<N>__<timestamp>[_<tag>]/ with trace.pftrace, trace_check.json,
page_stats.json, command.json, perfetto.log.

    .venv/Scripts/python perfetto/run_perfetto.py --model L1 --config webgpu --task infer --out <dir>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit import android, envlock, layout  # noqa: E402
from benchkit.chrome import CHROME_FLAGS  # noqa: E402
from nsight.run_nsys import PagePoller  # noqa: E402
from run_benchmark import CONFIGS  # noqa: E402

CFG = ROOT / "perfetto" / "android_gpu_trace.pbtxt"
# Diagnostic captures only (the laptop's Nsight captures likewise added --disable-gpu-sandbox): Chrome stable on Android
# registers its own track_event producers (browser, GPU, renderer) with the system Perfetto service only with this
# feature (checked with `perfetto --query`, 2026-11-02). gpu.renderstages is not registered by any Chrome process, with
# or without the GPU sandbox: render stages are a recorded gap, the Mali counters cover GPU activity.
TRACE_FLAGS = ["--enable-features=EnablePerfettoSystemTracing"]
TP_PY = ROOT / "tools" / "perfetto" / "venv" / "Scripts" / "python.exe"
REMOTE_DIR = "/data/misc/perfetto-traces"


def producer_up():
    """The Mali counter producer runs as a shell process (no root); started once and left running."""
    if "gpu_counter_producer" not in android.sh("ps -A -o NAME", check=False):
        android.sh("(nohup /system/bin/gpu_counter_producer >/dev/null 2>&1 &)", check=False)
        time.sleep(1.0)
    return "gpu_counter_producer" in android.sh("ps -A -o NAME", check=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--task", default="infer", choices=("infer", "train"))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--start-at-run", type=int, default=10)
    ap.add_argument("--measure-runs", type=int, default=200)
    ap.add_argument("--window-s", type=float, default=60.0, help="cap on the collection length")
    ap.add_argument("--load-timeout-s", type=float, default=180)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(layout.run_dir("chrome", "_dev/perfetto") / "perfetto"))
    args = ap.parse_args()
    if not android.is_android():
        sys.exit("set TRAINBENCH_TARGET=android")

    cfg = {k: v for k, v in CONFIGS[args.config].items() if k != "diskcache"}
    name = f"{args.model}__{args.config}__{args.task}__closed__run{args.start_at_run}"
    out = Path(args.out) / f"{name}__{time.strftime('%Y%m%d_%H%M%S')}{('_' + args.tag) if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    lock = envlock.verify(scope="launch")
    ok, info = android.server_ok()
    if not ok:
        sys.exit(f"page server on the phone is not answering: {info}")
    cool = android.wait_cool()
    if not producer_up():
        sys.exit("gpu_counter_producer did not start")

    url = (f"http://localhost:4185/nsight/nsight.html?model={quote(args.model)}&config={quote(args.config)}"
           f"&task={args.task}" + (f"&batch={args.batch}" if args.batch else "") + f"&wait=1&cfg={quote(json.dumps(cfg))}")
    # duration_ms caps the session on the phone too (window + margin), so a lost laptop never leaves it running
    text = CFG.read_text(encoding="utf-8").replace("duration_ms: 10000", f"duration_ms: {int((args.window_s + 30) * 1000)}")
    text += "write_into_file: true\nfile_write_period_ms: 1000\n"   # required by --detach (session runs without this shell)
    key = f"tb{int(time.time())}"
    remote_cfg, remote_trace = f"{android.TMP}/{key}.pbtxt", f"{REMOTE_DIR}/{key}.pftrace"
    android.sh(f"mkdir -p {android.TMP}")
    android.push_text(text, remote_cfg)
    flags = CHROME_FLAGS + TRACE_FLAGS
    (out / "command.json").write_text(json.dumps({"url": url, "flags": flags, "perfetto_config": text,
                                                  "remote_trace": remote_trace}, indent=1), encoding="utf-8")
    log = open(out / "perfetto.log", "w", encoding="utf-8", errors="replace")
    window, port = {}, None
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        try:
            port = android.start_chrome(flags, url)
            poller = PagePoller(pw, port)
            t_launch = time.time()
            while poller.fetch() is not None and poller.meta is None:      # page loaded and waiting (wait=1)
                if time.time() - t_launch > args.load_timeout_s:
                    raise RuntimeError(f"page did not get ready: error={poller.error}")
                time.sleep(0.2)
            window["start_request_epoch_s"] = time.time()
            r = android.sh(f"cat {remote_cfg} | perfetto --txt -c - -o {remote_trace} --detach={key} 2>&1", check=False)
            log.write(f"$ perfetto start\n{r}\n")
            time.sleep(1.0)                                              # data sources (producer, Chrome) attach
            window["started_epoch_s"] = time.time()
            window["skip_runs"] = args.start_at_run
            poller.go()
            target = args.start_at_run + args.measure_runs
            t_end = window["started_epoch_s"] + args.window_s
            while time.time() < t_end and poller.fetch() < target:
                time.sleep(0.1)
            window["runs_at_stop_request"] = len(poller.runs)
            window["stop_request_epoch_s"] = time.time()
            r = android.sh(f"perfetto --attach={key} --stop 2>&1", timeout=120, check=False)
            log.write(f"$ perfetto stop\n{r}\n")
            window["stopped_epoch_s"] = time.time()
            poller.fetch()
            poller.close()
        except Exception as e:  # noqa: BLE001
            window["error"] = f"{type(e).__name__}: {e}"
            android.sh(f"perfetto --attach={key} --stop 2>&1", timeout=120, check=False)
            poller = locals().get("poller")
        finally:
            android.stop_chrome(port)
    log.close()
    trace = out / "trace.pftrace"
    pulled = subprocess.run([android.ADB, *(["-s", android.SERIAL] if android.SERIAL else []), "pull", remote_trace, str(trace)],
                            capture_output=True, text=True)
    android.sh(f"rm -f {remote_trace} {remote_cfg}", check=False)
    check = {}
    if trace.exists():
        r = subprocess.run([str(TP_PY), str(ROOT / "perfetto" / "trace_check.py"), str(trace)], capture_output=True, text=True)
        try:
            check = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            check = {"error": (r.stdout + r.stderr)[-500:]}
    (out / "trace_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    runs = poller.runs if poller else []
    (out / "page_stats.json").write_text(json.dumps({"stats": {**((poller and poller.meta) or {}), "runs": runs},
                                                     "env_lock_sha256": lock["sha256"], "android_cool": cool,
                                                     "error": poller and poller.error, "window": window,
                                                     "pull": pulled.stdout[-200:] + pulled.stderr[-200:]}, indent=0),
                                         encoding="utf-8")
    keys = ("gpu_counter_tracks", "gpu_counter_samples", "gpu_freq_samples", "render_stage_slices", "power_rail_samples",
            "battery_samples", "thermal_samples", "chrome_slices", "chrome_gpu_slices")
    print(f"{name}: runs={len(runs)} window={window} trace={'yes' if trace.exists() else 'NO'} "
          + " ".join(f"{k}={check.get(k)}" for k in keys))


if __name__ == "__main__":
    main()
