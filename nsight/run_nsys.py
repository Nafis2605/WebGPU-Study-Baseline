"""
Nsight Systems capture of the TF.js page (diagnostic; explains the study's results, never replaces
its measurements). Port of the sibling TF.js study's nsight/run_nsys.py; --task selects inference
(trained weights, batch --batch) or training steps (initial weights, spec batch).

Launches Chrome Dev under `nsys launch` on nsight/nsight.html, which runs one model on one study
config (run_benchmark.CONFIGS) in a closed loop (back-to-back, as the latency stage) or paced (one
inference per --interval-ms, as the load stage). Collection is started by `nsys start` when the
page has completed --start-at-run inferences and stopped by `nsys stop` after --window-s, so a
capture covers a defined phase of the session:
  --start-at-run 10      the study's phase: right after its 10 warm-up runs (the latency stage
                         measures the next 200 runs, ~1-2 s for fast models)
  --start-at-s 15        sustained inference, 15 s after the first run
nsys follows Chrome's child processes; Chrome's GPU process (D3D12 via Dawn for WebGPU, D3D11 via
ANGLE for WebGL) is only traced with --disable-gpu-sandbox (added by default; probe:
VERIFICATION.md), which does not change latency (MobileNetV2 WebGPU 3.82 ms with and without).

The page's per-inference epoch timestamps are fetched over the Chrome DevTools Protocol, so the
trace can be segmented into inferences. NVML samples the SM/memory clock in parallel (independent
of nsys). The idle GPU clock is checked before and after (run_study.gpu_clock_check): nsys does not
lock clocks, and a locked clock is reported.

Output: nsight/out/nsys/<model>__<config>__<mode>__<phase>__<timestamp>/ with report.nsys-rep,
report.sqlite, page_stats.json, nvml_clock.csv, command.json, nsys.log, clock_check.json.

Usage (npm run bench:serve running on 4185):
    python nsight/run_nsys.py --model mobilenet_v2_100_224 --config webgpu
    python nsight/run_nsys.py --model mobilenet_v2_100_224 --config webgpu --start-at-s 15
    python nsight/run_nsys.py --model mobilenet_v2_100_224 --config webgpu --mode paced --interval-ms 1000 --start-at-s 5 --window-s 6
    python nsight/run_nsys.py --model movenet_lightning --config webgl --gpu-metrics
"""
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit.chrome import CHROME_FLAGS, find_chrome_dev  # noqa: E402
from run_benchmark import CONFIGS  # noqa: E402
from benchkit import envlock  # noqa: E402
from benchkit.clockguard import idle_check as gpu_clock_check  # noqa: E402

from benchkit.paths import NSYS  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def study_chrome_args():
    """The exact Chrome command line the study harness uses (benchkit.chrome.chrome_page: Playwright
    launch_persistent_context with CHROME_FLAGS adds ~35 flags of its own), read once from a real
    Playwright launch and cached. Returned without the executable, user-data-dir, the DevTools pipe
    and the start URL, in the original order."""
    cache = ROOT / "nsight" / "out" / "study_chrome_args.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import asyncio
    from playwright.async_api import async_playwright

    async def read():
        udd = tempfile.mkdtemp(prefix="trainbench-cmdline-")
        async with async_playwright() as pw:
            ctx = await pw.chromium.launch_persistent_context(udd, executable_path=find_chrome_dev(), headless=False,
                                                              args=CHROME_FLAGS, viewport={"width": 1280, "height": 900})
            await asyncio.sleep(1.5)
            cl = None
            for proc in psutil.process_iter(["cmdline"]):
                c = proc.info["cmdline"] or []
                if any(udd in x for x in c) and not any(x.startswith("--type=") for x in c):
                    cl = c
            await ctx.close()
        return [x for x in cl[1:] if not x.startswith("--user-data-dir") and x != "--remote-debugging-pipe"
                and x != "about:blank"]

    args = asyncio.run(read())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(args, indent=1), encoding="utf-8")
    return args


def kill_profile(udd):
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if udd in " ".join(p.info["cmdline"] or []):
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class PagePoller:
    """One CDP connection; fetch() appends the page's new runs (window.__nsight.stats(from))."""

    def __init__(self, pw, port):
        self.pw, self.port, self.browser, self.runs, self.meta, self.error = pw, port, None, [], None, None

    def fetch(self):
        try:
            if self.browser is None:
                self.browser = self.pw.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}", timeout=5000)
            page = next((pg for ctx in self.browser.contexts for pg in ctx.pages if "nsight.html" in pg.url), None)
            if page is None:
                return len(self.runs)
            v = page.evaluate("(f) => window.__nsight && (window.__nsight.stats ? window.__nsight.stats(f)"
                              " : {ready: window.__nsight.ready, error: window.__nsight.error})", len(self.runs))
            if v and "runs" in v:
                self.runs += v["runs"]
                self.meta = {k: v[k] for k in v if k != "runs"}
                self.error = None
            elif v and v.get("error"):
                self.error = v["error"]
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            self.browser = None
        return len(self.runs)

    def go(self, tries=20):
        """Release a page loaded with wait=1 (window.__nsight.go()); reconnects if needed."""
        for _ in range(tries):
            try:
                if self.browser is None:
                    self.browser = self.pw.chromium.connect_over_cdp(f"http://127.0.0.1:{self.port}", timeout=5000)
                page = next((pg for ctx in self.browser.contexts for pg in ctx.pages if "nsight.html" in pg.url), None)
                if page is not None and page.evaluate("() => !!(window.__nsight && window.__nsight.go)"):
                    page.evaluate("() => window.__nsight.go()")
                    return True
            except Exception as e:  # noqa: BLE001
                self.error = f"go: {type(e).__name__}: {str(e)[:200]}"
                self.browser = None
            time.sleep(0.1)
        raise RuntimeError(f"could not release the page: {self.error}")

    def close(self):
        if self.browser is not None:
            try:
                self.browser.close()   # disconnects only
            except Exception:  # noqa: BLE001
                pass


class NvmlClock:
    """SM/memory clock and utilisation via NVML every interval_s in a thread (independent of nsys)."""

    def __init__(self, interval_s=0.05):
        self.samples, self.interval_s, self._stop = [], interval_s, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        while not self._stop.is_set():
            self.samples.append({"epoch_s": time.time(), "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                                 "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                                 "util": nv.nvmlDeviceGetUtilizationRates(h).gpu})
            time.sleep(self.interval_s)
        nv.nvmlShutdown()

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)
        return self.samples


def nsys(*a, log=None, timeout=120):
    r = subprocess.run([str(NSYS), *a], capture_output=True, text=True, timeout=timeout, errors="replace")
    if log is not None:
        log.write(f"\n$ nsys {' '.join(a)}\n{r.stdout}{r.stderr}")
        log.flush()
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--mode", default="closed", choices=("closed",))
    ap.add_argument("--task", default="infer", choices=("infer", "train"))
    ap.add_argument("--batch", type=int, default=None, help="default: 1 (infer) / the spec batch (train)")
    ap.add_argument("--interval-ms", type=float, default=1000)
    ap.add_argument("--start-at-run", type=int, default=10, help="start collection after this many inferences")
    ap.add_argument("--start-at-s", type=float, default=None,
                    help="instead: start collection this many seconds after the first inference")
    ap.add_argument("--measure-runs", type=int, default=200,
                    help="study phase: stop collecting after --start-at-run warm-up + this many runs (as the "
                         "latency stage)")
    ap.add_argument("--window-s", type=float, default=60.0,
                    help="collection length (sustained phase), or the cap for the study phase")
    ap.add_argument("--trace", default="dx12,dx11", help="nsys --trace value")
    ap.add_argument("--gpu-metrics", action="store_true", help="sample GPU metrics (clocks, GR/SM active)")
    ap.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    ap.add_argument("--flags-file", default=None, help="JSON list of Chrome flags to use instead (diagnostics)")
    ap.add_argument("--plain-flags", action="store_true",
                    help="use only benchkit CHROME_FLAGS instead of the study harness's full command line")
    ap.add_argument("--keep-gpu-sandbox", action="store_true", help="do not add --disable-gpu-sandbox")
    ap.add_argument("--no-nsys", action="store_true",
                    help="control: same page, flags and NVML sampling, without nsys (no trace written)")
    ap.add_argument("--load-timeout-s", type=float, default=120)
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. r1)")
    ap.add_argument("--out", default=str(ROOT / "nsight" / "out" / "nsys"))
    args = ap.parse_args()

    cfg = {k: v for k, v in CONFIGS[args.config].items() if k != "diskcache"}
    phase = f"t{int(args.start_at_s)}s" if args.start_at_s is not None else f"run{args.start_at_run}"
    mode = args.mode + (f"{int(args.interval_ms)}ms" if args.mode == "paced" else "")
    name = f"{args.model}__{args.config}__{args.task}__{mode}__{phase}"
    out = Path(args.out) / f"{name}__{time.strftime('%Y%m%d_%H%M%S')}{('_' + args.tag) if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)

    # Right after a previous capture the GPU can still be boosted; give it up to 30 s to settle.
    lock = envlock.verify(scope="launch")
    t_wait = time.time()
    clk_before = gpu_clock_check()
    while not clk_before[0] and time.time() - t_wait < 30:
        time.sleep(2)
        clk_before = gpu_clock_check()
    if not clk_before[0]:
        sys.exit(f"GPU clocks look locked before the capture: {clk_before[1]}")

    udd = tempfile.mkdtemp(prefix="trainbench-nsys-")
    port = free_port()
    session = f"tfjs{port}"
    study_phase = args.start_at_s is None
    url = (f"http://localhost:4185/nsight/nsight.html?model={quote(args.model)}&config={quote(args.config)}"
           f"&task={args.task}" + (f"&batch={args.batch}" if args.batch else "") +
           f"&wait={int(study_phase)}&cfg={quote(json.dumps(cfg))}")
    extra = [] if args.keep_gpu_sandbox else ["--disable-gpu-sandbox"]
    base = (json.loads(Path(args.flags_file).read_text(encoding="utf-8")) if args.flags_file
            else CHROME_FLAGS if args.plain_flags else study_chrome_args())
    chrome = [find_chrome_dev(), *base, *extra, f"--remote-debugging-port={port}", f"--user-data-dir={udd}", url]
    trace_opts = ["--trace", args.trace]   # --sample/--cpuctxsw belong to `nsys start`
    if "dx12" in args.trace:
        trace_opts += ["--dx12-gpu-workload", "individual", "--dx12-wait-calls", "true"]
    launch = ["launch", f"--session-new={session}", *trace_opts, *chrome]
    start = ["start", f"--session={session}", "--sample", "none", "--cpuctxsw", "none",
             "--output", str(out / "report"), "--export", "sqlite", "--force-overwrite", "true"]
    if args.gpu_metrics:
        start += ["--gpu-metrics-devices", "all", "--gpu-metrics-frequency", str(args.gpu_metrics_frequency)]
    (out / "command.json").write_text(json.dumps({"launch": launch, "start": start, "chrome": chrome}, indent=1),
                                      encoding="utf-8")

    log = open(out / "nsys.log", "w", encoding="utf-8", errors="replace")
    nvml = NvmlClock().start()
    proc = subprocess.Popen(chrome if args.no_nsys else [str(NSYS), *launch], stdout=log, stderr=subprocess.STDOUT)
    t_launch = time.time()
    window = {}
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        poller = PagePoller(pw, port)
        try:
            # 1) wait for the phase start: study phase = page loaded and waiting (wait=1); collection
            #    starts before warm-up and the first --start-at-run runs are excluded as warm-up.
            while True:
                n = poller.fetch()
                if study_phase and poller.meta is not None:
                    break
                if not study_phase and n and time.time() - poller.runs[0]["start_epoch_ms"] / 1e3 >= args.start_at_s:
                    break
                if time.time() - t_launch > args.load_timeout_s:
                    raise RuntimeError(f"page did not reach the start condition: runs={n}, error={poller.error}")
                time.sleep(0.1)
            # 2) collect
            window["start_request_epoch_s"] = time.time()
            window["runs_at_start_request"] = len(poller.runs)
            if not args.no_nsys:
                r = nsys(*start, log=log)
                window["start_rc"] = r.returncode
            window["started_epoch_s"] = time.time()
            window["skip_runs"] = args.start_at_run if study_phase else 0
            if study_phase:
                poller.go()
            t_end = window["started_epoch_s"] + args.window_s
            target = args.start_at_run + args.measure_runs
            while time.time() < t_end:
                n = poller.fetch()
                if study_phase and n >= target:
                    break
                time.sleep(0.05 if study_phase else 0.25)
            window["runs_at_stop_request"] = len(poller.runs)
            window["stop_request_epoch_s"] = time.time()
            if not args.no_nsys:
                r = nsys("stop", f"--session={session}", log=log, timeout=600)
                window["stop_rc"] = r.returncode
            window["stopped_epoch_s"] = time.time()
            poller.fetch()   # Chrome is still running: the last runs of the window
        except Exception as e:  # noqa: BLE001
            window["error"] = f"{type(e).__name__}: {e}"
        finally:
            poller.close()
            if not args.no_nsys:
                nsys("shutdown", f"--session={session}", log=log, timeout=60)
            kill_profile(udd)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    log.close()
    pd.DataFrame(nvml.stop()).to_csv(out / "nvml_clock.csv", index=False)
    (out / "page_stats.json").write_text(json.dumps({"stats": {**(poller.meta or {}), "runs": poller.runs},
                                                     "env_lock_sha256": lock["sha256"],
                                                     "error": poller.error, "window": window,
                                                     "launch_epoch_s": t_launch, "duration_s": args.window_s},
                                                    indent=0), encoding="utf-8")
    clk_after = gpu_clock_check()
    (out / "clock_check.json").write_text(json.dumps({"before": clk_before[1], "after": clk_after[1],
                                                      "after_ok": clk_after[0]}, indent=1), encoding="utf-8")
    print(f"{name}: runs={len(poller.runs)} window={window} page_err={poller.error}")
    print("files:", sorted(p.name for p in out.iterdir()))
    print("clock after:", clk_after[1], "" if clk_after[0] else "  <-- LOCKED?")
    if not (out / "report.sqlite").exists() and not args.no_nsys:
        print("NO report.sqlite; nsys log tail:\n" + "\n".join(
            (out / "nsys.log").read_text(encoding="utf-8", errors="replace").splitlines()[-20:]))


if __name__ == "__main__":
    main()
