"""
Nsight Systems capture of the TF.js page in Firefox (diagnostic; counterpart of nsight/run_nsys.py, same output files, so
analysis/nsys_summary.py reads both).

Firefox is started under `nsys launch` with the study's profile prefs and --remote-debugging-port, which opens a WebDriver BiDi
endpoint (ws://127.0.0.1:<port>/session). The script talks to it with a minimal client (websocket-client): it finds the
nsight.html tab, polls window.__nsight.stats() and releases the page (window.__nsight.go()) once `nsys start` has been issued,
exactly as run_nsys.py does over the DevTools protocol. nsys follows Firefox's child processes; Firefox's GPU process (wgpu on
Direct3D 12 for WebGPU, ANGLE on Direct3D 11 for WebGL) is only traced when its sandbox is off (MOZ_DISABLE_GPU_SANDBOX=1,
set here unless --keep-gpu-sandbox) and the launcher process is bypassed (MOZ_LAUNCHER_PROCESS=1).

Output: <--out>/<model>__<config>__<task>__closed__run<N>__<timestamp>[_tag]/ with report.nsys-rep, report.sqlite,
page_stats.json, nvml_clock.csv, command.json, nsys.log, clock_check.json, firefox_processes.json (process tree at collection start).

Usage (npm run bench:serve running on 4185):
    python nsight/run_nsys_firefox.py --model D3 --config webgpu --task infer --trace dx12 --gpu-metrics
    python nsight/run_nsys_firefox.py --model L3 --config webgpu --task train --start-at-run 15 --measure-runs 100 --window-s 60
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nsys as base  # noqa: E402  (Chrome script: shared helpers)
from benchkit import envlock, layout  # noqa: E402
from benchkit.firefox import FIREFOX_PREFS, find_firefox  # noqa: E402
from benchkit.paths import NSYS  # noqa: E402
from run_benchmark import CONFIGS  # noqa: E402

import websocket  # noqa: E402  (websocket-client, installed with Selenium)


class BidiPoller:
    """Minimal WebDriver BiDi client for the nsight page: stats(from) and go()."""

    def __init__(self, port):
        self.port, self.ws, self.msg_id, self.context = port, None, 0, None
        self.runs, self.meta, self.error = [], None, None
        self.errors = []   # distinct client errors in order, kept in page_stats.json for diagnosis

    def _send(self, method, params, timeout=40):
        self.msg_id += 1
        mid = self.msg_id
        self.ws.settimeout(timeout)
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("id") == mid:
                if reply.get("type") == "error":
                    raise RuntimeError(f"{method}: {reply.get('error')}: {reply.get('message')}")
                return reply.get("result")

    def _drop(self):
        """End the BiDi session (Firefox allows one) and close the socket, so a reconnect can create a new one."""
        try:
            if self.ws is not None:
                try:
                    self._send("session.end", {}, timeout=3)
                except Exception:  # noqa: BLE001
                    pass
                self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        self.ws, self.context = None, None

    def _connect(self):
        self.ws = websocket.create_connection(f"ws://127.0.0.1:{self.port}/session", timeout=5, suppress_origin=True)
        try:
            self._send("session.new", {"capabilities": {}})
        except Exception:
            self._drop()
            raise
        self.context = None

    def _find_context(self):
        tree = self._send("browsingContext.getTree", {})
        for ctx in tree.get("contexts", []):
            if "nsight.html" in ctx.get("url", ""):
                self.context = ctx["context"]
                return
        self.context = None

    def _call(self, body, *args):
        if self.ws is None:
            self._connect()
        if self.context is None:
            self._find_context()
        if self.context is None:
            return None
        r = self._send("script.callFunction", {
            "functionDeclaration": "(...a) => { const v = (" + body + ")(...a); return JSON.stringify(v === undefined ? null : v); }",
            "awaitPromise": False,
            "target": {"context": self.context}, "arguments": [{"type": "number", "value": x} for x in args]})
        if r.get("type") != "success":
            raise RuntimeError(f"script exception: {r.get('exceptionDetails', {}).get('text')}")
        value = r.get("result", {}).get("value")
        return json.loads(value) if isinstance(value, str) else None

    def fetch(self):
        try:
            v = self._call("(f) => window.__nsight && (window.__nsight.stats ? window.__nsight.stats(f) : "
                           "{ready: window.__nsight.ready, error: window.__nsight.error})", len(self.runs))
            if v and "runs" in v:
                self.runs += v["runs"]
                self.meta = {k: v[k] for k in v if k != "runs"}
                self.error = None
            elif v and v.get("error"):
                self.error = v["error"]
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            if self.error not in self.errors:
                self.errors.append(self.error)
            if isinstance(e, (websocket.WebSocketException, OSError)):
                self._drop()
        return len(self.runs)

    def go(self, tries=30):
        for _ in range(tries):
            try:
                if self._call("() => !!(window.__nsight && window.__nsight.go && (window.__nsight.go(), true))"):
                    return True
            except Exception as e:  # noqa: BLE001
                self.error = f"go: {type(e).__name__}: {str(e)[:200]}"
                if isinstance(e, (websocket.WebSocketException, OSError)):
                    self._drop()
            time.sleep(0.2)
        raise RuntimeError(f"could not release the page: {self.error}")

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def kill_firefox(udd):
    """Kill the Firefox instance started with this profile and all its children."""
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() == "firefox.exe" and udd in " ".join(p.info["cmdline"] or []):
                for c in p.children(recursive=True):
                    c.kill()
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def process_tree(udd):
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if (p.info["name"] or "").lower() == "firefox.exe" and (udd in " ".join(cmd) or "-contentproc" in cmd):
                out.append({"pid": p.info["pid"], "ppid": p.ppid(), "type": cmd[-1] if "-contentproc" in cmd else "parent"})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--task", default="infer", choices=("infer", "train"))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--start-at-run", type=int, default=10)
    ap.add_argument("--measure-runs", type=int, default=200)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--trace", default="dx12,dx11")
    ap.add_argument("--gpu-metrics", action="store_true")
    ap.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    ap.add_argument("--system-wide", action="store_true",
                    help="nsys --system-wide=true: inject through Windows hooks instead of following child processes")
    ap.add_argument("--keep-launcher", action="store_true", help="do not set MOZ_LAUNCHER_PROCESS=1 (the GPU process is then not traced)")
    ap.add_argument("--keep-gpu-sandbox", action="store_true", help="do not set MOZ_DISABLE_GPU_SANDBOX=1")
    ap.add_argument("--load-timeout-s", type=float, default=120)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(layout.run_dir("firefox", "_dev/nsight") / "nsys"))
    args = ap.parse_args()

    cfg = {k: v for k, v in CONFIGS[args.config].items() if k != "diskcache"}
    name = f"{args.model}__{args.config}__{args.task}__closed__run{args.start_at_run}"
    out = Path(args.out) / f"{name}__{time.strftime('%Y%m%d_%H%M%S')}{('_' + args.tag) if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)

    lock = envlock.verify(scope="launch")
    t_wait = time.time()
    clk_before = base.gpu_clock_check()
    while not clk_before[0] and time.time() - t_wait < 30:
        time.sleep(2)
        clk_before = base.gpu_clock_check()
    if not clk_before[0]:
        sys.exit(f"GPU clocks look locked before the capture: {clk_before[1]}")

    udd = tempfile.mkdtemp(prefix="trainbench-nsysff-")
    (Path(udd) / "user.js").write_text("".join(f"user_pref({json.dumps(k)}, {json.dumps(v)});\n" for k, v in FIREFOX_PREFS.items()),
                                       encoding="utf-8")
    port = base.free_port()
    session = f"tfjsff{port}"
    url = (f"http://localhost:4185/nsight/nsight.html?model={quote(args.model)}&config={quote(args.config)}"
           f"&task={args.task}" + (f"&batch={args.batch}" if args.batch else "") + f"&wait=1&cfg={quote(json.dumps(cfg))}")
    firefox = [find_firefox(), "-no-remote", "-profile", udd, f"--remote-debugging-port={port}", url]
    trace_opts = ["--trace", args.trace]
    if "dx12" in args.trace:
        trace_opts += ["--dx12-gpu-workload", "individual", "--dx12-wait-calls", "true"]
    inject = ["--system-wide=true"] if args.system_wide else []
    launch = ["launch", f"--session-new={session}", *inject, *trace_opts, *firefox]
    start = ["start", f"--session={session}", "--sample", "none", "--cpuctxsw", "none",
             "--output", str(out / "report"), "--export", "sqlite", "--force-overwrite", "true"]
    if args.gpu_metrics:
        start += ["--gpu-metrics-devices", "all", "--gpu-metrics-frequency", str(args.gpu_metrics_frequency)]
    env = dict(os.environ)
    # firefox.exe normally starts a small launcher process that spawns the real browser process; Nsight follows
    # the launcher but not that child, so the GPU process is never instrumented. The launcher sets
    # MOZ_LAUNCHER_PROCESS for the browser it starts; pre-setting it makes firefox.exe run as the browser itself.
    if not args.keep_launcher:
        env["MOZ_LAUNCHER_PROCESS"] = "1"
    if not args.keep_gpu_sandbox:
        env["MOZ_DISABLE_GPU_SANDBOX"] = "1"
    (out / "command.json").write_text(json.dumps({"launch": launch, "start": start, "firefox": firefox,
                                                  "env": {"MOZ_DISABLE_GPU_SANDBOX": env.get("MOZ_DISABLE_GPU_SANDBOX"),
                                                           "MOZ_LAUNCHER_PROCESS": env.get("MOZ_LAUNCHER_PROCESS")}}, indent=1),
                                      encoding="utf-8")

    log = open(out / "nsys.log", "w", encoding="utf-8", errors="replace")
    nvml = base.NvmlClock().start()
    proc = subprocess.Popen([str(NSYS), *launch], stdout=log, stderr=subprocess.STDOUT, env=env)
    t_launch = time.time()
    window = {}
    poller = BidiPoller(port)
    try:
        # 1) wait until the page has loaded and waits for go() (wait=1)
        while True:
            n = poller.fetch()
            if poller.meta is not None:
                break
            if time.time() - t_launch > args.load_timeout_s:
                raise RuntimeError(f"page did not reach the start condition: runs={n}, error={poller.error}")
            time.sleep(0.2)
        # 2) collect
        window["start_request_epoch_s"] = time.time()
        window["runs_at_start_request"] = len(poller.runs)
        r = base.nsys(*start, log=log)
        window["start_rc"] = r.returncode
        window["started_epoch_s"] = time.time()
        window["skip_runs"] = args.start_at_run
        (out / "firefox_processes.json").write_text(json.dumps(process_tree(udd), indent=1), encoding="utf-8")
        poller.go()
        t_end = window["started_epoch_s"] + args.window_s
        target = args.start_at_run + args.measure_runs
        while time.time() < t_end:
            n = poller.fetch()
            if n >= target:
                break
            time.sleep(0.05)
        window["runs_at_stop_request"] = len(poller.runs)
        window["stop_request_epoch_s"] = time.time()
        r = base.nsys("stop", f"--session={session}", log=log, timeout=600)
        window["stop_rc"] = r.returncode
        window["stopped_epoch_s"] = time.time()
        poller.fetch()
    except Exception as e:  # noqa: BLE001
        window["error"] = f"{type(e).__name__}: {e}"
    finally:
        poller.close()
        base.nsys("shutdown", f"--session={session}", log=log, timeout=60)
        kill_firefox(udd)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    log.close()
    pd.DataFrame(nvml.stop()).to_csv(out / "nvml_clock.csv", index=False)
    (out / "page_stats.json").write_text(json.dumps({"stats": {**(poller.meta or {}), "runs": poller.runs},
                                                     "env_lock_sha256": lock["sha256"], "error": poller.error, "window": window,
                                                     "launch_epoch_s": t_launch, "duration_s": args.window_s, "bidi_errors": poller.errors}, indent=0),
                                         encoding="utf-8")
    clk_after = base.gpu_clock_check()
    (out / "clock_check.json").write_text(json.dumps({"before": clk_before[1], "after": clk_after[1], "after_ok": clk_after[0]}, indent=1),
                                          encoding="utf-8")
    print(f"{name}: runs={len(poller.runs)} window={window} page_err={poller.error}")
    print("files:", sorted(p.name for p in out.iterdir()))
    print("clock after:", clk_after[1], "" if clk_after[0] else "  <-- LOCKED?")
    if not (out / "report.sqlite").exists():
        print("NO report.sqlite; nsys log tail:\n" + "\n".join(
            (out / "nsys.log").read_text(encoding="utf-8", errors="replace").splitlines()[-20:]))
    kill_firefox(udd)


if __name__ == "__main__":
    main()
