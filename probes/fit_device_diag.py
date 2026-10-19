"""
Diagnostic for the Firefox WebGPU model.fit failures ("Context lost" / "Invalid buffer", train_fit stage of the
full Firefox flight). Not part of any study stage and not measurement data: results go to
results/<device>/firefox-<ver>/diagnostics/ and nothing in the study folders is touched.

It reruns one (model, config) train_fit experiment with the study's own settings (run_benchmark.page_cfg) and,
without changing the bundle, hooks the WebGPU device from outside the page: GPUAdapter.requestDevice is wrapped
after the page has loaded and before runExperiment creates the backend, so the page code is unmodified. The hook
records device.lost (reason, message), uncapturederror events, and the TF.js memory state at the end.
Firefox's own log (MOZ_LOG) goes into the geckodriver log; the process tree is sampled before and after, and the
profile's crash directory is listed before the profile is deleted.

Usage:
    python probes/fit_device_diag.py --models R3 D3 T3 --config webgpu [--extra-env KEY=VALUE ...] [--tag label]
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil  # noqa: E402
import subprocess
import threading
import time

from benchkit import layout  # noqa: E402
from benchkit.chrome import chrome_page, find_chrome_dev  # noqa: E402
from benchkit.firefox import find_firefox, firefox_page  # noqa: E402
from run_benchmark import CONFIGS, load_models, page_cfg, parse_args  # noqa: E402

GC_MB = 0
TRAIN_BATCH = 0
HOOK_JS = """() => {
  if (window.__gcPressure) {
    // Intervention for the diagnosis only: allocate and drop a large ArrayBuffer every 50 ms so the JS engine collects.
    window.__gcTicks = 0;
    setInterval(() => { try { const a = new ArrayBuffer(window.__gcPressure * 1048576); new Uint8Array(a)[0] = 1; window.__gcTicks++; } catch (e) {} }, 50);
  }
  window.__diag = { events: [], devices: 0, buf: { created: 0, destroyed: 0, live_bytes: 0, peak_bytes: 0, largest: 0, recent: [] } };
  const t = () => performance.now();
  const B = window.__diag.buf;
  const origCB = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (desc) {
    const b = origCB.call(this, desc);
    B.created++; B.live_bytes += desc.size; B.peak_bytes = Math.max(B.peak_bytes, B.live_bytes); B.largest = Math.max(B.largest, desc.size);
    B.recent.push(desc.size); if (B.recent.length > 8) B.recent.shift();
    b.__sz = desc.size;
    return b;
  };
  B.bind_groups = 0; B.pipelines = 0; B.pass_dispatches = 0;
  const origBG = GPUDevice.prototype.createBindGroup;
  B.bg_entries = 0; B.bg_finalized = 0;
  const fr = new FinalizationRegistry(() => { B.bg_finalized++; });
  GPUDevice.prototype.createBindGroup = function (d) {
    B.bind_groups++; B.bg_entries += d.entries.length;
    const g = origBG.call(this, d); fr.register(g, 0); return g;
  };
  const origPL = GPUDevice.prototype.createComputePipeline;
  GPUDevice.prototype.createComputePipeline = function (d) { B.pipelines++; return origPL.call(this, d); };
  const origDisp = GPUComputePassEncoder.prototype.dispatchWorkgroups;
  GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...a) { B.pass_dispatches++; return origDisp.apply(this, a); };
  const origD = GPUBuffer.prototype.destroy;
  GPUBuffer.prototype.destroy = function () { if (!this.__gone) { this.__gone = true; B.destroyed++; B.live_bytes -= (this.__sz || this.size); } return origD.call(this); };
  const orig = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc) {
    const d = await orig.call(this, desc);
    const id = ++window.__diag.devices;
    window.__diag.events.push({ t: t(), type: "device_created", id, features: [...d.features].length,
      maxBufferSize: d.limits.maxBufferSize, maxStorageBufferBindingSize: d.limits.maxStorageBufferBindingSize });
    d.lost.then(info => window.__diag.events.push({ t: t(), type: "lost", id, reason: info.reason, message: info.message }));
    d.addEventListener("uncapturederror", e => window.__diag.events.push({ t: t(), type: "uncaptured", id, live_bytes: B.live_bytes, peak_bytes: B.peak_bytes, created: B.created, destroyed: B.destroyed, recent: [...B.recent],
      bind_groups: B.bind_groups, bg_entries: B.bg_entries, bg_finalized: B.bg_finalized, pipelines: B.pipelines, dispatches: B.pass_dispatches,
      ctor: e.error && e.error.constructor && e.error.constructor.name, message: e.error && e.error.message }));
    return d;
  };
  return true;
}"""


def procs(profile_dir):
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() == "firefox.exe":
                cmd = p.info["cmdline"] or []
                if profile_dir in " ".join(cmd) or "-contentproc" in cmd:
                    kind = cmd[-1] if "-contentproc" in cmd else "parent"
                    out.append((p.info["pid"], kind))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


class GpuMem:
    """Samples whole-GPU memory.used (MiB) with nvidia-smi twice a second while a launch runs."""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            try:
                v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.split(",")
                self.samples.append((round(time.time(), 2), int(v[0]), int(v[1])))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(0.5)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)
        used = [s[1] for s in self.samples]
        return {"baseline_mib": used[0] if used else None, "peak_mib": max(used) if used else None,
                "total_mib": self.samples[0][2] if self.samples else None, "n": len(used)}


async def run_one(model, config, trial, tag, out_dir, browser="firefox", pw=None):
    args = parse_args(["--mode", "train_fit", "--browser", "firefox", "--fit-epochs", "2"]
                      + (["--train-batch", str(TRAIN_BATCH)] if TRAIN_BATCH else []))
    cfg = page_cfg(args, model, config, trial)
    log = out_dir / f"{tag}_{model['name']}_{config}.log"
    rec = {"model": model["name"], "config": config, "tag": tag, "at": datetime.now().isoformat(timespec="seconds")}
    holder = {}
    if browser == "chrome":
        launch = chrome_page(pw, find_chrome_dev(), args.url, False, log_path=log,
                             on_launch=lambda d: holder.setdefault("profile", d))
    else:
        launch = firefox_page(find_firefox(), args.url, False, log_path=log,
                              on_launch=lambda d: holder.setdefault("profile", d))
    async with launch as (page, version):
        rec["browser"] = browser
        rec["browser_version"] = version
        rec["procs_before"] = procs(holder["profile"]) if browser == "firefox" else []
        if GC_MB:
            await page.evaluate(f"() => {{ window.__gcPressure = {GC_MB}; return true; }}")
        await page.evaluate(HOOK_JS)
        gm = GpuMem()
        gm.start()
        t0 = asyncio.get_running_loop().time()
        try:
            r = await asyncio.wait_for(page.evaluate("cfg => window.__bench.runExperiment(cfg)", cfg), timeout=1800)
            rec["status"] = r.get("status")
            rec["failure"] = r.get("failure")
            rec["runs"] = [(x["phase"], round(x["dur_ms"], 1)) for x in r.get("runs", []) if "dur_ms" in x]
        except Exception as e:  # noqa: BLE001
            rec["status"] = "driver_exception"
            rec["failure"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = round(asyncio.get_running_loop().time() - t0, 1)
        rec["gpu_mem"] = gm.stop()
        try:
            rec["diag"] = await page.evaluate("() => window.__diag")
        except Exception as e:  # noqa: BLE001
            rec["diag"] = f"unreadable: {type(e).__name__}: {e}"
        rec["procs_after"] = procs(holder["profile"]) if browser == "firefox" else []
        crashes = Path(holder["profile"]) / "crashes"
        rec["crash_files"] = [str(p.name) for p in crashes.rglob("*")] if crashes.exists() else []
    return rec


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--config", default="webgpu", choices=list(CONFIGS))
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--browser", default="firefox", choices=("firefox", "chrome"))
    ap.add_argument("--train-batch", type=int, default=0, help="diagnostic: override the spec's training batch")
    ap.add_argument("--gc-pressure-mb", type=int, default=0, help="diagnostic intervention: allocate+drop this many MB every 50 ms in the page")
    ap.add_argument("--extra-env", nargs="*", default=[], help="KEY=VALUE set for the Firefox launch (e.g. MOZ_LOG)")
    a = ap.parse_args()
    global GC_MB, TRAIN_BATCH
    GC_MB = a.gc_pressure_mb
    TRAIN_BATCH = a.train_batch
    for kv in a.extra_env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    out_dir = layout.run_dir("firefox", "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        for m in load_models(a.models):
            rec = await run_one(m, a.config, 1, a.tag, out_dir, a.browser, pw)
            with open(out_dir / f"{a.tag}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(m["name"], rec["status"], rec.get("seconds"), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
