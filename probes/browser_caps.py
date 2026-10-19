"""
Capability probe for the cross-browser replication: one fresh launch per browser, recording what the page can
use (no timing data): the page's probeEnv() (WebGPU adapter and features, WebGL renderer and timer query), the
measured performance.now() step, and which scheduling / memory / input-timing APIs exist. Per-backend init info
(wasm threads/SIMD, TF.js flags) is in every experiment record. Results: results/<device>/probes/browser_caps.jsonl.

Usage:
    python probes/browser_caps.py --browser chrome firefox
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playwright.async_api import async_playwright  # noqa: E402

from benchkit import layout  # noqa: E402
from benchkit.chrome import chrome_page, find_chrome_dev  # noqa: E402
from benchkit.firefox import find_firefox, firefox_page  # noqa: E402

API_JS = """() => {
  const steps = new Set(); let a = performance.now();
  for (let i = 0; i < 3e6 && steps.size < 8; i++) { const b = performance.now(); if (b !== a) { steps.add(+(b - a).toFixed(5)); a = b; } }
  const has = (o, k) => { try { return k in o; } catch { return false; } };
  return {
    performance_now_steps_ms: [...steps].sort((x, y) => x - y),
    shared_array_buffer: typeof SharedArrayBuffer === "function",
    scheduler_yield: !!globalThis.scheduler?.yield,
    scheduler_post_task: !!globalThis.scheduler?.postTask,
    measure_user_agent_specific_memory: has(performance, "measureUserAgentSpecificMemory"),
    performance_memory: has(performance, "memory"),
    event_timing: PerformanceObserver.supportedEntryTypes.includes("event"),
    long_animation_frame: PerformanceObserver.supportedEntryTypes.includes("long-animation-frame"),
    longtask: PerformanceObserver.supportedEntryTypes.includes("longtask"),
    webgpu: !!navigator.gpu,
    navigator_webdriver: navigator.webdriver,
  };
}"""


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--browser", nargs="+", default=["chrome", "firefox"], choices=("chrome", "firefox"))
    ap.add_argument("--url", default="http://localhost:4185/")
    args = ap.parse_args()
    out = layout.probes_dir()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        for browser in args.browser:
            log = out / "logs" / f"browser_caps_{browser}.log"
            launch = (firefox_page(find_firefox(), args.url, False, log_path=log) if browser == "firefox"
                      else chrome_page(pw, find_chrome_dev(), args.url, False, log_path=log))
            rec = {"probe": "browser_caps", "browser": browser, "at": datetime.now().isoformat(timespec="seconds")}
            try:
                async with launch as (page, version):
                    rec["browser_version"] = version
                    rec["env"] = await page.evaluate("() => window.__bench.probeEnv()")
                    rec["apis"] = await page.evaluate(API_JS)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            with open(out / "browser_caps.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
