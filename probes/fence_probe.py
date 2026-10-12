"""
Probe: why does TF.js's WebGL fence (gpgpu.createAndWaitForFence) take ~260 ms on an idle queue in the
study's Playwright-launched Chrome for Testing but ~5 ms in an interactive Chrome? Measures, in a study
launch (same flags as run_benchmark.py): visibility / focus, setTimeout(0) and MessageChannel latency, rAF
interval, TF.js's fence, and a raw WebGL2 fence polled with MessageChannel.

Usage:  python probes/fence_probe.py [--headless] [--extra-flag ...]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playwright.async_api import async_playwright  # noqa: E402

from benchkit.chrome import chrome_page, find_chrome_dev  # noqa: E402

JS = """
async () => {
  const tf = window.__bench.tf;
  await tf.setBackend("webgl"); await tf.ready();
  const b = tf.backend(), gl = b.gpgpu.gl;
  const mc = () => new Promise(r => { const c = new MessageChannel(); c.port1.onmessage = r; c.port2.postMessage(0); });
  window.__N = 40;
  const out = {visibility: document.visibilityState, hasFocus: document.hasFocus(), fenceApi: tf.env().getBool("WEBGL_FENCE_API_ENABLED")};
  const time = async (f, n = Number(window.__N || 5)) => { const t = []; for (let i = 0; i < n; i++) { const t0 = performance.now(); await f(); t.push(+(performance.now() - t0).toFixed(2)); } return t; };
  out.setTimeout0 = await time(() => new Promise(r => setTimeout(r, 0)));
  out.messageChannel = await time(mc);
  out.raf = await time(() => new Promise(r => requestAnimationFrame(r)));
  out.tfFence = await time(() => b.gpgpu.createAndWaitForFence());
  out.rawFence = await time(async () => {
    const s = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0); gl.flush();
    while (true) { const st = gl.clientWaitSync(s, 0, 0); if (st === gl.ALREADY_SIGNALED || st === gl.CONDITION_SATISFIED) break; await mc(); }
    gl.deleteSync(s);
  });
  out.rawFenceSetTimeout = await time(async () => {
    const s = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0); gl.flush();
    while (true) { const st = gl.clientWaitSync(s, 0, 0); if (st === gl.ALREADY_SIGNALED || st === gl.CONDITION_SATISFIED) break; await new Promise(r => setTimeout(r, 0)); }
    gl.deleteSync(s);
  });
  return out;
}
"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--extra-flag", nargs="*", default=[])
    ap.add_argument("--bring-to-front", action="store_true")
    args = ap.parse_args()
    async with async_playwright() as pw:
        async with chrome_page(pw, find_chrome_dev(), "http://localhost:4185/", args.headless,
                               extra_flags=args.extra_flag) as (page, ver):
            if args.bring_to_front:
                await page.bring_to_front()
            r = await page.evaluate(JS)
    print(json.dumps({"browser": ver, "headless": args.headless, "flags": args.extra_flag, **r}))


if __name__ == "__main__":
    asyncio.run(main())
