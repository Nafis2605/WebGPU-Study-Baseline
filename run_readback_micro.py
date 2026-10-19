"""
Readback microbenchmark driver (port of the sibling TF.js study's, itself the ONNX study's): GPU -> JS copy time per raw API path and
payload size, plus TF.js data()/dataSync() on each GPU backend (src/readback_micro.js). One fresh
Chrome Dev process per (trial, backend); the raw API paths run in the first backend's launch only.

Output (<output-dir>/<timestamp>/): env.json, readback_micro.csv, experiments.jsonl

Usage:
    npm run bench:serve
    python run_readback_micro.py --trials 3 --reps 20 --backends webgl webgpu
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

from benchkit.chrome import CHROME_FLAGS, chrome_page, find_chrome_dev
from benchkit.firefox import FIREFOX_PREFS, find_firefox, firefox_page
from benchkit import layout
from benchkit.envinfo import host_info
from benchkit import envlock  # noqa: E402
from benchkit.results import ResultWriter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:4185/")
    p.add_argument("--browser", default="chrome", choices=("chrome", "firefox"))
    p.add_argument("--backends", nargs="+", default=["webgl", "webgpu"], help="TF.js backends for the TF.js paths")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 4, 16, 64], help="payload sizes in MiB")
    p.add_argument("--cooldown-s", type=float, default=5.0)
    p.add_argument("--executable-path", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output-dir", default=None, help="default: results/<device>/<browser>/adhoc/readback_micro")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = str(layout.run_dir(args.browser, "adhoc") / "readback_micro")
    return args


async def main():
    args = parse_args()
    pinned = find_firefox() if args.browser == "firefox" else find_chrome_dev()
    chrome = args.executable_path or pinned
    if not chrome:
        sys.exit(f"Pinned {args.browser} not found (tools/; see README).")
    if Path(chrome).resolve() != Path(pinned).resolve():
        sys.exit(f"Only the pinned {args.browser} may be used (got {chrome})")
    lock = envlock.verify(scope="launch")
    out = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ResultWriter(out)
    writer.write_json("env.json", {"host": host_info(), "browser": args.browser, "chrome_flags": CHROME_FLAGS,
                                   **({"firefox_prefs": FIREFOX_PREFS} if args.browser == "firefox" else {}),
                                   "args": vars(args),
                                   "env_lock_sha256": lock["sha256"]})
    print(f"Output: {out}")
    rows = []
    async with async_playwright() as pw:
        for trial in range(1, args.trials + 1):
            for k, backend in enumerate(args.backends):
                envlock.verify(scope="launch")
                log = out / "logs" / f"t{trial}_{backend}.log"
                launch = (firefox_page(chrome, args.url, args.headless, log_path=log) if args.browser == "firefox"
                          else chrome_page(pw, chrome, args.url, args.headless, log_path=log))
                async with launch as (page, version):
                    res = await page.evaluate("cfg => window.__bench.readbackMicro(cfg)",
                                              {"sizesMiB": args.sizes_mib, "reps": args.reps, "warmup": args.warmup,
                                               "backend": backend, "raw": k == 0})
                res.update({"browser": args.browser, "trial": trial, "backend": backend, "browser_version": version,
                            "env_lock_sha256": lock["sha256"]})
                writer.append(res)
                rows += [{"trial": trial, "launch_backend": backend, **r} for r in res["records"]]
                errs = {kk: v for kk, v in res["env"].items() if "error" in kk or "incomplete" in kk or "mismatch" in kk}
                print(f"trial {trial} {backend}: {len(res['records'])} records {errs or ''}", flush=True)
                await asyncio.sleep(args.cooldown_s)
    pd.DataFrame(rows).to_csv(out / "readback_micro.csv", index=False)
    print(f"Done -> {out / 'readback_micro.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
