"""
Phase-0 probes that need dedicated page code (src/probes.js): one fresh Chrome for Testing launch per
(probe, model, config), results appended to probes/out/<probe>.jsonl. The remaining probes run the real
harness (run_benchmark.py / native/run_native.py) with small counts; see study_report/FEASIBILITY.md.

Usage:
    python probes/run_probes.py --probe syncSemantics --config webgl webgpu wasm-8t --filter D1 L1 T1
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

from benchkit.chrome import chrome_page, find_chrome_dev  # noqa: E402
from run_benchmark import CONFIGS, load_models  # noqa: E402

PROBES = ("syncSemantics", "trainPrecompile", "fitYields")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, choices=PROBES)
    ap.add_argument("--config", nargs="+", default=["wasm-8t", "webgl", "webgpu"], choices=list(CONFIGS))
    ap.add_argument("--filter", nargs="+", required=True)
    ap.add_argument("--url", default="http://localhost:4185/")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    out = ROOT / "probes" / "out"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    chrome = find_chrome_dev()
    async with async_playwright() as pw:
        for m in load_models(args.filter):
            for c in args.config:
                cfg = {**{k: v for k, v in CONFIGS[c].items() if k != "diskcache"}, "configLabel": c, "model": m,
                       "steps": args.steps}
                rec = {"probe": args.probe, "model": m["name"], "config": c,
                       "at": datetime.now().isoformat(timespec="seconds")}
                try:
                    async with chrome_page(pw, chrome, args.url, args.headless,
                                           log_path=out / "logs" / f"{args.probe}_{m['name']}_{c}.log") as (page, ver):
                        rec["result"] = await asyncio.wait_for(
                            page.evaluate(f"cfg => window.__bench.probes.{args.probe}(cfg)", cfg), timeout=1800)
                        rec["browser_version"] = ver
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"
                with open(out / f"{args.probe}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                r = rec.get("result") or {}
                brief = {k: v for k, v in r.items() if not isinstance(v, list)}
                print(f"{args.probe} {m['name']} {c}: {rec.get('error') or json.dumps(brief)[:400]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
