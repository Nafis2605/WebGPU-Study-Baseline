"""
Catalogue of everything under results/: one row per run, written to results/INDEX.json and results/INDEX.csv, so an
analysis (or an agent) can find the right folder without walking the tree.

Layout (benchkit/layout.py):  results/<device>/<browser>-<version>/<run>/   with run = full | verify | diagnostics |
adhoc/<timestamp> | _dev/<name>;  results/<device>/native/<run>/;  results/<device>/probes/.

The scan only reads. Missing device.json / browser.json / run.json files are created from the data already in the
run (env.json, stage_done.json, sessions.csv) and marked "backfilled": true; nothing existing is changed.

Per run folder (a flight): <stage>/<timestamp>/{env.json, experiments.jsonl (raw, source of truth), sessions.csv (one row
per launch), runs.csv (one row per timed record), kernels.csv, telemetry CSVs, io/, logs/}, <stage>/stage_done.json,
analysis/ (cross-stage tables), env.lock.json (the environment lock the run was verified against).

Usage:
    python scripts/index_results.py [--results DIR]
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit import layout  # noqa: E402

SKIP = {"_guards", "_interrupted", "analysis"}


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def run_folders(browser_dir):
    """Run folders under a browser folder: full, verify, diagnostics, adhoc/<ts>, _dev/<name>."""
    for child in sorted(p for p in browser_dir.iterdir() if p.is_dir()):
        if child.name in ("adhoc", "_dev"):
            for sub in sorted(p for p in child.iterdir() if p.is_dir()):
                yield child.name, sub
        else:
            yield child.name, child


def stage_dirs(run):
    """Result folders of a run: every folder below it that holds sessions.csv (or, for readback_micro, experiments.jsonl
    only), skipping _guards, _interrupted and analysis."""
    found = set()
    for name in ("sessions.csv", "experiments.jsonl"):
        for f in run.rglob(name):
            rel = f.relative_to(run).parts
            if any(part in SKIP or part.startswith("_") for part in rel[:-1]):
                continue
            found.add(f.parent)
    return sorted(found)


def count_launches(run):
    launches = failed = known = 0
    for d in stage_dirs(run):
        if (d / "sessions.csv").exists():
            try:
                s = pd.read_csv(d / "sessions.csv", usecols=lambda c: c in ("status", "known_gap"), low_memory=False)
            except Exception:  # noqa: BLE001
                continue
            launches += len(s)
            bad = s[s["status"] != "ok"] if "status" in s else s.iloc[0:0]
            failed += len(bad)
            if "known_gap" in bad:
                known += int(bad["known_gap"].notna().sum())
        elif d.name != "diagnostics" and (d / "experiments.jsonl").exists() and (d / "env.json").exists():
            with open(d / "experiments.jsonl", encoding="utf-8") as f:   # readback_micro: one record per launch, no sessions.csv
                launches += sum(1 for line in f if line.strip())
    return launches, failed, known


def first_env(root):
    for p in sorted(root.rglob("env.json"))[:1]:
        return read_json(p)
    return None


def backfill_run(run, kind, browser_id, device):
    f = run / "run.json"
    if f.exists():
        return read_json(f)
    dones = [read_json(p) for p in sorted(run.glob("*/stage_done.json"))]
    dones = [d for d in dones if d]
    env = first_env(run) or {}
    host = env.get("host") or {}
    mtimes = [p.stat().st_mtime for p in run.rglob("stage_done.json")] or [run.stat().st_mtime]
    info = {"run": run.name, "kind": kind, "browser": browser_id, "device": device, "backfilled": True,
            "started_at": env.get("created_at"),
            "finished_at": datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds"),
            "harness": {"commit": host.get("git_commit"), "dirty": host.get("git_dirty"), "patch_sha256": None},
            "env_lock_sha256": dones[0].get("env_lock_sha256") if dones else env.get("env_lock_sha256"),
            "stage_status": {p.parent.name: (read_json(p) or {}).get("status") for p in sorted(run.glob("*/stage_done.json"))}}
    f.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return info


def backfill_device(dev_dir):
    f = dev_dir / "device.json"
    if f.exists():
        return read_json(f)
    host = (first_env(dev_dir) or {}).get("host") or {}
    info = {"id": dev_dir.name, "backfilled": True, "os": host.get("os"), "cpu": host.get("cpu"),
            "cpu_logical_cores": host.get("cpu_logical_cores"), "ram_total_gb": host.get("ram_total_gb"),
            "power_plan": host.get("power_plan"), "nvidia": host.get("nvidia")}
    f.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return info


def backfill_browser(bdir):
    f = bdir / "browser.json"
    if f.exists():
        return read_json(f)
    env = first_env(bdir) or {}
    name, _, ver = bdir.name.partition("-")
    info = {"id": bdir.name, "browser": name, "version": ver, "backfilled": True,
            "executable": env.get("chrome_executable"), "flags": env.get("chrome_flags") if name == "chrome" else None,
            "prefs": env.get("firefox_prefs")}
    f.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(layout.RESULTS))
    args = ap.parse_args()
    results = Path(args.results)
    rows, devices = [], {}
    for dev in sorted(p for p in results.iterdir() if p.is_dir()):
        devices[dev.name] = backfill_device(dev)
        for child in sorted(p for p in dev.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if child.name == "probes":
                rows.append({"device": dev.name, "browser": "", "run": "probes", "kind": "probes",
                             "path": child.relative_to(results).as_posix(), "launches": "", "failed": "", "known_gap": ""})
                continue
            if child.name == "native":
                runs = []
                for r in sorted(p for p in child.iterdir() if p.is_dir()):
                    if r.name in ("adhoc", "_dev"):
                        runs += [(r.name, sub) for sub in sorted(p for p in r.iterdir() if p.is_dir())]
                    else:
                        runs.append(("native", r))
                browser_id = ""
            else:
                backfill_browser(child)
                runs, browser_id = list(run_folders(child)), child.name
            for kind_dir, run in runs:
                kind = "flight" if run.name in ("full", "verify") else ("diagnostics" if run.name == "diagnostics" else
                                                                       ("adhoc" if kind_dir == "adhoc" else ("dev" if kind_dir == "_dev" else kind_dir)))
                meta = backfill_run(run, kind, browser_id, dev.name) or {}
                launches, failed, known = count_launches(run)
                h = meta.get("harness") or {}
                rows.append({"device": dev.name, "browser": browser_id, "run": f"{kind_dir}/{run.name}" if kind_dir in ("adhoc", "_dev") else run.name,
                             "kind": kind, "path": run.relative_to(results).as_posix(), "launches": launches, "failed": failed,
                             "known_gap": known, "stages": len(meta.get("stage_status") or {}),
                             "started_at": meta.get("started_at"), "finished_at": meta.get("finished_at"),
                             "harness_commit": h.get("commit"), "harness_dirty": h.get("dirty"),
                             "env_lock_sha256": meta.get("env_lock_sha256")})
    index = {"generated_at": datetime.now().isoformat(timespec="seconds"), "results_root": results.name, "devices": devices,
             "runs": rows, "note": "failed counts launches whose status is not ok; known_gap is the subset that is a recorded "
                                   "TF.js coverage gap (CNN training on wasm)"}
    (results / "INDEX.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    cols = ["device", "browser", "run", "kind", "path", "launches", "failed", "known_gap", "stages", "started_at", "finished_at",
            "harness_commit", "harness_dirty", "env_lock_sha256"]
    with open(results / "INDEX.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs on {len(devices)} device(s) -> {results / 'INDEX.json'}")
    for r in rows:
        print(f"  {r['device']:16s} {r['browser']:22s} {r['run']:34s} launches {r['launches']!s:>6s} failed {r['failed']!s:>4s}")


if __name__ == "__main__":
    main()
