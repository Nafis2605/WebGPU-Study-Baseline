"""
Where results and reports live. One place, so a run on another device or browser never overlaps an existing one.

    results/<device-id>/<browser-id>/<run>/      e.g. results/win11-rtx3060/chrome-157.0.8081.0/full
    results/<device-id>/native/<run>/            native TensorFlow runs (browser-independent)
    results/<device-id>/probes/                  device-level probe outputs (records carry their browser)
    results/<device-id>/_ops/                    logs of how runs were launched
    reports/<device-id>/<browser-id>/            analyses, tables, figures, report scripts (git-ignored, local)
    reports/<device-id>/compare/                 cross-browser inputs (margins) and notes

Run names: full (complete flight), verify (reduced replication flight), diagnostics, adhoc/<timestamp> (single
commands), _dev/<name> (development and smoke runs; not for analysis). Each level carries a manifest
(device.json, browser.json, run.json); scripts/index_results.py collects them into results/INDEX.json and INDEX.csv.

Device id: TRAINBENCH_DEVICE, else device.local.json at the repo root (created on first use from the host: OS and
GPU, e.g. win11-rtx3060; edit it to rename), so a copied results/<device-id>/ folder never collides with another device.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

from benchkit.paths import ROOT

RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
DEVICE_FILE = ROOT / "device.local.json"


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        name = None
    if name:
        # "NVIDIA GeForce RTX 3060" -> "rtx3060"
        name = re.sub(r"(?i)nvidia|geforce|laptop gpu|graphics", "", name)
        return _slug(name)
    return None


def _os_slug():
    system = platform.system()
    if system == "Windows":
        return "win11" if platform.release() == "11" else f"win{_slug(platform.release())}"
    if system == "Darwin":
        return "macos"


def _apple_chip():
    """"Apple M3" -> "m3" (macOS on Apple silicon), else None."""
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        m = re.match(r"Apple (M\d+\s*\w*)", r.stdout.strip())
        return _slug(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None
    return _slug(system) or "unknown-os"


def derive_device_id():
    chip = _apple_chip() if platform.system() == "Darwin" else None
    return f"{_os_slug()}-{chip or _gpu_name() or _slug(platform.machine()) or 'device'}"


def device_id(override=None):
    """The id of this machine's result folder. Order: argument, TRAINBENCH_DEVICE, device.local.json, derived (and saved)."""
    if override:
        return override
    if os.environ.get("TRAINBENCH_DEVICE"):
        return os.environ["TRAINBENCH_DEVICE"]
    if DEVICE_FILE.exists():
        return json.loads(DEVICE_FILE.read_text(encoding="utf-8"))["id"]
    dev = derive_device_id()
    DEVICE_FILE.write_text(json.dumps({"id": dev, "note": "edit 'id' to rename this device's result folder"}, indent=1),
                           encoding="utf-8")
    return dev


def browser_id(browser):
    """<name>-<pinned version>, e.g. chrome-157.0.8081.0 or firefox-158.0."""
    if browser == "chrome":
        from benchkit.chrome import version
    elif browser == "firefox":
        from benchkit.firefox import version
    else:
        raise ValueError(f"unknown browser {browser!r}")
    v = version()
    return f"{browser}-{v}" if v else browser


def device_dir(dev=None):
    return RESULTS / device_id(dev)


def browser_dir(browser, dev=None):
    return device_dir(dev) / browser_id(browser)


def run_dir(browser, run="full", dev=None):
    return browser_dir(browser, dev) / run


def native_dir(run="adhoc", dev=None):
    return device_dir(dev) / "native" / run


def probes_dir(dev=None):
    return device_dir(dev) / "probes"


def reports_dir(browser=None, dev=None):
    base = REPORTS / device_id(dev)
    return base / browser_id(browser) if browser else base


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def harness_state():
    """Commit, whether tracked files differ from it, and the sha256 of that difference."""
    try:
        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD"], capture_output=True, timeout=120).stdout   # raw bytes
    except Exception:  # noqa: BLE001
        diff = b""
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(diff),
            "patch_sha256": hashlib.sha256(diff).hexdigest() if diff else None}


def _write_if_new(path, obj):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def write_device_manifest(dev=None):
    """results/<device>/device.json, written once."""
    from benchkit.envinfo import host_info
    h = host_info()
    _write_if_new(device_dir(dev) / "device.json", {
        "id": device_id(dev), "created_at": datetime.now().isoformat(timespec="seconds"), "os": h["os"], "cpu": h["cpu"],
        "cpu_logical_cores": h["cpu_logical_cores"], "ram_total_gb": h["ram_total_gb"], "power_plan": h["power_plan"],
        "nvidia": h["nvidia"], **({"apple_gpu": h["apple_gpu"], "cpu_perf_cores": h["cpu_perf_cores"],
                                   "cpu_efficiency_cores": h["cpu_efficiency_cores"]} if h.get("apple_gpu") else {})})


def write_browser_manifest(browser, dev=None):
    """results/<device>/<browser-id>/browser.json, written once."""
    info = {"id": browser_id(browser), "browser": browser, "created_at": datetime.now().isoformat(timespec="seconds")}
    if browser == "chrome":
        from benchkit.chrome import CHROME_FLAGS, PINNED_CHROME
        info.update({"executable": str(PINNED_CHROME.relative_to(ROOT)), "flags": CHROME_FLAGS})
    else:
        from benchkit.firefox import FIREFOX_PREFS, PINNED_FIREFOX, PINNED_GECKODRIVER
        info.update({"executable": str(PINNED_FIREFOX.relative_to(ROOT)), "driver": str(PINNED_GECKODRIVER.relative_to(ROOT)),
                     "prefs": FIREFOX_PREFS})
    _write_if_new(browser_dir(browser, dev) / "browser.json", info)


def update_run_manifest(path, **fields):
    """Create or merge results/.../<run>/run.json (started_at kept from the first write)."""
    path = Path(path)
    f = path / "run.json"
    cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"started_at": datetime.now().isoformat(timespec="seconds")}
    fields = {k: v for k, v in fields.items() if v is not None}
    if "harness" in cur and "harness" in fields:
        # A resumed run keeps the provenance of its first launch; later launches are listed under "resumes".
        cur.setdefault("resumes", []).append({"at": datetime.now().isoformat(timespec="seconds"),
                                              "harness": fields.pop("harness"), "command": fields.pop("command", None)})
    cur.update(fields)
    path.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cur, indent=1, default=str), encoding="utf-8")
    return cur
