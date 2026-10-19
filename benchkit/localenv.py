"""
Per-device settings from <repo>/trainbench.env (git-ignored; trainbench.env.example lists every key).

Lines are KEY=VALUE, '#' starts a comment. Only TRAINBENCH_* keys are read. A variable already set in the real
environment wins over the file, so a one-off override on the command line still works. load() runs when benchkit.paths,
benchkit.clockguard or benchkit.gpucheck is imported, and child processes inherit the result. The values in force are
recorded in run.json (guard_settings, tool_paths, require_gpu_vendor).
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / "trainbench.env"
_loaded = False


def load():
    """Apply trainbench.env to os.environ (once per process); returns the keys it set."""
    global _loaded
    set_keys = []
    if _loaded or not ENV_FILE.exists():
        _loaded = True
        return set_keys
    _loaded = True
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        val = val.strip("\"'")
        if key.startswith("TRAINBENCH_") and key not in os.environ:
            os.environ[key] = val
            set_keys.append(key)
    return set_keys


def file_keys():
    """Keys that trainbench.env defines (whether or not the real environment overrode them)."""
    if not ENV_FILE.exists():
        return []
    return [l.split("#", 1)[0].split("=", 1)[0].strip() for l in ENV_FILE.read_text(encoding="utf-8").splitlines()
            if "=" in l.split("#", 1)[0]]
