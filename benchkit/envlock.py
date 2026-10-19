"""
Environment lock for the study: everything that could change a measurement is recorded once
(`freeze`) in env.lock.json and checked (`verify`) before the study, before every stage and before
every launch. Any mismatch aborts the run; nothing degrades silently. Every result folder records
the lock's sha256.

Locked items:
  host       Windows build, CPU, logical cores, RAM, display refresh rate, active power plan GUID
  gpu        name, vBIOS, driver version, default power limit, CUDA driver (UMD) version
  browser    pinned Chrome for Testing: path, version, sha256 of chrome.exe and chrome.dll
  firefox    pinned Firefox copy: path, version + BuildID (application.ini), sha256 of firefox.exe, xul.dll and
             distribution/policies.json; geckodriver path, version and sha256
  node       node / npm versions, installed versions of every @tensorflow package, vite, playwright (npm)
  python     .venv interpreter version and the sha256 of its sorted `pip freeze`
  native     tfgpu env: interpreter, `pip freeze` sha256, conda package list sha256, TensorFlow version
             and its CUDA / cuDNN build versions, GPU visible
  tools      Nsight Systems version, Nsight Graphics install
  artifacts  sha256 of public/spec/models.json, public/data/manifest.json, public/weights/manifest.json;
             full verify also re-hashes every data and weight file against its manifest
  code       git commit; the working tree must be clean (TRAINBENCH_ALLOW_DIRTY=1 only for development)

(Port of the sibling TF.js inference study's envlock; zoo-model items replaced by the study artifacts.)

Scopes for verify():
  full     everything (study start, every stage)
  native   everything except the browser-only items' hashes (native stage start)
  launch   cheap subset checked before every launch: GPU driver + name, browser path + version,
           display refresh, power plan

Usage:
    python -m benchkit.envlock freeze      # write env.lock.json (after setup, before the preflight)
    python -m benchkit.envlock verify      # print the comparison; exit 1 on mismatch
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchkit.paths import ROOT, TFGPU, CONDA, NSYS, NGFX_DIR, SPEC, DATA_DIR, WEIGHTS_DIR  # noqa: E402

LOCK = ROOT / "env.lock.json"


def _run(cmd, **kw):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)
    except FileNotFoundError:
        sys.exit(f"envlock: '{cmd[0]}' was not found; install it or put it on PATH (see README, Setup).")
    return r.stdout.strip()


def _ps(expr):
    return _run(["powershell", "-NoProfile", "-Command", expr])


def _sha_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def gpu_info():
    q = _run(["nvidia-smi", "--query-gpu=name,vbios_version,driver_version,power.default_limit", "--format=csv,noheader"])
    name, vbios, driver, plimit = [x.strip() for x in q.split(",")]
    umd = next((l.split(":", 1)[1].strip().split()[0] for l in _run(["nvidia-smi", "-q"]).splitlines()
                if l.strip().startswith("CUDA Version")), None)
    return {"name": name, "vbios": vbios, "driver": driver, "default_power_limit": plimit, "cuda_driver_version": umd}


def host_items():
    os_build = _ps("(Get-CimInstance Win32_OperatingSystem | % { $_.Caption + ' ' + $_.Version })")
    refresh = _ps("(Get-CimInstance Win32_VideoController | ? { $_.CurrentRefreshRate } | "
                  "Select-Object -First 1).CurrentRefreshRate")
    plan = _run(["powercfg", "/getactivescheme"]).split(":", 1)[-1].strip()
    import psutil
    return {"os": os_build, "cpu": platform.processor(), "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "display_refresh_hz": refresh,
            "power_plan": plan}


def browser_items(hash_files=True):
    from benchkit.chrome import PINNED_CHROME
    d = PINNED_CHROME.parent
    version = next((p.stem.replace(".manifest", "") for p in d.glob("*.manifest")), None)
    out = {"path": str(PINNED_CHROME.relative_to(ROOT)), "version": version, "exists": PINNED_CHROME.is_file()}
    if hash_files and PINNED_CHROME.is_file():
        out["chrome_exe_sha256"] = _sha_file(PINNED_CHROME)
        out["chrome_dll_sha256"] = _sha_file(d / "chrome.dll")
    return out


def firefox_items(hash_files=True):
    from benchkit.firefox import PINNED_FIREFOX, PINNED_GECKODRIVER
    d = PINNED_FIREFOX.parent
    ini = {}
    if (d / "application.ini").exists():
        for line in (d / "application.ini").read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                ini.setdefault(k.strip(), v.strip())
    out = {"path": str(PINNED_FIREFOX.relative_to(ROOT)), "version": ini.get("Version"), "build_id": ini.get("BuildID"),
           "exists": PINNED_FIREFOX.is_file(), "geckodriver_path": str(PINNED_GECKODRIVER.relative_to(ROOT)),
           "geckodriver_version": _run([str(PINNED_GECKODRIVER), "--version"]).splitlines()[0]
           if PINNED_GECKODRIVER.is_file() else None}
    if hash_files and PINNED_FIREFOX.is_file():
        out["firefox_exe_sha256"] = _sha_file(PINNED_FIREFOX)
        out["xul_dll_sha256"] = _sha_file(d / "xul.dll")
        out["policies_sha256"] = _sha_file(d / "distribution" / "policies.json")
        out["geckodriver_sha256"] = _sha_file(PINNED_GECKODRIVER)
    return out


def node_items():
    def pkg(name):
        p = ROOT / "node_modules" / name / "package.json"
        return json.loads(p.read_text(encoding="utf-8"))["version"] if p.exists() else None
    tf_pkgs = sorted(p.name for p in (ROOT / "node_modules" / "@tensorflow").iterdir()) \
        if (ROOT / "node_modules" / "@tensorflow").exists() else []
    return {"node": _run(["node", "--version"]), "npm": _run(["npm.cmd", "--version"], shell=False),
            "packages": {**{f"@tensorflow/{n}": pkg(f"@tensorflow/{n}") for n in tf_pkgs}, "vite": pkg("vite")},
            "package_lock_sha256": _sha_file(ROOT / "package-lock.json")}


def python_items():
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    freeze = _run([str(venv_py), "-m", "pip", "freeze", "--all"])
    lines = sorted(freeze.splitlines())
    key = {l.split("==")[0].lower(): l.split("==")[1] for l in lines if "==" in l}
    return {"version": _run([str(venv_py), "--version"]), "pip_freeze_sha256": _sha_text("\n".join(lines)),
            "key_packages": {k: key.get(k) for k in ("playwright", "pynvml", "nvidia-ml-py", "numpy", "pandas",
                                                     "scipy", "psutil", "matplotlib", "selenium")}}


def native_items():
    py = TFGPU / "python.exe"
    if not py.exists():
        return {"present": False}
    freeze = sorted(_run([str(py), "-m", "pip", "freeze", "--all"]).splitlines())
    conda = _run([str(CONDA), "list", "-p", str(TFGPU), "--export"]) if CONDA.exists() else ""
    conda_lines = sorted(l for l in conda.splitlines() if l and not l.startswith("#"))
    env = {**os.environ, "PATH": str(TFGPU / "Library" / "bin") + os.pathsep + os.environ.get("PATH", ""),
           "TF_CPP_MIN_LOG_LEVEL": "3"}
    tfinfo = _run([str(py), "-c", "import json, tensorflow as tf; b = tf.sysconfig.get_build_info(); "
                   "print(json.dumps({'tf': tf.__version__, 'cuda': b.get('cuda_version'), "
                   "'cudnn': b.get('cudnn_version'), 'gpus': len(tf.config.list_physical_devices('GPU'))}))"],
                  env=env)
    tfinfo = json.loads(tfinfo.splitlines()[-1]) if tfinfo else None
    cudnn = next((l for l in conda_lines if l.startswith("cudnn=")), None)
    cudatk = next((l for l in conda_lines if l.startswith("cudatoolkit=")), None)
    return {"present": True, "path": str(TFGPU), "python": _run([str(py), "--version"]),
            "pip_freeze_sha256": _sha_text("\n".join(freeze)), "conda_list_sha256": _sha_text("\n".join(conda_lines)),
            "tensorflow": tfinfo, "conda_cudatoolkit": cudatk, "conda_cudnn": cudnn}


def tool_items():
    v = _run([str(NSYS), "--version"]) if NSYS.exists() else None
    return {"nsight_systems": v, "nsight_graphics_dir": NGFX_DIR.name if NGFX_DIR.exists() else None}


def _artifact_files():
    """(relative path, sha256) of every data file (data manifest) and weight file (weights manifest)."""
    out = []
    dman = DATA_DIR / "manifest.json"
    if dman.exists():
        for name, f in json.loads(dman.read_text(encoding="utf-8"))["files"].items():
            out.append((f"public/data/{name}", f["sha256"]))
    wman = WEIGHTS_DIR / "manifest.json"
    if wman.exists():
        for model, kinds in json.loads(wman.read_text(encoding="utf-8")).items():
            for kind, e in kinds.items():
                out.append((f"public/weights/{e['file']}", e["sha256"]))
    return out


def artifact_items():
    def sha_or_none(p):
        return _sha_file(p) if p.exists() else None
    wman = WEIGHTS_DIR / "manifest.json"
    kinds = sorted({k for v in json.loads(wman.read_text(encoding="utf-8")).values() for k in v}) if wman.exists() else []
    return {"spec_sha256": sha_or_none(SPEC), "data_manifest_sha256": sha_or_none(DATA_DIR / "manifest.json"),
            "weights_manifest_sha256": sha_or_none(wman), "weight_kinds": kinds, "files": len(_artifact_files())}


def check_artifact_files():
    """Re-hash every data and weight file against its manifest (full verify). Returns mismatching paths."""
    return [rel for rel, sha in _artifact_files() if not (ROOT / rel).exists() or _sha_file(ROOT / rel) != sha]


def code_items():
    commit = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "clean": not dirty}


def collect(scope="full"):
    if scope == "launch":
        return {"gpu": gpu_info(), "browser": browser_items(hash_files=False), "firefox": firefox_items(hash_files=False),
                "host": {k: v for k, v in host_items().items() if k in ("display_refresh_hz", "power_plan")}}
    return {"host": host_items(), "gpu": gpu_info(), "browser": browser_items(hash_files=(scope == "full")),
            "firefox": firefox_items(hash_files=(scope == "full")),
            "node": node_items(), "python": python_items(), "native": native_items(), "tools": tool_items(),
            "artifacts": artifact_items()}


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif a != b:
        out.append(f"{path}: locked={a!r} now={b!r}")
    return out


def _subset(locked, now):
    """Keep only the keys of `locked` present in `now` (launch/native scopes check a subset)."""
    if isinstance(now, dict) and isinstance(locked, dict):
        return {k: _subset(locked.get(k), v) for k, v in now.items()}
    return locked


def freeze():
    items = collect("full")
    items["code"] = code_items()
    body = json.dumps(items, indent=1, sort_keys=True)
    lock = {"lock_sha256": _sha_text(body), **items}
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    return lock


def verify(scope="full", check_files=None):
    """Compare the current environment with env.lock.json; SystemExit on mismatch.
    Returns {"sha256": lock hash, "scope": scope}."""
    if not LOCK.exists():
        raise SystemExit("env.lock.json missing: run `python -m benchkit.envlock freeze` after setup")
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    now = collect("launch" if scope == "launch" else ("native" if scope == "native" else "full"))
    if scope == "native":
        now["browser"] = {k: v for k, v in now["browser"].items() if not k.endswith("sha256")}
        now["firefox"] = {k: v for k, v in now["firefox"].items() if not k.endswith("sha256")}
    problems = _diff(_subset({k: locked.get(k) for k in now}, now), now)
    if scope != "launch":
        code = code_items()
        allow_dirty = os.environ.get("TRAINBENCH_ALLOW_DIRTY") == "1"
        if not code["clean"] and not allow_dirty:
            problems.append("code: working tree has uncommitted changes (commit before running)")
        if code["commit"] != locked.get("code", {}).get("commit") and not allow_dirty:
            problems.append(f"code: commit {code['commit'][:10]} differs from the locked {str(locked.get('code', {}).get('commit'))[:10]}")
    if check_files or (check_files is None and scope == "full"):
        problems += [f"artifact file hash mismatch: {p}" for p in check_artifact_files()]
    if problems:
        raise SystemExit("ENVIRONMENT LOCK MISMATCH (" + scope + "):\n  " + "\n  ".join(problems))
    return {"sha256": locked["lock_sha256"], "scope": scope}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "freeze":
        lock = freeze()
        print(f"wrote {LOCK} (sha256 {lock['lock_sha256'][:16]})")
        print(json.dumps({k: lock[k] for k in ("gpu", "browser", "native", "tools", "code")}, indent=1))
    elif cmd == "verify":
        scope = sys.argv[2] if len(sys.argv) > 2 else "full"
        r = verify(scope)
        print(f"environment matches the lock ({scope}; sha256 {r['sha256'][:16]})")
    elif cmd == "show":
        print(json.dumps(collect(sys.argv[2] if len(sys.argv) > 2 else "full"), indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
