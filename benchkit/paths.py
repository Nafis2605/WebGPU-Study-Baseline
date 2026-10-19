"""Machine paths shared by the drivers (kept in one place; part of the environment lock)."""
import os
from pathlib import Path

from benchkit import localenv

ROOT = Path(__file__).resolve().parents[1]
localenv.load()   # per-device TRAINBENCH_* settings from trainbench.env (git-ignored), before any override below is read

# Install locations. The defaults are the layout of the first study machine; another machine overrides them with
# TRAINBENCH_TFGPU / TRAINBENCH_CONDA / TRAINBENCH_NSYS / TRAINBENCH_NGFX, or (conda, Nsight) is found when the default
# is absent. An override that does not exist is kept as given, so the environment lock records the tool as absent.
_DEFAULT_TFGPU = Path(r"C:\tfjsbench\tfgpu")
_DEFAULT_CONDA = Path(r"C:\ProgramData\miniconda3\Scripts\conda.exe")
_DEFAULT_NSYS = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe")
_DEFAULT_NGFX = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1")


def _newest_glob(patterns):
    """Newest match of the glob patterns (absolute, with '*' in a directory name), by path order."""
    found = []
    for pat in patterns:
        p = Path(pat)
        anchor = Path(p.anchor)
        found += list(anchor.glob(str(p.relative_to(anchor))))
    return sorted(found)[-1] if found else None


def _conda_default():
    home = Path(os.environ.get("USERPROFILE", ""))
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for p in (_DEFAULT_CONDA, home / "miniconda3" / "Scripts" / "conda.exe", local / "miniconda3" / "Scripts" / "conda.exe",
              home / "anaconda3" / "Scripts" / "conda.exe"):
        if p.exists():
            return p
    return _DEFAULT_CONDA


def _nsys_default():
    if _DEFAULT_NSYS.exists():
        return _DEFAULT_NSYS
    return _newest_glob([r"C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe"]) or _DEFAULT_NSYS


def _ngfx_default():
    if _DEFAULT_NGFX.exists():
        return _DEFAULT_NGFX
    return _newest_glob([r"C:\Program Files\NVIDIA Corporation\Nsight Graphics *"]) or _DEFAULT_NGFX


TFGPU = Path(os.environ.get("TRAINBENCH_TFGPU") or _DEFAULT_TFGPU)
CONDA = Path(os.environ.get("TRAINBENCH_CONDA") or _conda_default())
NSYS = Path(os.environ.get("TRAINBENCH_NSYS") or _nsys_default())
NGFX_DIR = Path(os.environ.get("TRAINBENCH_NGFX") or _ngfx_default())
SPEC = ROOT / "public" / "spec" / "models.json"
DATA_DIR = ROOT / "public" / "data"
WEIGHTS_DIR = ROOT / "public" / "weights"


def path_settings():
    """The resolved install locations and whether each exists (recorded in run.json)."""
    return {name: {"path": str(p), "exists": p.exists(), "from_env": bool(os.environ.get(f"TRAINBENCH_{name}"))}
            for name, p in (("TFGPU", TFGPU), ("CONDA", CONDA), ("NSYS", NSYS), ("NGFX", NGFX_DIR))}


def tfgpu_env(extra=None, cpu_only=False):
    """Environment for a tfgpu child process: CUDA/cuDNN DLLs from the conda env first on PATH."""
    env = {**os.environ, "PATH": str(TFGPU / "Library" / "bin") + os.pathsep + os.environ.get("PATH", ""),
           "TF_CPP_MIN_LOG_LEVEL": "2"}
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.update(extra or {})
    return env


def resolve_result_dir(study, stage, recorded):
    """A stage's result folder from its stage_done.json. The recorded path is absolute; when the study folder
    has moved since (e.g. into results/<device>/chrome-<ver>/), rebase it onto <study>/<stage>/<timestamp>."""
    if not recorded:
        return None
    p = Path(recorded)
    if p.exists():
        return p
    # Stages whose folder is itself the result (gpu_trace) record <study>/<stage>.
    rebased = Path(study) / stage if p.name == stage else Path(study) / stage / p.name
    return rebased if rebased.exists() else None
