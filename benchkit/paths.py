"""Machine paths shared by the drivers (kept in one place; part of the environment lock)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TFGPU = Path(r"C:\tfjsbench\tfgpu")
CONDA = Path(r"C:\ProgramData\miniconda3\Scripts\conda.exe")
NSYS = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe")
SPEC = ROOT / "public" / "spec" / "models.json"
DATA_DIR = ROOT / "public" / "data"
WEIGHTS_DIR = ROOT / "public" / "weights"


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
