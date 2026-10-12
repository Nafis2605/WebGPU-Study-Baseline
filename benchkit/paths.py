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
