"""Machine paths shared by the drivers (kept in one place; part of the environment lock).

Windows (the first flight's machine) and macOS (Apple silicon) have separate paths; nothing else differs here.
On macOS the native env is tensorflow-macos 2.10.0 + tensorflow-metal (no CUDA) and Nsight is not available
(NSYS is None; the GPU trace stage uses Xcode Instruments instead, see nsight/run_xctrace.py).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IS_MAC = sys.platform == "darwin"
if IS_MAC:
    TFGPU = Path(os.environ.get("TRAINBENCH_TFENV", Path.home() / "tfjsbench" / "tfmac"))
    CONDA = Path(os.environ.get("TRAINBENCH_CONDA", Path.home() / "miniconda3" / "bin" / "conda"))
    NSYS = None
    TFGPU_PY = TFGPU / "bin" / "python"
    VENV_PY = ROOT / ".venv" / "bin" / "python"
else:
    TFGPU = Path(r"C:\tfjsbench\tfgpu")
    CONDA = Path(r"C:\ProgramData\miniconda3\Scripts\conda.exe")
    NSYS = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe")
    TFGPU_PY = TFGPU / "python.exe"
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
SPEC = ROOT / "public" / "spec" / "models.json"
DATA_DIR = ROOT / "public" / "data"
WEIGHTS_DIR = ROOT / "public" / "weights"


def tfgpu_env(extra=None, cpu_only=False):
    """Environment for a tfgpu child process: CUDA/cuDNN DLLs from the conda env first on PATH (Windows).
    cpu_only hides the GPU: CUDA_VISIBLE_DEVICES on Windows; on macOS the worker hides the Metal device itself
    (tf.config.set_visible_devices), signalled by TRAINBENCH_CPU_ONLY."""
    path = str(TFGPU / ("bin" if IS_MAC else "Library/bin"))
    env = {**os.environ, "PATH": path + os.pathsep + os.environ.get("PATH", ""), "TF_CPP_MIN_LOG_LEVEL": "2"}
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["TRAINBENCH_CPU_ONLY"] = "1"
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
