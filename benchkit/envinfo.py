"""Host environment capture for env.json."""
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import psutil


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def nvidia_info():
    """Static NVIDIA GPU facts via NVML, or {'error': ...} if NVML is unavailable."""
    try:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)

        def safe(fn, *a):
            try:
                return fn(*a)
            except nv.NVMLError as e:
                return f"unavailable: {e}"

        info = {
            "name": safe(nv.nvmlDeviceGetName, h),
            "driver_version": safe(nv.nvmlSystemGetDriverVersion),
            "vbios_version": safe(nv.nvmlDeviceGetVbiosVersion, h),
            "memory_total_mb": safe(lambda: nv.nvmlDeviceGetMemoryInfo(h).total / 2**20),
            "power_limit_w": safe(lambda: nv.nvmlDeviceGetPowerManagementLimit(h) / 1000),
            "max_sm_clock_mhz": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_SM),
            "max_mem_clock_mhz": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_MEM),
            "pcie_max_gen": safe(nv.nvmlDeviceGetMaxPcieLinkGeneration, h),
            "pcie_max_width": safe(nv.nvmlDeviceGetMaxPcieLinkWidth, h),
            "driver_model": safe(lambda: "WDDM" if nv.nvmlDeviceGetDriverModel(h)[0] == nv.NVML_DRIVER_WDDM else "TCC"),
        }
        nv.nvmlShutdown()
        return info
    except Exception as e:
        return {"error": str(e)}


def _powershell(cmd):
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return None


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def js_packages():
    """Installed versions of the page's npm dependencies (read from node_modules)."""
    root = Path(__file__).resolve().parent.parent
    out = {}
    deps = json.loads((root / "package.json").read_text(encoding="utf-8")).get("dependencies", {})
    for name in deps:
        pj = root / "node_modules" / name / "package.json"
        out[name] = json.loads(pj.read_text(encoding="utf-8")).get("version") if pj.exists() else None
    return out


def host_info():
    packages = {}
    for name in ("playwright", "pandas", "numpy", "psutil", "nvidia-ml-py"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    packages.update(js_packages())
    mac = platform.system() == "Darwin"
    if mac:
        from benchkit import macgpu
        mac_power = macgpu.power_state()
    try:
        cpu_max = psutil.cpu_freq().max if psutil.cpu_freq() else None
    except Exception:  # noqa: BLE001  (psutil has no cpu_freq on some Apple silicon builds)
        cpu_max = None
    return {
        "os": platform.platform(),
        "cpu": (_powershell("(Get-CimInstance Win32_Processor).Name") if platform.system() == "Windows" else None)
               or (_sysctl("machdep.cpu.brand_string") if mac else None) or platform.processor(),
        "cpu_max_mhz": cpu_max,
        "power_plan": _powershell("powercfg /getactivescheme") if platform.system() == "Windows" else
                      ({"low_power_mode": mac_power["low_power_mode"], "on_ac": mac_power["on_ac"]} if mac else None),
        "cpu_perf_cores": _sysctl("hw.perflevel0.logicalcpu") if mac else None,
        "cpu_efficiency_cores": _sysctl("hw.perflevel1.logicalcpu") if mac else None,
        "apple_gpu": macgpu.gpu_static() if mac else None,
        "memory_pressure": macgpu.swap_counters() if mac else None,
        "on_battery": (psutil.sensors_battery().power_plugged is False) if psutil.sensors_battery() else False,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "python": sys.version,
        "packages": packages,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "nvidia": nvidia_info(),
    }
