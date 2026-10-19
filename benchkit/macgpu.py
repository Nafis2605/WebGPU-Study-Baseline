"""
macOS (Apple silicon) replacements for the NVML parts of the harness. Nothing here needs sudo.

  gpu_static()       GPU model, core count, Metal family, unified memory size (ioreg + system_profiler)
  gpu_sample()       one IOAccelerator PerformanceStatistics reading: Device / Renderer / Tiler utilisation (%),
                     in-use and allocated system memory of the GPU driver (bytes). Apple GPUs share system memory,
                     so there is no device-memory or PCIe counter; those NVML columns are left empty, not estimated.
  power_state()      Low Power Mode, AC power, thermal / performance warning level (pmset)
  guard()            the macOS counterpart of the NVIDIA idle / boost guards: AC power, Low Power Mode off,
                     no thermal or performance warning recorded. Returns (ok, message).
  display_refresh_hz()  refresh rate of the main display (CoreGraphics), or None
  swap_counters()    vm.swapusage + pageouts, so memory pressure on an 8 GB machine is visible per launch

GPU clock and power are only available through `powermetrics`, which needs root; MacSampler can run it when
TRAINBENCH_POWERMETRICS=1 and passwordless sudo for powermetrics is configured, otherwise those columns are empty.
"""
import ctypes
import ctypes.util
import json
import os
import plistlib
import re
import subprocess
import threading
import time


def _run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace").stdout
    except Exception:  # noqa: BLE001
        return ""


def _accelerator():
    """The first IOAccelerator entry as a dict (ioreg -a gives a plist)."""
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-a", "-c", "IOAccelerator"], capture_output=True, timeout=10).stdout
    if not out:
        return {}
    entries = plistlib.loads(out)
    return entries[0] if entries else {}


def gpu_static():
    acc = _accelerator()
    sp = _run(["system_profiler", "SPDisplaysDataType", "SPHardwareDataType", "-json"], timeout=60)
    try:
        sp = json.loads(sp)
    except ValueError:
        sp = {}
    disp = (sp.get("SPDisplaysDataType") or [{}])[0]
    hw = (sp.get("SPHardwareDataType") or [{}])[0]
    return {"name": acc.get("model") or disp.get("sppci_model"), "gpu_cores": acc.get("gpu-core-count"),
            "ioclass": acc.get("IOClass"), "metal": disp.get("spdisplays_mtlgpufamilysupport"),
            "chip": hw.get("chip_type"), "machine_model": hw.get("machine_model"),
            "unified_memory": hw.get("physical_memory"), "os_loader": hw.get("os_loader_version"),
            "boot_rom": hw.get("boot_rom_version")}


def gpu_sample():
    stats = _accelerator().get("PerformanceStatistics", {})
    return {"gpu_util_pct": stats.get("Device Utilization %"), "renderer_util_pct": stats.get("Renderer Utilization %"),
            "tiler_util_pct": stats.get("Tiler Utilization %"), "gpu_mem_in_use_mb": _mb(stats.get("In use system memory")),
            "gpu_mem_alloc_mb": _mb(stats.get("Alloc system memory")), "gpu_recovery_count": stats.get("recoveryCount")}


def _mb(v):
    return v / 2**20 if isinstance(v, (int, float)) else None


def power_state():
    pm = _run(["pmset", "-g"])
    lpm = re.search(r"lowpowermode\s+(\d)", pm)
    batt = _run(["pmset", "-g", "batt"])
    therm = _run(["pmset", "-g", "therm"])
    return {"low_power_mode": None if lpm is None else lpm.group(1) == "1",
            "on_ac": "AC Power" in batt.splitlines()[0] if batt else None,
            "thermal_warning": "No thermal warning level has been recorded" not in therm,
            "performance_warning": "No performance warning level has been recorded" not in therm,
            "therm_raw": therm.strip()}


def guard():
    s = power_state()
    bad = []
    if s["low_power_mode"]:
        bad.append("Low Power Mode is on")
    if s["on_ac"] is False:
        bad.append("running on battery")
    if s["thermal_warning"]:
        bad.append("thermal warning recorded")
    if s["performance_warning"]:
        bad.append("performance warning recorded")
    return (not bad), ("; ".join(bad) if bad else "AC power, Low Power Mode off, no thermal/performance warning")


def wait_cool(timeout_s=900, poll_s=5):
    """macOS has no GPU temperature without root: wait until no thermal/performance warning is recorded.
    Returns seconds waited. (A GPU-utilisation criterion was tried and removed: the idle floor of this machine is
    about 22 % because the compositor drives the retina display, so "back near idle" never held and every wait ran
    into its timeout; see results/<device>/chrome-*/_dev/preflight_aborted_1.)"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = power_state()
        if not s["thermal_warning"] and not s["performance_warning"]:
            break
        time.sleep(poll_s)
    return time.time() - t0


def display_refresh_hz():
    """Main display refresh rate via CoreGraphics (CGDisplayModeGetRefreshRate). ProMotion panels report the
    current mode's rate; set a fixed rate in System Settings > Displays so it stays constant."""
    try:
        cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p
        cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
        cg.CGDisplayModeGetRefreshRate.restype = ctypes.c_double
        cg.CGDisplayModeGetRefreshRate.argtypes = [ctypes.c_void_p]
        cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
        mode = cg.CGDisplayCopyDisplayMode(cg.CGMainDisplayID())
        if not mode:
            return None
        hz = cg.CGDisplayModeGetRefreshRate(mode)
        cg.CGDisplayModeRelease(mode)
        return round(hz, 2)
    except Exception:  # noqa: BLE001
        return None


def swap_counters():
    sw = _run(["sysctl", "-n", "vm.swapusage"])
    used = re.search(r"used = ([\d.]+)M", sw)
    vm = _run(["vm_stat"])
    pageouts = re.search(r"Pageouts:\s+(\d+)", vm)
    swapouts = re.search(r"Swapouts:\s+(\d+)", vm)
    return {"swap_used_mb": float(used.group(1)) if used else None,
            "pageouts": int(pageouts.group(1)) if pageouts else None,
            "swapouts": int(swapouts.group(1)) if swapouts else None}


class MacSampler:
    """Background ioreg sampler with the same start()/stop() -> list-of-dicts shape as the NVML samplers.
    ioreg takes ~10-20 ms per call, so the interval is not shorter than 100 ms."""

    def __init__(self, interval_s=0.1):
        self.interval_s = max(interval_s, 0.1)
        self.samples = []
        self._stop = threading.Event()
        self._t = None
        self._pm = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append({"epoch_s": time.time(), **gpu_sample()})
            except Exception as e:  # noqa: BLE001
                self.samples.append({"epoch_s": time.time(), "error": str(e)})
            self._stop.wait(self.interval_s)

    def start(self):
        if os.environ.get("TRAINBENCH_POWERMETRICS") == "1":
            self._pm = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", "100",
                                         "--format", "plist"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        if self._pm:
            self._pm.terminate()
            try:
                raw = self._pm.communicate(timeout=10)[0]
                self.powermetrics = [plistlib.loads(b) for b in raw.split(b"\x00") if b.strip()]
            except Exception:  # noqa: BLE001
                self.powermetrics = []
        return self.samples


if __name__ == "__main__":
    print(json.dumps({"static": gpu_static(), "sample": gpu_sample(), "power": power_state(), "guard": guard(),
                      "refresh_hz": display_refresh_hz(), "swap": swap_counters()}, indent=1, default=str))
