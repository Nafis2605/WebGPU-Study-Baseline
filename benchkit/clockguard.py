"""
GPU power-state guards (NVML), independent of the telemetry stream so they also run in stages
recorded without telemetry (responsiveness, correctness).

ClockSampler      samples the SM clock every interval_s in a thread; max_mhz() after stop().
idle_check()      the GPU can idle: the minimum SM clock over a few seconds is <= idle_max_mhz
                  (a clock lock keeps it at base even when idle)
LOW_CLOCK_MHZ     a GPU launch whose SM clock never exceeded this ran in a stuck low-power state
                  ("GPU idle" throttle reason while busy); such launches are re-run, never recorded

macOS (Apple silicon): the GPU clock is readable only as root (powermetrics), so ClockSampler records nothing
(max_mhz() is None: no low-clock re-run) and idle_check() is the power-state guard of benchkit/macgpu.py
(AC power, Low Power Mode off, no thermal / performance warning).
"""
import sys
import threading
import time

IS_MAC = sys.platform == "darwin"

LOW_CLOCK_MHZ = 1000


class ClockSampler:
    def __init__(self, interval_s=0.05):
        self.interval_s, self.samples = interval_s, []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        if IS_MAC:
            return
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        try:
            while not self._stop.is_set():
                self.samples.append((time.time(), nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)))
                time.sleep(self.interval_s)
        finally:
            nv.nvmlShutdown()

    def start(self):
        self._stop.clear()
        self.samples = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        return self.max_mhz()

    def max_mhz(self):
        return max((c for _, c in self.samples), default=None)


def idle_check(seconds=3.0, idle_max_mhz=1000):
    if IS_MAC:
        from benchkit.macgpu import guard
        return guard()
    s = ClockSampler(0.1).start()
    time.sleep(seconds)
    s.stop()
    clocks = [c for _, c in s.samples]
    return (min(clocks) <= idle_max_mhz if clocks else True), \
        f"idle SM clock min {min(clocks, default=0)} MHz, max {max(clocks, default=0)} MHz"
