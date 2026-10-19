"""
GPU power-state guards (NVML), independent of the telemetry stream so they also run in stages
recorded without telemetry (responsiveness, correctness).

ClockSampler      samples the SM clock every interval_s in a thread; max_mhz() after stop().
idle_check()      the GPU can idle: the minimum SM clock over a few seconds is <= idle_max_mhz
                  (a clock lock keeps it at base even when idle)
LOW_CLOCK_MHZ     a GPU launch whose SM clock never exceeded this ran in a stuck low-power state
                  ("GPU idle" throttle reason while busy); such launches are re-run, never recorded
IDLE_MAX_MHZ      idle_check() limit; BOOST_MHZ the clock the boost probe must reach; COOL_TO_C the temperature
                  a launch waits for (at most COOL_TIMEOUT_S seconds). Defaults are the first study machine's
                  (RTX 3060); another GPU sets TRAINBENCH_IDLE_MAX_MHZ / _BOOST_MHZ / _COOL_TO_C / _COOL_TIMEOUT_S
                  in the environment or in trainbench.env. guard_settings() lists the values in force so
                  the drivers can record them in run.json.
"""
import os
import threading
import time

from benchkit import localenv

localenv.load()
LOW_CLOCK_MHZ = 1000


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not an integer")


IDLE_MAX_MHZ = _env_int("TRAINBENCH_IDLE_MAX_MHZ", 1000)
BOOST_MHZ = _env_int("TRAINBENCH_BOOST_MHZ", 1700)
COOL_TO_C = _env_int("TRAINBENCH_COOL_TO_C", 60)
COOL_TIMEOUT_S = _env_int("TRAINBENCH_COOL_TIMEOUT_S", 180)   # how long a launch waits for COOL_TO_C before going on


def guard_settings():
    """Thresholds in force and whether each comes from the environment (or trainbench.env)."""
    return {name: {"value": val, "from_env": bool(os.environ.get(f"TRAINBENCH_{name}"))}
            for name, val in (("IDLE_MAX_MHZ", IDLE_MAX_MHZ), ("BOOST_MHZ", BOOST_MHZ), ("COOL_TO_C", COOL_TO_C),
                              ("COOL_TIMEOUT_S", COOL_TIMEOUT_S))} | \
        {"LOW_CLOCK_MHZ": {"value": LOW_CLOCK_MHZ, "from_env": False}}


class ClockSampler:
    def __init__(self, interval_s=0.05):
        self.interval_s, self.samples = interval_s, []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
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


def idle_check(seconds=3.0, idle_max_mhz=None):
    idle_max_mhz = IDLE_MAX_MHZ if idle_max_mhz is None else idle_max_mhz
    s = ClockSampler(0.1).start()
    time.sleep(seconds)
    s.stop()
    clocks = [c for _, c in s.samples]
    return (min(clocks) <= idle_max_mhz if clocks else True), \
        f"idle SM clock min {min(clocks, default=0)} MHz, max {max(clocks, default=0)} MHz"
