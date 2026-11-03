"""Android target (Pixel 8a): adb access, Chrome on the phone, phone-side telemetry and guards.

Selected with TRAINBENCH_TARGET=android. The laptop only orchestrates: the page is served on the phone (Termux `vite preview`
on the phone's localhost:4185), Chrome runs on the phone, and telemetry is sampled by a shell loop on the phone that writes
to a file there; the laptop pulls the file after each launch. Nothing here locks clocks or changes the governor.

Readable as the adb shell user (checked 2026-11-02, Android 17, no root): Mali `cur_freq`, CPU `scaling_cur_freq`,
battery current/voltage/temperature (sysfs), `dumpsys thermalservice`. Not readable: thermal_zone sysfs, IIO power rails,
other Mali nodes. Power rails are taken from the Perfetto traces instead (`perfetto/`).
"""
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from benchkit import localenv  # noqa: F401  (loads trainbench.env)

ROOT = Path(__file__).resolve().parents[1]
ADB = os.environ.get("TRAINBENCH_ADB", str(ROOT / "tools" / "android" / "platform-tools" / "adb.exe"))
SERIAL = os.environ.get("TRAINBENCH_ANDROID_SERIAL", "")
CHROME_PKG = "com.android.chrome"
CHROME_ACTIVITY = "com.google.android.apps.chrome.Main"
CMDLINE_FILE = "/data/local/tmp/chrome-command-line"
TMP = "/data/local/tmp/trainbench"
GPU_FREQ = "/sys/class/misc/mali0/device/cur_freq"   # kHz

# Guards (calibrated on this phone; override in trainbench.env).
COOL_SKIN_C = float(os.environ.get("TRAINBENCH_ANDROID_COOL_SKIN_C", "34"))
COOL_GPU_C = float(os.environ.get("TRAINBENCH_ANDROID_COOL_GPU_C", "45"))
COOL_TIMEOUT_S = float(os.environ.get("TRAINBENCH_ANDROID_COOL_TIMEOUT_S", "600"))
IDLE_MAX_MHZ = float(os.environ.get("TRAINBENCH_ANDROID_IDLE_MAX_MHZ", "400"))
BOOST_MHZ = float(os.environ.get("TRAINBENCH_ANDROID_BOOST_MHZ", "700"))
LOW_CLOCK_MHZ = float(os.environ.get("TRAINBENCH_ANDROID_LOW_CLOCK_MHZ", "400"))


def is_android():
    return os.environ.get("TRAINBENCH_TARGET", "").lower() == "android"


def adb(*args, timeout=60, check=True):
    cmd = [ADB, *(["-s", SERIAL] if SERIAL else []), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stdout}{r.stderr}")
    return r.stdout


def sh(cmd, timeout=60, check=True):
    return adb("shell", cmd, timeout=timeout, check=check)


def push_text(text, remote):
    """Write text to a phone file through a local temporary file (avoids shell quoting)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", newline="\n", encoding="utf-8") as f:
        f.write(text)
        local = f.name
    try:
        adb("push", local, remote)
    finally:
        os.unlink(local)


# ---- device facts ------------------------------------------------------------------------------------------------
PROPS = ["ro.product.model", "ro.product.device", "ro.build.fingerprint", "ro.build.version.release",
         "ro.build.version.sdk", "ro.soc.model", "ro.hardware.egl", "ro.board.platform", "ro.build.type",
         "ro.boot.verifiedbootstate"]


def device_info():
    out = sh("; ".join(f"getprop {p}" for p in PROPS)).splitlines()
    info = dict(zip(PROPS, (x.strip() for x in out)))
    info["mem_total_kb"] = int(re.search(r"MemTotal:\s+(\d+)", sh("cat /proc/meminfo")).group(1))
    info["cpu_count"] = int(sh("nproc").strip())
    info["cpu_max_khz"] = sorted({int(x) for x in sh("cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq").split()})
    return info


def chrome_version():
    m = re.search(r"versionName=(\S+)", sh(f"dumpsys package {CHROME_PKG}", check=False))
    return m.group(1) if m else None


def chrome_apk_sha256():
    paths = [ln.split(":", 1)[1].strip() for ln in sh(f"pm path {CHROME_PKG}").splitlines() if ln.startswith("package:")]
    base = next((p for p in paths if p.endswith("base.apk")), paths[0] if paths else None)
    return sh(f"sha256sum {base}").split()[0] if base else None


def device_id():
    return os.environ.get("TRAINBENCH_DEVICE") or "pixel8a-android17"


def browser_id():
    return f"chrome-android-{chrome_version()}"


SETTINGS = ["global stay_on_while_plugged_in", "system screen_brightness", "system screen_brightness_mode",
            "global window_animation_scale", "global transition_animation_scale", "global animator_duration_scale",
            "global low_power", "global settings_enable_monitor_phantom_procs", "global zen_mode",
            "system screen_off_timeout", "global wifi_on", "global bluetooth_on"]


def settings_snapshot():
    snap = {}
    for k in SETTINGS:
        ns, key = k.split()
        snap[f"{ns}/{key}"] = sh(f"settings get {ns} {key}", check=False).strip()
    snap["deviceidle_enabled"] = sh("dumpsys deviceidle enabled", check=False).strip()
    snap["global/debug_app"] = sh("settings get global debug_app", check=False).strip()
    snap["fixed_performance_mode"] = "not used"
    return snap


def apply_study_settings():
    """Idempotent: the settings the study needs (Phase 1 of the Android plan). Doze resets on reboot, so re-applied per stage."""
    for cmd in ("settings put system screen_brightness_mode 0", "settings put system screen_brightness 29",
                "settings put global window_animation_scale 0", "settings put global transition_animation_scale 0",
                "settings put global animator_duration_scale 0", "settings put global settings_enable_monitor_phantom_procs false",
                "settings put global low_power 0", "settings put system screen_off_timeout 1800000",
                "cmd notification set_dnd priority", "dumpsys deviceidle disable",
                # Chrome (user build, no root) reads CMDLINE_FILE only when it is the debug app; no wait-for-debugger (-w).
                f"am set-debug-app --persistent {CHROME_PKG}"):
        sh(cmd, check=False)


# ---- guards ------------------------------------------------------------------------------------------------------
def gpu_mhz():
    try:
        return int(sh(f"cat {GPU_FREQ}", timeout=10).strip()) / 1000
    except Exception:  # noqa: BLE001
        return None


def thermal():
    """{'status': int, 'temps': {name: degC}} from the HAL's current values."""
    out = sh("dumpsys thermalservice", check=False)
    st = re.search(r"Thermal Status:\s*(\d+)", out)
    part = out.split("Current temperatures from HAL:", 1)[-1].split("Current cooling devices", 1)[0]
    temps = {m.group(2): float(m.group(1)) for m in re.finditer(r"mValue=([-\d.]+), mType=-?\d+, mName=([^,]+),", part)}
    cdev = {m.group(2): int(m.group(1)) for m in re.finditer(r"CoolingDevice\{mValue=(\d+), mType=\d+, mName=([^}]+)\}", out)}
    return {"status": int(st.group(1)) if st else None, "temps": temps, "cooling": cdev}


def battery():
    vals = {}
    for f in ("capacity", "status", "current_now", "voltage_now", "temp"):
        vals[f] = sh(f"cat /sys/class/power_supply/battery/{f}", check=False).strip()
    return vals


def wait_cool(log=print):
    """Wait until skin and G3D are below the thresholds and the thermal status is 0 (the phone's --cool-to-c)."""
    t0, last = time.time(), None
    while True:
        th = thermal()
        skin, g3d = th["temps"].get("VIRTUAL-SKIN"), th["temps"].get("G3D")
        cdev = {k: v for k, v in th["cooling"].items() if k.startswith(("thermal-gpufreq", "thermal-cpufreq"))}
        last = {"status": th["status"], "skin_c": skin, "g3d_c": g3d, "cooling": cdev}
        if (th["status"] == 0 and cdev and not any(cdev.values()) and skin is not None and skin <= COOL_SKIN_C
                and g3d is not None and g3d <= COOL_GPU_C):
            return {"ok": True, "waited_s": round(time.time() - t0, 1), **last}
        if time.time() - t0 > COOL_TIMEOUT_S:
            log(f"cooling timeout: {last}")
            return {"ok": False, "waited_s": round(time.time() - t0, 1), **last}
        time.sleep(5)


def idle_check():
    """(ok, info): GPU idle frequency below IDLE_MAX_MHZ (median of 10 reads, 0.2 s apart)."""
    f = sorted(x for x in (gpu_mhz() for _ in range(10)) if x is not None and not time.sleep(0.2))
    med = f[len(f) // 2] if f else None
    return (med is not None and med <= IDLE_MAX_MHZ), {"gpu_idle_mhz_median": med, "limit": IDLE_MAX_MHZ}


# ---- phone-side telemetry ----------------------------------------------------------------------------------------
SAMPLER = r"""#!/system/bin/sh
# trainbench phone-side sampler: one CSV line per interval. sysfs values are read with the shell's `read` builtin (no
# process per value); the thermal HAL (dumpsys, one process) is read every 20th line.
out=$1; dt=$2; i=0
S=/sys/devices/system/cpu; B=/sys/class/power_supply/battery
echo "epoch_s,gpu_khz,cpu0_khz,cpu4_khz,cpu8_khz,bat_ua,bat_uv,bat_temp_dc,thermal_status,skin_c,g3d_c,soc_c,gpu_cdev,cpu_cdev0,cpu_cdev1,cpu_cdev2" > $out
while [ -f $out.run ]; do
  t=${EPOCHREALTIME:-0}
  read g < /sys/class/misc/mali0/device/cur_freq; read c0 < $S/cpu0/cpufreq/scaling_cur_freq
  read c4 < $S/cpu4/cpufreq/scaling_cur_freq; read c8 < $S/cpu8/cpufreq/scaling_cur_freq
  read bi < $B/current_now; read bv < $B/voltage_now; read bt < $B/temp
  ts=; sk=; gc=; sc=; cd=; c0d=; c1d=; c2d=
  if [ $((i % 20)) -eq 0 ]; then
    d=$(dumpsys thermalservice | sed -n -e 's/^Thermal Status: \(.*\)/TS \1/p' \
      -e 's/.*CoolingDevice{mValue=\([0-9]*\), mType=[0-9]*, mName=thermal-gpufreq-0}.*/GCD \1/p' \
      -e 's/.*CoolingDevice{mValue=\([0-9]*\), mType=[0-9]*, mName=thermal-cpufreq-\([012]\)}.*/CCD\2 \1/p' \
      -e '/Current temperatures from HAL/,/Current cooling devices/s/.*mValue=\([-0-9.]*\), mType=[-0-9]*, mName=VIRTUAL-SKIN,.*/VIRTUAL-SKIN \1/p' \
      -e '/Current temperatures from HAL/,/Current cooling devices/s/.*mValue=\([-0-9.]*\), mType=[-0-9]*, mName=G3D,.*/G3D \1/p' \
      -e '/Current temperatures from HAL/,/Current cooling devices/s/.*mValue=\([-0-9.]*\), mType=[-0-9]*, mName=soc_therm,.*/soc_therm \1/p')
    ts=$(echo "$d" | sed -n 's/^TS //p'); sk=$(echo "$d" | sed -n 's/^VIRTUAL-SKIN //p')
    gc=$(echo "$d" | sed -n 's/^G3D //p'); sc=$(echo "$d" | sed -n 's/^soc_therm //p')
    cd=$(echo "$d" | sed -n 's/^GCD //p'); c0d=$(echo "$d" | sed -n 's/^CCD0 //p')
    c1d=$(echo "$d" | sed -n 's/^CCD1 //p'); c2d=$(echo "$d" | sed -n 's/^CCD2 //p')
  fi
  echo "$t,$g,$c0,$c4,$c8,$bi,$bv,$bt,$ts,$sk,$gc,$sc,$cd,$c0d,$c1d,$c2d" >> $out
  i=$((i+1)); sleep $dt
done
"""


class AndroidTelemetry:
    """Start/stop the phone-side sampler around one launch; the CSV is pulled to the laptop afterwards."""

    def __init__(self, interval_s=0.1):
        self.interval_s, self.remote = interval_s, None
        sh(f"mkdir -p {TMP}")
        push_text(SAMPLER, f"{TMP}/sampler.sh")

    def start(self, tag):
        self.remote = f"{TMP}/telemetry_{tag}.csv"
        sh(f"touch {self.remote}.run; (nohup sh {TMP}/sampler.sh {self.remote} {self.interval_s} >/dev/null 2>&1 &)")

    def stop(self, local_path):
        if self.remote is None:
            return None
        sh(f"rm -f {self.remote}.run")
        time.sleep(max(1.0, self.interval_s * 3))
        adb("pull", self.remote, str(local_path))
        sh(f"rm -f {self.remote}")
        self.remote = None
        return local_path


class AndroidClockSampler:
    """Counterpart of clockguard.ClockSampler for the Mali GPU: summarises gpu_khz from the telemetry CSV."""

    @staticmethod
    def summarize(csv_path, t0_s=None, t1_s=None):
        import pandas as pd
        d = pd.read_csv(csv_path)
        if t0_s is not None:
            d = d[(d.epoch_s >= t0_s) & (d.epoch_s <= t1_s)]
        f = d.gpu_khz.dropna() / 1000
        return {"gpu_mhz_median": float(f.median()) if len(f) else None, "gpu_mhz_max": float(f.max()) if len(f) else None,
                "samples": int(len(f)), "throttled": bool((d.thermal_status.dropna() > 0).any())}


# ---- Chrome ------------------------------------------------------------------------------------------------------
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def chrome_command_line(flags):
    """Chrome on Android reads its flags from this file (needs chrome://flags 'command line on non-rooted devices')."""
    push_text("_ " + " ".join(flags) + "\n", CMDLINE_FILE)


def start_chrome(flags, url="about:blank"):
    """Fresh Chrome processes: force-stop, new command line, start; returns the local port forwarded to DevTools."""
    sh(f"am force-stop {CHROME_PKG}")
    chrome_command_line(flags)
    sh(f"am start -n {CHROME_PKG}/{CHROME_ACTIVITY} -a android.intent.action.VIEW -d '{url}'")
    port = free_port()
    adb("forward", f"tcp:{port}", "localabstract:chrome_devtools_remote")
    import urllib.request
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                json.load(r)
                return port
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("Chrome DevTools did not come up on the phone")


def stop_chrome(port=None):
    sh(f"am force-stop {CHROME_PKG}", check=False)
    if port:
        adb("forward", "--remove", f"tcp:{port}", check=False)


def server_ok(port=4185):
    """The page server must run on the phone (Termux); checked from the phone itself with nc (no curl on Android)."""
    # vite answers 403 without a matching Host; toybox nc quits at stdin EOF, so stdin stays open until the reply.
    out = sh(f"(printf 'HEAD / HTTP/1.1\\r\\nHost: localhost:{port}\\r\\nConnection: close\\r\\n\\r\\n'; sleep 2) "
             f"| nc -w 3 127.0.0.1 {port}", check=False)
    return ("same-origin" in out.lower() and "require-corp" in out.lower()), out[:400]

# ---- per-launch telemetry with the NvmlTelemetry interface (run_benchmark) -----------------------------------------
def phone_clock_offset_s():
    """phone epoch - laptop epoch (s), from the midpoint of one adb round trip; recorded with every launch."""
    t0 = time.time()
    p = float(sh("echo $EPOCHREALTIME").strip())
    t1 = time.time()
    return p - (t0 + t1) / 2, t1 - t0


class AndroidRunTelemetry:
    """Same interface as telemetry.NvmlTelemetry: start/mark/stop -> dict of sample lists.

    samples: the phone-side sampler rows (GPU/CPU frequency, battery current and voltage, thermals), with t_epoch_ms on the
    laptop clock (phone time minus the measured offset). util/pcie/chrome/system_cpu are NVIDIA/desktop series and stay
    empty, with the reason in field_errors. stop() caches its result, so the low-clock guard and the writer share one pull.
    """

    def __init__(self, interval_s=0.1, out_dir=None):
        self.sampler = AndroidTelemetry(interval_s)
        self.out_dir = Path(out_dir) if out_dir else Path(os.environ.get("TEMP", "."))
        self._n, self._cache = 0, None

    def watch_chrome(self, *_):
        pass

    def start(self):
        self._n += 1
        self._cache, self.marks = None, []
        self.offset_s, self.rtt_s = phone_clock_offset_s()
        self.tag = f"{int(time.time())}_{self._n}"
        self.sampler.start(self.tag)

    def mark(self, name):
        self.marks.append({"t_epoch_ms": time.time() * 1000.0, "mark": name})

    def stop(self):
        if self._cache is not None:
            return self._cache
        import pandas as pd
        local = self.out_dir / f"android_telemetry_{self.tag}.csv"
        self.sampler.stop(local)
        d = pd.read_csv(local)
        local.unlink(missing_ok=True)
        d["t_epoch_ms"] = (d.pop("epoch_s") - self.offset_s) * 1000.0
        d["gpu_mhz"] = d.pop("gpu_khz") / 1000.0
        samples = d.astype(object).where(d.notna(), None).to_dict("records")
        why = "not available on Android (no NVML / desktop process table)"
        self._cache = {"samples": samples, "util": [], "pcie": [], "chrome": [], "system_cpu": [], "marks": self.marks,
                       "errors": [], "field_errors": {"util": why, "pcie": why, "chrome": why, "system_cpu": why,
                                                      "clock_offset_s": self.offset_s, "clock_offset_rtt_s": self.rtt_s}}
        return self._cache


class AndroidLowClock:
    """clockguard.ClockSampler counterpart. stop() -> max Mali GPU MHz after experiment_start (recorded, not a guard: Mali
    DVFS keeps light loads at low clocks, so a low maximum is the device's normal behaviour, checked 2026-11-02 on C1).
    The phone's guard is thermal throttling instead: `throttle` is set when the GPU cooling device (thermal-gpufreq-0)
    capped the clock or the thermal status was above 0 during the experiment. For WASM launches (cpu=True) the three CPU
    cooling devices (thermal-cpufreq-0/1/2) are checked instead of the GPU one."""

    def __init__(self, telemetry, cpu=False):
        self.telemetry = telemetry
        self.cdev_cols = ["cpu_cdev0", "cpu_cdev1", "cpu_cdev2"] if cpu else ["gpu_cdev"]
        self.throttle = None

    def start(self):
        return self

    def stop(self):
        data = self.telemetry.stop()
        m = {x["mark"]: x["t_epoch_ms"] for x in data["marks"]}
        t0 = m.get("experiment_start", 0)
        run = [s for s in data["samples"] if s["t_epoch_ms"] >= t0]
        cdev = [s[c] for s in run for c in self.cdev_cols if s.get(c) is not None]
        status = [s["thermal_status"] for s in run if s.get("thermal_status") is not None]
        self.throttle = {"devices": self.cdev_cols, "cdev_max": max(cdev) if cdev else None,
                         "thermal_status_max": max(status) if status else None, "reads": len(status)}
        self.throttle["throttled"] = bool((cdev and max(cdev) > 0) or (status and max(status) > 0))
        f = [s["gpu_mhz"] for s in run if s.get("gpu_mhz") is not None]
        return max(f) if f else None


# ---- environment lock items ------------------------------------------------------------------------------------------
def lock_items(scope="full"):
    info = device_info()
    items = {"fingerprint": info["ro.build.fingerprint"], "model": info["ro.product.model"],
             "chrome_version": chrome_version()}
    if scope != "launch":
        items.update({"device": info, "chrome_apk_sha256": chrome_apk_sha256(),
                      "settings": {k: v for k, v in settings_snapshot().items()}})
    return items


# ---- Chrome page (same yield as benchkit.chrome.chrome_page) ---------------------------------------------------------
from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def chrome_page_android(playwright, url, flags, log_path=None, cdp_setup=None, ready_global="__bench"):
    """Fresh Chrome processes on the phone (force-stop + command line + start), driven over adb-forwarded CDP.

    The page URL is the phone's own localhost (Termux server). Yields (page, browser_version)."""
    import asyncio
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    port = await asyncio.to_thread(start_chrome, list(flags))
    browser, teardown = None, None
    try:
        browser = await playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}", timeout=30_000)
        context = browser.contexts[0]
        old = list(context.pages)
        page = await context.new_page()
        for p in old:
            try:
                await p.close()
            except Exception:  # noqa: BLE001
                pass
        if log_file:
            page.on("console", lambda m: log_file.write(f"[console.{m.type}] {m.text}\n"))
            page.on("pageerror", lambda e: log_file.write(f"[pageerror] {e}\n"))
            page.on("crash", lambda _: log_file.write("[crash] renderer crashed\n"))
        cdp = await context.new_cdp_session(page)
        version = (await cdp.send("Browser.getVersion"))["product"]
        if cdp_setup:
            teardown = await cdp_setup(cdp)
        await page.goto(url, wait_until="load", timeout=120_000)
        await page.wait_for_function(f"() => window.{ready_global} !== undefined", timeout=120_000)
        if not await page.evaluate(f"() => window.{ready_global}.ready"):
            raise RuntimeError(f"page failed to initialise: {await page.evaluate(f'() => window.{ready_global}.error')}")
        yield page, version
    finally:
        if teardown is not None:
            try:
                await asyncio.wait_for(teardown(), timeout=120)
            except Exception as e:  # noqa: BLE001
                if log_file:
                    log_file.write(f"[cdp teardown failed] {type(e).__name__}: {e}\n")
        if browser is not None:
            try:
                await asyncio.wait_for(browser.close(), timeout=30)
            except Exception:  # noqa: BLE001
                pass
        await asyncio.to_thread(stop_chrome, port)
        if log_file:
            log_file.close()
