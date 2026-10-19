"""Launch a fresh pinned Firefox process per experiment and drive the benchmark page (counterpart of chrome.py).

Stock Firefox has no CDP, so it is driven with WebDriver (geckodriver + Selenium): classic WebDriver only to
start the session and navigate, WebDriver BiDi for everything during the experiment (script.callFunction with
awaitPromise, input.performActions, console events). BiDi commands are asynchronous, so the responsiveness
modes can click while runExperiment is pending, as page.evaluate + page.mouse do on Chrome.

firefox_page() yields a page object with the subset of Playwright's Page that the drivers use
(evaluate, locator().scroll_into_view_if_needed / bounding_box, mouse.click, viewport_size), so
run_benchmark.run_one is the same code for both browsers.
"""
import asyncio
import json
import math
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Firefox 158.0 (BuildID 20261007201254), copied from the installed release into the project, with
# distribution/policies.json disabling updates, telemetry and studies. The installed copy may auto-update;
# this one cannot. Path, version and file hashes are part of env.lock.json.
PINNED_FIREFOX = ROOT / "tools" / "firefox-158.0" / "firefox.exe"
PINNED_GECKODRIVER = ROOT / "tools" / "geckodriver-0.37.1" / "geckodriver.exe"

# Prefs written into every fresh profile, recorded in env.json. Each one is the counterpart of a Chrome flag
# in chrome.CHROME_FLAGS (or of a default Playwright passes to Chrome); everything else stays at Firefox's stock
# value, including WebGPU and timer-precision prefs. geckodriver adds its own automation prefs on top; the
# profile's final user.js is saved per launch (logs/<tag>.user.js).
FIREFOX_PREFS = {
    # --no-first-run, --no-default-browser-check
    "browser.shell.checkDefaultBrowser": False,
    "browser.aboutwelcome.enabled": False,
    "browser.startup.homepage_override.mstone": "ignore",
    "datareporting.policy.dataSubmissionPolicyBypassNotification": True,
    "toolkit.telemetry.reportingpolicy.firstRun": False,
    # No background work that Chrome for Testing doesn't do either: updates, studies, telemetry upload,
    # captive-portal / connectivity checks.
    "app.update.auto": False,
    "app.normandy.enabled": False,
    "app.shield.optoutstudies.enabled": False,
    "datareporting.healthreport.uploadEnabled": False,
    "datareporting.policy.dataSubmissionEnabled": False,
    "toolkit.telemetry.unified": False,
    "network.captive-portal-service.enabled": False,
    "network.connectivity-service.enabled": False,
    "extensions.update.enabled": False,
    # --disable-background-timer-throttling, --disable-renderer-backgrounding,
    # --disable-backgrounding-occluded-windows, --disable-features=CalculateNativeWinOcclusion
    "dom.min_background_timeout_value": 4,
    "dom.timeout.enable_budget_timer_throttling": False,
    "widget.windows.window_occlusion_tracking.enabled": False,
    # --disable-gpu-shader-disk-cache: nothing to set; every launch gets a brand-new profile, which is where
    # Firefox keeps its WebGL/WebGPU shader caches.
}

VIEWPORT = {"width": 1280, "height": 900}

# Runs a page expression of the form "cfg => ..." or "() => ..." with one JSON argument and returns the result
# as a JSON string. Non-finite numbers are tagged so NaN / Infinity survive (Playwright keeps them; plain JSON
# would turn them into null).
_CALL = """async (src, argJson) => {
  const fn = (0, eval)("(" + src + ")");
  const v = await fn(argJson === undefined ? undefined : JSON.parse(argJson));
  return JSON.stringify(v === undefined ? null : v,
    (k, x) => (typeof x === "number" && !Number.isFinite(x)) ? {__nonfinite__: String(x)} : x);
}"""


def _restore(o):
    if isinstance(o, dict):
        if len(o) == 1 and "__nonfinite__" in o:
            return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[o["__nonfinite__"]]
        return {k: _restore(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_restore(v) for v in o]
    return o


def find_firefox():
    """Return the pinned Firefox executable (see PINNED_FIREFOX), or None if missing."""
    return str(PINNED_FIREFOX) if PINNED_FIREFOX.is_file() and PINNED_GECKODRIVER.is_file() else None


class _Mouse:
    def __init__(self, page):
        self._page = page

    async def click(self, x, y):
        actions = [{"type": "pointer", "id": "mouse", "parameters": {"pointerType": "mouse"}, "actions": [
            {"type": "pointerMove", "x": int(round(x)), "y": int(round(y)), "origin": "viewport"},
            {"type": "pointerDown", "button": 0}, {"type": "pointerUp", "button": 0}]}]
        await self._page._bidi("input.performActions", {"context": self._page.context_id, "actions": actions})


class _Locator:
    def __init__(self, page, selector):
        self._page, self._sel = page, selector

    async def scroll_into_view_if_needed(self):
        await self._page.evaluate("s => { document.querySelector(s).scrollIntoView({block: 'nearest'}); }", self._sel)

    async def bounding_box(self):
        return await self._page.evaluate(
            "s => { const r = document.querySelector(s).getBoundingClientRect();"
            " return {x: r.x, y: r.y, width: r.width, height: r.height}; }", self._sel)


class FirefoxPage:
    """The Playwright Page subset used by run_benchmark / run_readback_micro / probes."""

    def __init__(self, driver, context_id):
        self.driver, self.context_id = driver, context_id
        self.viewport_size = dict(VIEWPORT)
        self.mouse = _Mouse(self)

    def _send(self, method, params, timeout_s):
        conn = self.driver._websocket_connection
        conn.response_wait_timeout = timeout_s
        reply = conn.send_cmd(method, params)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']}: {reply.get('message', '')}")
        return reply["result"]

    async def _bidi(self, method, params, timeout_s=60):
        return await asyncio.to_thread(self._send, method, params, timeout_s)

    async def evaluate(self, expression, arg=None, timeout_s=24 * 3600):
        args = [{"type": "string", "value": expression}]
        if arg is not None:
            args.append({"type": "string", "value": json.dumps(arg, allow_nan=False)})
        # The caller bounds the wait (asyncio.wait_for); timeout_s is only the transport's upper limit.
        r = await self._bidi("script.callFunction", {
            "functionDeclaration": _CALL, "awaitPromise": True, "target": {"context": self.context_id},
            "arguments": args}, timeout_s)
        if r.get("type") == "exception":
            d = r.get("exceptionDetails", {})
            raise RuntimeError(f"page exception: {d.get('text')}")
        return _restore(json.loads(r["result"]["value"]))

    def locator(self, selector):
        return _Locator(self, selector)


@asynccontextmanager
async def firefox_page(executable_path, url, headless, prefs=None, log_path=None, on_launch=None, ready_global="__bench"):
    """Yield (page, browser_version) for a brand-new Firefox process with an empty profile.

    on_launch(profile_dir) is called right after the browser starts (the directory is unique per launch and
    appears as `-profile <dir>` on the parent process's command line, which identifies this launch's processes).
    prefs replaces FIREFOX_PREFS when given.
    """
    from selenium import webdriver
    from selenium.webdriver.firefox.service import Service

    profile_dir = tempfile.mkdtemp(prefix="trainbench-ffprofile-")
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    log_lock = threading.Lock()

    def log(line):
        if log_file:
            with log_lock:
                log_file.write(line + "\n")

    driver = None
    try:
        opts = webdriver.FirefoxOptions()
        opts.binary_location = executable_path
        opts.enable_bidi = True
        opts.add_argument("-profile")
        opts.add_argument(profile_dir)
        if headless:
            opts.add_argument("-headless")
        for k, v in (FIREFOX_PREFS if prefs is None else prefs).items():
            opts.set_preference(k, v)
        service = Service(executable_path=str(PINNED_GECKODRIVER),
                          log_output=str(Path(log_path).with_suffix(".geckodriver.log")) if log_path else None)
        driver = await asyncio.to_thread(webdriver.Firefox, options=opts, service=service)
        if on_launch:
            on_launch(profile_dir)
        if log_file:
            driver.script.add_console_message_handler(lambda e: log(f"[console.{e.method or e.level}] {e.text}"))
            driver.script.add_javascript_error_handler(lambda e: log(f"[pageerror] {e.text}"))
        caps = driver.capabilities
        version = f"Firefox/{caps.get('browserVersion')} (build {caps.get('moz:buildID')})"
        page = FirefoxPage(driver, driver.current_window_handle)
        await page._bidi("browsingContext.setViewport", {"context": page.context_id, "viewport": VIEWPORT})
        await page._bidi("browsingContext.navigate", {"context": page.context_id, "url": url, "wait": "complete"})
        ready = await asyncio.wait_for(page.evaluate(
            f"() => new Promise((res, rej) => {{ const t0 = performance.now(); (function poll() {{"
            f" if (window.{ready_global} !== undefined) return res(true);"
            f" if (performance.now() - t0 > 60000) return rej(new Error('timeout waiting for {ready_global}'));"
            f" setTimeout(poll, 50); }})(); }})"), timeout=90)
        if ready and not await page.evaluate(f"() => window.{ready_global}.ready"):
            raise RuntimeError(f"page failed to initialise: {await page.evaluate(f'() => window.{ready_global}.error')}")
        if log_path:
            uj = Path(profile_dir) / "user.js"
            if uj.exists():
                shutil.copyfile(uj, Path(log_path).with_suffix(".user.js"))
        yield page, version
    finally:
        if driver is not None:
            try:
                await asyncio.wait_for(asyncio.to_thread(driver.quit), timeout=30)
            except Exception:
                pass
        if log_file:
            log_file.close()
        shutil.rmtree(profile_dir, ignore_errors=True)


def version():
    """Pinned Firefox version (application.ini next to firefox.exe), or None."""
    ini = PINNED_FIREFOX.parent / "application.ini"
    if not ini.exists():
        return None
    for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Version="):
            return line.split("=", 1)[1].strip()
    return None
