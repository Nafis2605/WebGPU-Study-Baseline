"""Launch a fresh Chrome for Testing process per experiment and drive the benchmark page (sibling port)."""
import asyncio
import os
import platform
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Flags applied to every launch. Each one is recorded in env.json.
CHROME_FLAGS = [
    "--no-first-run",
    "--no-default-browser-check",
    # WebGPU: unquantized timestamp queries and full adapter info (vendor/description).
    "--enable-webgpu-developer-features",
    # Cold GPU state: no shader/pipeline cache survives between launches.
    "--disable-gpu-shader-disk-cache",
    # Keep timers and rendering at full rate even if the window is occluded.
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=CalculateNativeWinOcclusion",
]


# The study's browser is pinned: Chrome for Testing 157.0.8081.0 (win64), unpacked in the project.
# It never auto-updates. There is deliberately no fallback to an installed Chrome (Chrome Dev updates
# itself); the path and version are part of env.lock.json and verified before every launch.
# macOS: the same version's mac-arm64 build, unpacked to tools/chrome-for-testing/chrome-mac-arm64/. CHROME_FLAGS stay
# identical on both OSes (CalculateNativeWinOcclusion is a no-op on macOS) so the recorded flag set is the same.
_CFT = Path(__file__).resolve().parents[1] / "tools" / "chrome-for-testing"
if platform.system() == "Darwin":
    PINNED_CHROME_APP = _CFT / "chrome-mac-arm64" / "Google Chrome for Testing.app"
    PINNED_CHROME = PINNED_CHROME_APP / "Contents" / "MacOS" / "Google Chrome for Testing"
else:
    PINNED_CHROME_APP = None
    PINNED_CHROME = _CFT / "chrome-win64" / "chrome.exe"


def find_chrome_dev():
    """Return the pinned Chrome for Testing executable (see PINNED_CHROME), or None if missing."""
    return str(PINNED_CHROME) if PINNED_CHROME.is_file() else None


@asynccontextmanager
async def chrome_page(playwright, executable_path, url, headless, extra_flags=(), log_path=None, on_launch=None,
                      cdp_setup=None, ready_global="__bench", user_data_dir=None, flags=None):
    """Yield (page, browser_version) for a brand-new Chrome process with an empty profile.

    on_launch(user_data_dir) is called right after the browser starts (the directory name is
    unique per launch, so it identifies this launch's Chrome processes).
    cdp_setup(cdp) (async) runs on the page's CDP session before navigation; it may return an
    async teardown, which runs after the caller's block and before the browser closes.
    ready_global names the window object the page exposes once initialised ({ready, error}).
    user_data_dir: reuse this profile directory (kept afterwards; the caller owns it) instead of a new
    empty one; used only by the *-diskcache warm-up configs. flags replaces CHROME_FLAGS when given.
    """
    own_profile = user_data_dir is None
    if own_profile:
        user_data_dir = tempfile.mkdtemp(prefix="trainbench-profile-")
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    context = None
    teardown = None
    try:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir,
            executable_path=executable_path,
            headless=headless,
            args=list(CHROME_FLAGS if flags is None else flags) + list(extra_flags),
            viewport={"width": 1280, "height": 900},
        )
        if on_launch:
            on_launch(user_data_dir)
        page = context.pages[0] if context.pages else await context.new_page()
        if log_file:
            page.on("console", lambda m: log_file.write(f"[console.{m.type}] {m.text}\n"))
            page.on("pageerror", lambda e: log_file.write(f"[pageerror] {e}\n"))
            page.on("crash", lambda _: log_file.write("[crash] renderer crashed\n"))
        cdp = await context.new_cdp_session(page)
        version = (await cdp.send("Browser.getVersion"))["product"]
        if cdp_setup:
            teardown = await cdp_setup(cdp)
        await page.goto(url, wait_until="load", timeout=60_000)
        await page.wait_for_function(f"() => window.{ready_global} !== undefined", timeout=60_000)
        if not await page.evaluate(f"() => window.{ready_global}.ready"):
            raise RuntimeError(f"page failed to initialise: {await page.evaluate(f'() => window.{ready_global}.error')}")
        yield page, version
    finally:
        if teardown is not None:
            try:
                await asyncio.wait_for(teardown(), timeout=120)
            except Exception as e:
                if log_file:
                    log_file.write(f"[cdp teardown failed] {type(e).__name__}: {e}\n")
        if context is not None:
            try:
                await asyncio.wait_for(context.close(), timeout=30)
            except Exception:
                pass
        if log_file:
            log_file.close()
        if own_profile:
            shutil.rmtree(user_data_dir, ignore_errors=True)


def version():
    """Pinned Chrome for Testing version (Windows: the *.manifest file next to chrome.exe; macOS: the app's
    Info.plist CFBundleShortVersionString), or None."""
    if PINNED_CHROME_APP is not None:
        import plistlib
        plist = PINNED_CHROME_APP / "Contents" / "Info.plist"
        if not plist.exists():
            return None
        with open(plist, "rb") as f:
            return plistlib.load(f).get("CFBundleShortVersionString")
    return next((p.stem.replace(".manifest", "") for p in PINNED_CHROME.parent.glob("*.manifest")), None)
