"""
The browser must render on the GPU that is being measured.

On a laptop with an integrated and a discrete GPU, Chrome can start on the integrated one while the telemetry
(NVML) and the clock guards still read the NVIDIA GPU, so a run would be silently invalid. require_gpu() compares the
WebGPU adapter and the WebGL renderer that the page reports (window.__bench.probeEnv()) with the required vendor.

TRAINBENCH_REQUIRE_GPU_VENDOR   vendor name that must appear in the adapter and renderer (default: nvidia);
                                "any" turns the check off (recorded in run.json).
The check applies to the webgl and webgpu backends only; it fails when the page reports nothing to compare.
"""
import os

from benchkit import localenv

localenv.load()
DEFAULT_VENDOR = "nvidia"


class WrongGpuError(RuntimeError):
    """The page runs on a GPU other than the required one."""


def required_vendor():
    v = (os.environ.get("TRAINBENCH_REQUIRE_GPU_VENDOR") or DEFAULT_VENDOR).strip().lower()
    return None if v in ("any", "none", "off") else v


def require_gpu(page_env, backend, vendor=None):
    """Raise WrongGpuError unless the adapter (webgpu) or the renderer (webgl) is the required vendor."""
    vendor = required_vendor() if vendor is None else vendor
    if vendor is None or backend not in ("webgpu", "webgl"):
        return
    page_env = page_env or {}
    if backend == "webgpu":
        a = page_env.get("webgpu_adapter") or {}
        seen = " ".join(str(a.get(k) or "") for k in ("vendor", "description", "architecture")).lower()
        what = f"WebGPU adapter {a.get('vendor')!r} / {a.get('description')!r}"
    else:
        r = (page_env.get("webgl") or {}).get("renderer")
        seen = str(r or "").lower()
        what = f"WebGL renderer {r!r}"
    if vendor not in seen:
        raise WrongGpuError(f"{what} is not {vendor!r}. Set the Windows graphics preference of the browser executable "
                            f"to High performance (Settings > System > Display > Graphics).")
