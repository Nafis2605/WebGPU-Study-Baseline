"""
Stage the page server for Termux on the phone (Android target; the laptop never serves the page).

Copies what `vite preview` needs (dist/, public/ without tflite/, vite.config.js, package files, scripts/copy-tfjs-wasm.js)
with a SHA256SUMS list to /sdcard/trainbench/serve, plus scripts/termux_serve.sh. In Termux the user then runs
    bash /sdcard/trainbench/termux_serve.sh setup     # once: node, copy to ~/trainbench, sha256 check, npm ci
    bash /sdcard/trainbench/termux_serve.sh serve     # vite preview on localhost:4185 (keep Termux open)
The phone-side copy in /sdcard is checked here with toybox sha256sum; the Termux copy is checked by the setup step.

    .venv/Scripts/python scripts/android_stage_server.py [--no-push]
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchkit import android  # noqa: E402

REMOTE = "/sdcard/trainbench"
FILES = ["vite.config.js", "package.json", "package-lock.json", "scripts/copy-tfjs-wasm.js"]
DIRS = ["dist", "public"]
SKIP = ("public/tflite/",)   # TFLite models go to /data/local/tmp with the TFLite runner, not to the page server


def staged():
    out = [ROOT / f for f in FILES]
    for d in DIRS:
        out += sorted(p for p in (ROOT / d).rglob("*") if p.is_file())
    return [p for p in out if not p.relative_to(ROOT).as_posix().startswith(SKIP)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    files = staged()
    sums = "".join(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files)
    out = ROOT / "SHA256SUMS.serve"
    out.write_text(sums, encoding="utf-8", newline="\n")
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB")
    if a.no_push:
        return
    android.sh(f"rm -rf {REMOTE}/serve && mkdir -p {REMOTE}/serve")
    for top in FILES + DIRS:
        src = ROOT / top
        dst = f"{REMOTE}/serve/{top}"
        android.sh(f"mkdir -p {Path(dst).parent.as_posix()}")
        android.adb("push", str(src), Path(dst).parent.as_posix() + "/" if src.is_dir() else dst, timeout=1800)
    android.sh(f"rm -rf {REMOTE}/serve/public/tflite", check=False)
    android.adb("push", str(out), f"{REMOTE}/serve/SHA256SUMS")
    android.adb("push", str(ROOT / "scripts" / "termux_serve.sh"), f"{REMOTE}/termux_serve.sh")
    r = subprocess.run([android.ADB, *(["-s", android.SERIAL] if android.SERIAL else []), "shell",
                        f"cd {REMOTE}/serve && sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$'; echo rc=$?"],
                       capture_output=True, text=True, timeout=1800)
    bad = [ln for ln in r.stdout.splitlines() if ln and not ln.startswith("rc=")]
    print("phone copy sha256:", "OK" if not bad else f"{len(bad)} mismatches: {bad[:5]}")
    out.unlink()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
