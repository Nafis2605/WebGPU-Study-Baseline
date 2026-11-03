#!/data/data/com.termux/files/usr/bin/bash
# Page server on the phone (Android target). Run inside Termux, not over adb:
#   bash /sdcard/trainbench/termux_serve.sh setup   # once after each staging (scripts/android_stage_server.py)
#   bash /sdcard/trainbench/termux_serve.sh serve   # vite preview on localhost:4185, COOP/COEP as on the laptop
# setup installs Node.js LTS and rsync with pkg, copies the staged tree to ~/trainbench (internal storage, not the
# /sdcard FUSE mount), checks every file against SHA256SUMS, and installs the pinned npm packages (npm ci).
set -euo pipefail
SRC=/sdcard/trainbench/serve
DST="$HOME/trainbench"

case "${1:-serve}" in
  setup)
    pkg update -y
    pkg install -y nodejs-lts rsync coreutils
    mkdir -p "$DST"
    rsync -a --delete "$SRC/" "$DST/"
    cd "$DST"
    sha256sum -c --quiet SHA256SUMS && echo "SHA256 OK: $(wc -l < SHA256SUMS) files"
    npm ci --no-audit --no-fund
    sha256sum -c --quiet SHA256SUMS && echo "SHA256 OK after npm ci"   # postinstall rewrites public/tfjs-wasm
    { echo "node $(node --version)"; echo "vite $(npx vite --version)"; } | tee "$DST/server.versions"
    cp "$DST/server.versions" /sdcard/trainbench/server.versions
    echo "setup done; now run: bash $0 serve"
    ;;
  serve)
    termux-wake-lock || true
    cd "$DST"
    exec npx vite preview
    ;;
  *)
    echo "usage: $0 setup|serve"; exit 2 ;;
esac
