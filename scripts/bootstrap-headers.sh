#!/usr/bin/env bash
# Fetch X11/GL development headers into third_party/sysroot without root.
# Only needed when the -dev packages are not installed system-wide
# (normal route: sudo apt install libx11-dev libxrandr-dev libxinerama-dev
#  libxcursor-dev libxi-dev libgl-dev libxkbcommon-dev).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/third_party/sysroot"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
apt-get download libx11-dev x11proto-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev libxext-dev libxrender-dev libxfixes-dev \
    libxkbcommon-dev libgl-dev libglx-dev libxcb1-dev libxau-dev libxdmcp-dev
mkdir -p "$dest"
for d in *.deb; do dpkg-deb -x "$d" "$dest"; done

# Point the dev .so symlinks at the runtime libraries already on the system.
for l in "$dest"/usr/lib/*/lib*.so; do
    [ -L "$l" ] || continue
    tgt="$(readlink "$l")"
    [ -e "$(dirname "$l")/$tgt" ] || ln -sf "/usr/lib/$(basename "$(dirname "$l")")/$tgt" "$l"
done
echo "Headers extracted to $dest/usr/include"
