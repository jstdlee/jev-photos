#!/usr/bin/env bash
# Start a private virtual X display for GUI tests (so tests never touch the desktop you are working on).
#   scripts/test-display.sh            -> Xvfb on :99 (fetched into third_party/xvfb without root)
#   DISPLAY=:99 __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 build/jev-photos --ui-script "tab:photos,j,Return,shot:/tmp/a.png"
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
T="$root/third_party/xvfb"
if [ ! -x "$T/usr/bin/Xvfb" ]; then
    tmp="$(mktemp -d)"; (cd "$tmp" && apt-get download xvfb libxfont2 xserver-common libfontenc1 libxkbfile1 >/dev/null)
    mkdir -p "$T"; for d in "$tmp"/*.deb; do dpkg-deb -x "$d" "$T"; done; rm -rf "$tmp"
fi
n="${1:-99}"
[ -S "/tmp/.X11-unix/X$n" ] && { echo "display :$n already running"; exit 0; }
LD_LIBRARY_PATH="$T/usr/lib/$(uname -m)-linux-gnu" setsid nohup "$T/usr/bin/Xvfb" ":$n" -screen 0 1600x1000x24 -nolisten tcp \
    -xkbdir /usr/share/X11/xkb >/dev/null 2>&1 &
sleep 1; echo "virtual display :$n ready"
