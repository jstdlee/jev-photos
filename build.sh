#!/usr/bin/env bash
# One-shot build: fetch deps, fetch X11/GL headers locally if missing, compile.
set -euo pipefail
cd "$(dirname "$0")"
scripts/fetch-deps.sh
scripts/fetch-models.sh   # CLIP ViT-B/32 (~600 MB, once); "scripts/fetch-models.sh l14" adds the better ViT-L/14
if [ ! -f /usr/include/X11/Xlib.h ] || [ ! -f /usr/include/X11/extensions/Xrandr.h ]; then
    [ -f third_party/sysroot/usr/include/X11/Xlib.h ] || scripts/bootstrap-headers.sh
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "Built: $(pwd)/build/jev-photos"
