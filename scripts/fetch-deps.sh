#!/usr/bin/env bash
# Fetch pinned third-party sources into third_party/: Dear ImGui, GLFW, stb, SQLite, nlohmann/json, xxHash, ONNX Runtime.
#   scripts/fetch-deps.sh            for this system
#   scripts/fetch-deps.sh windows    also the Windows ONNX Runtime (cross builds, and MSYS2 on Windows)
set -euo pipefail
target="${1:-}"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) target=windows ;; esac
unzip_to() {  # zip dir
    if command -v unzip >/dev/null; then unzip -q "$1" -d "$2"
    else python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; fi
}
tp="$(cd "$(dirname "$0")/.." && pwd)/third_party"
mkdir -p "$tp"

IMGUI_REV=09f7a0f062b902dc0b377439421abc58d369b1d9   # 1.93 WIP (dynamic fonts), same as gpu-hud
GLFW_REV=7b6aead9fb88b3623e3b3725ebb42670cbe4c579    # 3.4
STB_REV=2c980bb59875b0d32144a71867fbdebb2f77cd20
SQLITE_ZIP=2025/sqlite-amalgamation-3500400.zip      # 3.50.4
JSON_VER=v3.12.0
XXHASH_VER=v0.8.3
ORT_VER=1.22.0
FA_VER=6.7.2                                         # Font Awesome Free icon font (OFL 1.1 / CC BY 4.0)

fetch() {  # dir url rev
    if [ -d "$tp/$1/.git" ]; then return; fi
    git init -q "$tp/$1"
    git -C "$tp/$1" fetch -q --depth 1 "$2" "$3"
    git -C "$tp/$1" -c advice.detachedHead=false checkout -q FETCH_HEAD
}
fetch imgui https://github.com/ocornut/imgui.git "$IMGUI_REV"
fetch glfw https://github.com/glfw/glfw.git "$GLFW_REV"

mkdir -p "$tp/stb"
for h in stb_image.h stb_image_write.h stb_image_resize2.h; do
    [ -f "$tp/stb/$h" ] || curl -fsSL -o "$tp/stb/$h" "https://raw.githubusercontent.com/nothings/stb/$STB_REV/$h"
done

if [ ! -f "$tp/sqlite/sqlite3.c" ]; then
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/s.zip" "https://www.sqlite.org/$SQLITE_ZIP"
    unzip_to "$tmp/s.zip" "$tmp"
    mkdir -p "$tp/sqlite"
    cp "$tmp"/sqlite-amalgamation-*/sqlite3.[ch] "$tp/sqlite/"
    rm -rf "$tmp"
fi

mkdir -p "$tp/nlohmann"
[ -f "$tp/nlohmann/json.hpp" ] || curl -fsSL -o "$tp/nlohmann/json.hpp" \
    "https://github.com/nlohmann/json/releases/download/$JSON_VER/json.hpp"
mkdir -p "$tp/xxhash"
[ -f "$tp/xxhash/xxhash.h" ] || curl -fsSL -o "$tp/xxhash/xxhash.h" \
    "https://raw.githubusercontent.com/Cyan4973/xxHash/$XXHASH_VER/xxhash.h"
if [ "$target" = windows ]; then
    if [ ! -f "$tp/onnxruntime-win/lib/onnxruntime.dll" ]; then
        tmp="$(mktemp -d)"
        curl -fsSL -o "$tmp/ort.zip" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-win-x64-$ORT_VER.zip"
        unzip_to "$tmp/ort.zip" "$tmp"
        rm -rf "$tp/onnxruntime-win" && mv "$tmp"/onnxruntime-win-x64-* "$tp/onnxruntime-win"
        rm -rf "$tmp"
    fi
fi
if [ "$(uname -s)" = Linux ] && [ ! -f "$tp/onnxruntime/lib/libonnxruntime.so" ]; then
    arch=$(uname -m); case "$arch" in aarch64) ortarch=aarch64 ;; x86_64) ortarch=x64 ;; *) echo "no ONNX Runtime build for $arch"; exit 1 ;; esac
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/onnxruntime-linux-$ortarch-$ORT_VER.tgz"
    tar -xzf "$tmp/ort.tgz" -C "$tmp"
    rm -rf "$tp/onnxruntime" && mv "$tmp"/onnxruntime-linux-* "$tp/onnxruntime"
    rm -rf "$tmp"
fi

mkdir -p "$tp/fontawesome"
for f in webfonts/fa-solid-900.ttf css/all.css LICENSE.txt; do
    [ -f "$tp/fontawesome/$(basename $f)" ] || curl -fsSL -o "$tp/fontawesome/$(basename $f)" "https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@$FA_VER/$f"
done
echo "third_party ready"
