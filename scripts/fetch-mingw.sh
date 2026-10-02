#!/usr/bin/env bash
# Unpacks the mingw-w64 cross compiler (Ubuntu's packages) into third_party/mingw without root, for
#   cmake -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
T="$root/third_party/mingw"
if [ -x "$T/usr/bin/x86_64-w64-mingw32-gcc-posix" ]; then echo "already in $T"; exit 0; fi
tmp="$(mktemp -d)"
(cd "$tmp" && apt-get download gcc-mingw-w64-x86-64-posix g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix-runtime \
   gcc-mingw-w64-base binutils-mingw-w64-x86-64 mingw-w64-x86-64-dev mingw-w64-common >/dev/null)
mkdir -p "$T"; for d in "$tmp"/*.deb; do dpkg-deb -x "$d" "$T"; done; rm -rf "$tmp"
echo "mingw-w64 in $T"
