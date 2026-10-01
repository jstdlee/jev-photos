#!/usr/bin/env bash
# Install the jev photos launcher for this user: icon, application-menu entry and a desktop shortcut.
#   scripts/install-desktop.sh            (after ./build.sh)
#   scripts/install-desktop.sh --remove
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
data="${XDG_DATA_HOME:-$HOME/.local/share}"
apps="$data/applications"
icons="$data/icons/hicolor"
desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
entry="jev-photos.desktop"

if [ "${1:-}" = "--remove" ]; then
    rm -f "$apps/$entry" "$desktop_dir/$entry" "$icons/scalable/apps/jev-photos.svg" "$icons/256x256/apps/jev-photos.png"
    echo "removed"
    exit 0
fi
[ -x "$root/build/jev-photos" ] || { echo "build first: ./build.sh"; exit 1; }

mkdir -p "$apps" "$icons/scalable/apps" "$icons/256x256/apps"
cp "$root/assets/jev-photos.svg" "$icons/scalable/apps/jev-photos.svg"
cp "$root/assets/jev-photos-256.png" "$icons/256x256/apps/jev-photos.png"
sed -e "s|^Exec=.*|Exec=$root/build/jev-photos %f|" -e "s|^Path=.*|Path=$root|" "$root/jev-photos.desktop" > "$apps/$entry"
chmod +x "$apps/$entry"
if [ -d "$desktop_dir" ]; then
    cp "$apps/$entry" "$desktop_dir/$entry"
    chmod +x "$desktop_dir/$entry"
    # GNOME only starts desktop launchers marked as trusted.
    gio set "$desktop_dir/$entry" metadata::trusted true 2>/dev/null || true
fi
command -v update-desktop-database >/dev/null && update-desktop-database -q "$apps" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$icons" 2>/dev/null || true
echo "installed: $apps/$entry"
[ -d "$desktop_dir" ] && echo "desktop shortcut: $desktop_dir/$entry"
