#!/usr/bin/env bash
# Fetch the CLIP models used for tagging and semantic search (ONNX, from Hugging Face). Full-precision weights:
# the int8 exports lose too much (screenshots of people came out as "app screen", portraits as "tea").
#   scripts/fetch-models.sh        ViT-B/32 (~600 MB)   fast (~60 ms per photo on the CPU)
#   scripts/fetch-models.sh l14    ViT-L/14 (~1.7 GB)   much better (~0.3 s per photo); used automatically once present
#   scripts/fetch-models.sh b32 q8 the old int8 files (~150 MB)
set -euo pipefail
dir="${XDG_DATA_HOME:-$HOME/.local/share}/jev-photos/models"
case "${1:-b32}" in
    b32) repo=Xenova/clip-vit-base-patch32;  name=clip-vit-base-patch32 ;;
    l14) repo=Xenova/clip-vit-large-patch14; name=clip-vit-large-patch14 ;;
    *) echo "usage: $0 [b32|l14] [q8]"; exit 2 ;;
esac
sfx=""
[ "${2:-}" = "q8" ] && sfx="_quantized"
mkdir -p "$dir/$name"
for f in "onnx/vision_model$sfx.onnx" "onnx/text_model$sfx.onnx" vocab.json merges.txt; do
    out="$dir/$name/$(basename "$f")"
    [ -s "$out" ] && continue
    echo "fetching $repo/$f"
    curl -fL --retry 3 -o "$out.part" "https://huggingface.co/$repo/resolve/main/$f" && mv "$out.part" "$out"
done
echo "models ready in $dir/$name"
