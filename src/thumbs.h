// Thumbnail cache: one upright JPEG per photo version, made once and reused by vision tagging, the perceptual
// hash and the UI. Photos get a small side (512 px); screenshots and other text-heavy images keep a larger one
// (1024 px) so the model can still read them.
#pragma once

#include <cstdint>
#include <string>

struct ThumbKey {
    std::string src_path;
    int64_t size = 0, mtime = 0;
};

// Cache location: $XDG_CACHE_HOME/jev-photos/thumbs/ab/<hash>-<side>.jpg (hash of path, size and mtime, so an edited
// file gets a new thumbnail and identical copies never share a wrong one).
std::string thumb_path(const ThumbKey& k, int side);
// Side for this file: text_side for screenshots / PNG / WebP / GIF, photo_side otherwise.
int thumb_side_for(const std::string& src_path, int photo_side, int text_side);
// Makes the thumbnail if it is missing. Returns its path, or "" when the image cannot be decoded (err set).
std::string ensure_thumb(const ThumbKey& k, int orientation, int side, std::string& err);
std::string thumbs_dir();
