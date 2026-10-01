#include "thumbs.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "util.h"
#include "vision.h"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

std::string thumbs_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    return (xdg && *xdg ? std::string(xdg) : util::home() + "/.cache") + "/jev-photos/thumbs";
}

std::string thumb_path(const ThumbKey& k, int side) {
    std::string id = k.src_path + "\n" + std::to_string(k.size) + "\n" + std::to_string(k.mtime);
    std::string h = util::fmt("%016llx", (unsigned long long)XXH3_64bits(id.data(), id.size()));
    return thumbs_dir() + "/" + h.substr(0, 2) + "/" + h + "-" + std::to_string(side) + ".jpg";
}

int thumb_side_for(const std::string& src_path, int photo_side, int text_side) {
    std::string e = util::ext_lower(src_path), n = util::lower(util::basename(src_path));
    bool text = e == "png" || e == "webp" || e == "gif" || n.find("screenshot") != std::string::npos ||
                n.find("screen shot") != std::string::npos || n.find("\xE6\x88\xAA\xE5\x9B\xBE") != std::string::npos ||  // 截图
                n.find("\xE6\x88\xAA\xE5\xB1\x8F") != std::string::npos;                                                 // 截屏
    return text ? text_side : photo_side;
}

std::string ensure_thumb(const ThumbKey& k, int orientation, int side, std::string& err) {
    std::string p = thumb_path(k, side);
    if (util::file_exists(p)) return p;
    std::string jpeg;
    if (!prepare_image(k.src_path, orientation, side, jpeg, err)) return "";
    util::mkdirs(util::dirname(p));
    std::string part = util::temp_path("thumb.jpg");
    if (!util::write_file(part, jpeg) || rename(part.c_str(), p.c_str()) != 0) {
        // Different filesystem for TMPDIR: write next to the target instead.
        unlink(part.c_str());
        part = p + ".part";
        if (!util::write_file(part, jpeg) || rename(part.c_str(), p.c_str()) != 0) {
            unlink(part.c_str());
            err = "cannot write thumbnail " + p;
            return "";
        }
    }
    return p;
}
