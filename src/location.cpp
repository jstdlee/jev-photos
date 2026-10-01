#include "location.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <regex>
#include <set>

#include "nlohmann/json.hpp"
#include "util.h"

using json = nlohmann::json;

namespace {

// Generic folder/file words that never name a place (compared lower-case, whole phrase).
const std::set<std::string>& stop_phrases() {
    static const std::set<std::string> s = {
        "img", "image", "images", "photo", "photos", "picture", "pictures", "pics", "pic", "dcim", "camera", "camera roll",
        "cameraroll", "screenshot", "screenshots", "screen shots", "screen", "download", "downloads", "desktop", "documents",
        "backup", "backups", "export", "exports", "exported", "edited", "edit", "edits", "original", "originals", "new folder",
        "untitled", "untitled folder", "misc", "other", "others", "temp", "tmp", "wallpaper", "wallpapers", "whatsapp",
        "whatsapp images", "whatsapp image", "wechat", "weixin", "telegram", "instagram", "facebook", "snapchat", "messenger",
        "line", "qq", "iphone", "ipad", "android", "samsung", "xiaomi", "huawei", "oppo", "vivo", "pixel", "google", "apple",
        "icloud", "icloud photos", "dropbox", "onedrive", "google photos", "drive", "sd card", "sdcard", "card", "raw", "jpg",
        "jpeg", "heic", "png", "video", "videos", "movie", "movies", "mobile", "phone", "gallery", "album", "albums", "media",
        "user", "users", "home", "mnt", "volumes", "data", "share", "shared", "public", "private", "me", "mine", "my photos",
        "my pictures", "all", "all photos", "selected", "best", "favorites", "favourites", "import", "imported", "imports",
        "upload", "uploads", "camera uploads", "snapseed", "lightroom", "photoshop", "vsco", "burst", "portrait", "selfie",
        "selfies", "panorama", "pano", "thumbnails", "thumbs", "cache", "sent", "received", "restored", "recovered", "copy",
        "mmexport", "wx_camera", "microsoft", "windows", "library", "photos library", "masters", "previews", "resources",
        "canon", "nikon", "sony", "fuji", "fujifilm", "olympus", "panasonic", "leica", "gopro", "dji", "ricoh", "pentax",
        "lumix", "eos", "src", "source", "sources", "input", "output", "out", "in", "test", "tests", "new", "old", "files",
        "file", "folder", "folders", "stuff", "archive", "archives", "sort", "sorted", "unsorted", "todo", "done", "work",
        "\xE7\x85\xA7\xE7\x89\x87",                   // 照片
        "\xE7\x9B\xB8\xE7\x89\x87",                   // 相片
        "\xE5\x9B\xBE\xE7\x89\x87",                   // 图片
        "\xE7\x9B\xB8\xE5\x86\x8C",                   // 相册
        "\xE7\x9B\xB8\xE6\x9C\xBA",                   // 相机
        "\xE6\x88\xAA\xE5\x9B\xBE",                   // 截图
        "\xE6\x88\xAA\xE5\xB1\x8F",                   // 截屏
        "\xE5\xB1\x8F\xE5\xB9\x95\xE6\x88\xAA\xE5\x9B\xBE",  // 屏幕截图
        "\xE6\x96\xB0\xE5\xBB\xBA\xE6\x96\x87\xE4\xBB\xB6\xE5\xA4\xB9",  // 新建文件夹
        "\xE4\xB8\x8B\xE8\xBD\xBD",                   // 下载
        "\xE6\xA1\x8C\xE9\x9D\xA2",                   // 桌面
        "\xE5\xA4\x87\xE4\xBB\xBD",                   // 备份
        "\xE5\xAF\xBC\xE5\x87\xBA",                   // 导出
        "\xE5\xBE\xAE\xE4\xBF\xA1",                   // 微信
        "\xE5\xBE\xAE\xE4\xBF\xA1\xE5\x9B\xBE\xE7\x89\x87",  // 微信图片
        "\xE6\x89\x8B\xE6\x9C\xBA",                   // 手机
        "\xE5\x85\xA8\xE9\x83\xA8",                   // 全部
        "\xE5\x85\xB6\xE4\xBB\x96",                   // 其他
        "\xE6\x9C\xAA\xE5\x91\xBD\xE5\x90\x8D",       // 未命名
        "\xE6\x88\x91\xE7\x9A\x84\xE7\x85\xA7\xE7\x89\x87",  // 我的照片
        "\xE5\xA3\x81\xE7\xBA\xB8",                   // 壁纸
        "\xE8\xA7\x86\xE9\xA2\x91",                   // 视频
        "\xE5\x8E\x9F\xE5\x9B\xBE",                   // 原图
        "\xE7\xB2\xBE\xE9\x80\x89",                   // 精选
        "\xE6\x94\xB6\xE8\x97\x8F",                   // 收藏
        "\xE6\x96\x87\xE6\xA1\xA3",                   // 文档
    };
    return s;
}

// Camera/app file-name prefixes, matched as the leading token.
bool camera_token(const std::string& lw) {
    static const std::regex re(
        "^(img|dsc|dscn|dscf|dsci|pxl|mvimg|vid|pano|burst|sam|cimg|imag|kimg|dji|gopr|gp|gh|p|pict|dcp|mvc|wa|mmexport|"
        "wx_camera|screenshot|photo|image|mms|received|snapchat|fb_img|signal|trim|edit|copy|[0-9]{3}[a-z_]{5})$",
        std::regex::ECMAScript | std::regex::optimize);
    return std::regex_match(lw, re);
}

// Trip words around a place ("Paris trip", "巴黎旅行", "日本之旅").
const std::vector<std::string>& travel_affixes() {
    static const std::vector<std::string> v = {
        " day trip", " road trip", " trip", " travels", " travel", " vacation", " holidays", " holiday", " tour",
        " visit", " journey", " getaway", " weekend",
        "\xE8\x87\xAA\xE9\xA9\xBE\xE6\xB8\xB8",  // 自驾游
        "\xE4\xB8\x80\xE6\x97\xA5\xE6\xB8\xB8",  // 一日游
        "\xE4\xB9\x8B\xE6\x97\x85",              // 之旅
        "\xE6\x97\x85\xE8\xA1\x8C",              // 旅行
        "\xE6\x97\x85\xE6\xB8\xB8",              // 旅游
        "\xE6\xB8\xB8\xE7\x8E\xA9",              // 游玩
        "\xE6\xB8\xB8\xE8\xAE\xB0",              // 游记
        "\xE8\x87\xAA\xE9\xA9\xBE",              // 自驾
        "\xE5\x87\xBA\xE5\xB7\xAE",              // 出差
        "\xE8\xA1\x8C",                          // 行 (日本行)
        "\xE6\xB8\xB8",                          // 游 (西湖游)
    };
    return v;
}

std::string strip_travel(const std::string& s, bool& cue) {
    std::string l = util::lower(s);
    for (auto& a : travel_affixes()) {
        bool cjk = a[0] != ' ';
        if (!util::ends_with(l, a) || s.size() <= a.size()) continue;
        std::string rest = util::trim(s.substr(0, s.size() - a.size()));
        // Single-character CJK suffixes need at least two characters left ("日本行" yes, "行" no).
        if (cjk && util::utf8_len(a) == 1 && util::utf8_len(rest) < 2) continue;
        if (rest.empty()) continue;
        cue = true;
        return rest;
    }
    // Leading English trip words: "Trip to Paris", "Visit Kyoto"
    static const std::regex lead("^(trip to|travel to|visit to|visit|vacation in|holiday in|in) (.+)$", std::regex::icase);
    std::smatch m;
    if (std::regex_match(s, m, lead)) {
        cue = true;
        return m[2].str();
    }
    return s;
}

// Remove date-like runs, bracketed counters and separators; returns phrases.
std::vector<std::string> phrases_of(const std::string& name) {
    static const std::regex dateish(
        "((19|20)[0-9]{2}([-_. ]?[0-9]{1,2}){0,2}(\xE5\xB9\xB4|\xE6\x9C\x88|\xE6\x97\xA5)?)|([0-9]+)|"
        "(\xE5\xB9\xB4|\xE6\x9C\x88|\xE6\x97\xA5)",
        std::regex::ECMAScript);
    std::string s = std::regex_replace(name, dateish, "|");
    // Separators become phrase breaks; single spaces stay inside a phrase ("New York").
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        char ch = s[i];
        if (strchr("_-/\\,;()[]{}+&#@!~=|.", ch)) out += '|';
        else out += ch;
    }
    // Full-width separators: 、，（）【】·
    for (const char* sep : {"\xE3\x80\x81", "\xEF\xBC\x8C", "\xEF\xBC\x88", "\xEF\xBC\x89", "\xE3\x80\x90", "\xE3\x80\x91", "\xC2\xB7"})
        out = util::replace_all(out, sep, "|");
    std::vector<std::string> res;
    for (auto& p : util::split(out, '|')) {
        std::string t = util::trim(p);
        while (t.find("  ") != std::string::npos) t = util::replace_all(t, "  ", " ");
        if (!t.empty()) res.push_back(t);
    }
    return res;
}

bool usable(const std::string& t) {
    std::string lw = util::lower(t);
    if (stop_phrases().count(lw) || camera_token(lw)) return false;
    size_t n = util::utf8_len(t);
    if (util::has_cjk(t)) return n >= 2 && n <= 16;
    if (n < 3 || n > 40) return false;
    int letters = 0;
    for (unsigned char c : t) letters += isalpha(c) || c >= 0x80;
    return letters >= 3;
}

}  // namespace

std::vector<PlaceCandidate> place_candidates(const std::vector<std::string>& dirs, const std::string& file_stem) {
    std::vector<PlaceCandidate> out;
    static const std::regex dcf("^[0-9]{3}[A-Za-z0-9_]{5}$");  // DCIM/100CANON, 101MSDCF, 100APPLE
    auto add = [&](const std::string& origin, int level) {
        if (std::regex_match(origin, dcf)) return;
        for (auto& ph : phrases_of(origin)) {
            if (stop_phrases().count(util::lower(ph))) continue;  // "Camera Roll" as a whole, before word filtering
            // Drop leading camera tokens inside a phrase ("IMG Paris" -> "Paris").
            std::vector<std::string> words;
            // Word level only drops camera/photo noise; generic words like "new" belong to names ("New York").
            static const std::set<std::string> noise = {"img", "image", "images", "photo", "photos", "picture", "pictures", "pics",
                                                        "dcim", "camera", "screenshot", "screenshots", "copy", "edited", "export"};
            for (auto& w : util::split(ph, ' '))
                if (!w.empty() && !camera_token(util::lower(w)) && !noise.count(util::lower(w))) words.push_back(w);
            std::string phrase;
            for (auto& w : words) phrase += (phrase.empty() ? "" : " ") + w;
            bool cue = false;
            std::string t = strip_travel(phrase, cue);
            if (!usable(t)) continue;
            auto dup = std::find_if(out.begin(), out.end(), [&](const PlaceCandidate& c) { return util::lower(c.text) == util::lower(t); });
            if (dup != out.end()) { dup->travel_cue |= cue; continue; }
            PlaceCandidate c;
            c.text = t;
            c.origin = origin;
            c.level = level;
            c.travel_cue = cue;
            out.push_back(c);
        }
    };
    for (size_t i = 0; i < dirs.size() && i < 3; i++) add(dirs[i], int(i));
    add(file_stem, -1);
    for (auto& c : out) {
        double lvl = c.level == 0 ? 0.35 : c.level == 1 ? 0.30 : c.level == 2 ? 0.25 : 0.20;
        c.prior = std::min(0.95, lvl + (c.travel_cue ? 0.35 : 0.0));
    }
    if (out.size() > 19) out.resize(19);  // jev accepts 20 options including "none"
    return out;
}

PlaceVote rules_vote(const std::vector<PlaceCandidate>& cands, double weight) {
    PlaceVote v;
    v.voter = "rules";
    v.weight = weight;
    // The best-scored candidate carries its prior; the others share what is left in proportion.
    double best = 0;
    for (auto& c : cands) best = std::max(best, c.prior);
    for (auto& c : cands) v.scores.push_back(c.prior >= best ? c.prior : c.prior * (1 - best) * 0.5);
    v.note = "folder-level and trip-word priors";
    return v;
}

int match_candidate(const std::vector<PlaceCandidate>& cands, const std::string& answer_in) {
    std::string a = util::lower(util::trim(answer_in));
    if (a.empty() || a == "none" || a == "null") return -1;
    for (size_t i = 0; i < cands.size(); i++)
        if (util::lower(cands[i].text) == a) return int(i);
    for (size_t i = 0; i < cands.size(); i++) {
        std::string t = util::lower(cands[i].text), o = util::lower(cands[i].origin);
        if (t.find(a) != std::string::npos || a.find(t) != std::string::npos || o.find(a) != std::string::npos) return int(i);
    }
    return -1;
}

PlaceDecision decide_place(const std::vector<PlaceCandidate>& cands, const std::vector<PlaceVote>& votes, double accept) {
    PlaceDecision d;
    json ev;
    ev["candidates"] = json::array();
    for (auto& c : cands)
        ev["candidates"].push_back({{"text", c.text}, {"origin", c.origin}, {"level", c.level}, {"travel_cue", c.travel_cue},
                                    {"prior", std::round(c.prior * 1000) / 1000}});
    double wsum = 0;
    std::vector<double> total(cands.size(), 0.0);
    ev["votes"] = json::array();
    for (auto& v : votes) {
        if (v.weight <= 0 || v.scores.size() != cands.size()) continue;
        wsum += v.weight;
        for (size_t i = 0; i < cands.size(); i++) total[i] += v.weight * v.scores[i];
        json s = json::array();
        for (double x : v.scores) s.push_back(std::round(x * 1000) / 1000);
        ev["votes"].push_back({{"voter", v.voter}, {"weight", v.weight}, {"scores", s}, {"note", v.note}});
    }
    // jev (Julia) cannot say "none" reliably, so it only breaks ties: a candidate must first be backed by the rules
    // (a trip cue) or the VL model.
    std::vector<bool> eligible(cands.size(), false);
    for (auto& v : votes)
        if (v.voter != "jev" && v.weight > 0 && v.scores.size() == cands.size())
            for (size_t i = 0; i < cands.size(); i++) eligible[i] = eligible[i] || v.scores[i] >= 0.5;
    int best = -1;
    if (wsum > 0)
        for (size_t i = 0; i < cands.size(); i++) {
            total[i] /= wsum;
            if (eligible[i] && (best < 0 || total[i] > total[best])) best = int(i);
        }
    ev["accept"] = accept;
    if (best >= 0 && total[best] >= accept) {
        d.text = cands[best].text;
        d.confidence = total[best];
        d.source = cands[best].level < 0 ? "file-name" : "folder:" + cands[best].origin;
    }
    ev["result"] = d.text;
    ev["confidence"] = std::round((best >= 0 ? total[best] : 0) * 1000) / 1000;
    d.evidence_json = ev.dump(-1, ' ', false, json::error_handler_t::replace);
    return d;
}
