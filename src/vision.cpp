#include "vision.h"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>

#include <map>
#include <mutex>

#include "dupes.h"
#include "http.h"
#include "nlohmann/json.hpp"
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct Pixels {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;
};

bool load_stb(const std::string& path, Pixels& px) {
    int w, h, n;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!d) return false;
    px.w = w;
    px.h = h;
    px.rgb.assign(d, d + size_t(w) * h * 3);
    stbi_image_free(d);
    return true;
}

// Largest embedded preview (RAW, HEIC on exiv2 builds with BMFF, TIFF...).
bool load_preview(const std::string& path, Pixels& px) {
    util::ProcResult list = util::run({"exiv2", "-q", "-pp", "--", path}, "", 30);
    int best = 0;
    long best_bytes = 0;
    for (auto& line : util::split(list.out, '\n')) {
        int n = 0;
        long bytes = 0;
        const char* comma = strrchr(line.c_str(), ',');
        if (sscanf(line.c_str(), "Preview %d:", &n) == 1 && comma && sscanf(comma + 1, "%ld", &bytes) == 1 && bytes > best_bytes) {
            best = n;
            best_bytes = bytes;
        }
    }
    if (!best) return false;
    std::string dir = util::temp_path("preview");
    util::mkdirs(dir);
    util::run({"exiv2", "-q", "-ep" + std::to_string(best), "-l", dir, "--", path}, "", 30);
    bool ok = false;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(fs::u8path(dir), ec))
        if (!ok) ok = load_stb(e.path().u8string(), px);
    fs::remove_all(fs::u8path(dir), ec);
    return ok;
}

bool load_ffmpeg(const std::string& path, int max_side, Pixels& px) {
    if (!util::which("ffmpeg")) return false;
    std::string out = util::temp_path("frame.png");
    std::string scale = util::fmt("scale='min(%d,iw)':'min(%d,ih)':force_original_aspect_ratio=decrease", max_side, max_side);
    util::ProcResult r = util::run({"ffmpeg", "-v", "error", "-y", "-i", path, "-frames:v", "1", "-vf", scale, out}, "", 60);
    bool ok = r.rc == 0 && load_stb(out, px);
    unlink(out.c_str());
    return ok;
}

Pixels orient(const Pixels& s, int o) {
    if (o <= 1 || o > 8) return s;
    Pixels d;
    bool swap = o >= 5;
    d.w = swap ? s.h : s.w;
    d.h = swap ? s.w : s.h;
    d.rgb.resize(size_t(d.w) * d.h * 3);
    for (int y = 0; y < d.h; y++)
        for (int x = 0; x < d.w; x++) {
            int sx = x, sy = y;
            switch (o) {
                case 2: sx = s.w - 1 - x; sy = y; break;
                case 3: sx = s.w - 1 - x; sy = s.h - 1 - y; break;
                case 4: sx = x; sy = s.h - 1 - y; break;
                case 5: sx = y; sy = x; break;
                case 6: sx = y; sy = s.h - 1 - x; break;
                case 7: sx = s.w - 1 - y; sy = s.h - 1 - x; break;
                case 8: sx = s.w - 1 - y; sy = x; break;
            }
            memcpy(&d.rgb[(size_t(y) * d.w + x) * 3], &s.rgb[(size_t(sy) * s.w + sx) * 3], 3);
        }
    return d;
}

void jpeg_sink(void* ctx, void* data, int size) { static_cast<std::string*>(ctx)->append(static_cast<char*>(data), size_t(size)); }

std::string str_of(const json& j, const char* k) {
    if (!j.contains(k)) return "";
    const json& v = j[k];
    if (v.is_string()) return util::trim(v.get<std::string>());
    if (v.is_number()) return v.dump();
    return "";
}

std::vector<std::string> list_of(const json& j, const char* k, size_t max_n) {
    std::vector<std::string> out;
    if (!j.contains(k)) return out;
    const json& v = j[k];
    auto push = [&](std::string s) {
        s = util::trim(s);
        if (s.empty() || out.size() >= max_n) return;
        for (auto& e : out)
            if (util::lower(e) == util::lower(s)) return;
        out.push_back(s);
    };
    if (v.is_array()) {
        for (auto& x : v)
            if (x.is_string()) push(x.get<std::string>());
    } else if (v.is_string()) {
        for (auto& p : util::split(v.get<std::string>(), ',')) push(p);
    }
    return out;
}

// Server root for an OpenAI-style base URL: "http://h:11434/v1" -> "http://h:11434".
std::string server_root(const std::string& base) {
    std::string b = base;
    while (!b.empty() && b.back() == '/') b.pop_back();
    if (util::ends_with(b, "/v1")) b.resize(b.size() - 3);
    return b;
}

// Ollama's OpenAI endpoint cannot set the context size, so a request there makes Ollama (re)load the model with
// its default context, which may not fit in memory. When the server is Ollama, use its native /api/chat instead.
bool is_ollama(const Config& c) {
    static std::mutex mu;
    static std::map<std::string, bool> cache;
    std::lock_guard<std::mutex> l(mu);
    auto it = cache.find(c.vl_url);
    if (it != cache.end()) return it->second;
    HttpResponse h = http_get(server_root(c.vl_url) + "/api/version", c.vl_key, 5);
    bool yes = h.ok() && h.body.find("\"version\"") != std::string::npos;
    if (h.status || yes) cache[c.vl_url] = yes;  // do not cache a connection failure
    return yes;
}

// One user turn (text + optional JPEG) -> the model's text. Empty on error (err set).
std::string ask(const Config& c, const std::string& prompt, const std::string* jpeg, int max_tokens, std::string& err) {
    std::string b64 = jpeg ? util::base64(reinterpret_cast<const unsigned char*>(jpeg->data()), jpeg->size()) : "";
    if (is_ollama(c)) {
        json msg = {{"role", "user"}, {"content", prompt}};
        if (jpeg) msg["images"] = json::array({b64});
        json req = {{"model", c.vl_model},
                    {"messages", json::array({msg})},
                    {"stream", false},
                    {"keep_alive", c.vl_keep_alive},
                    {"options", {{"temperature", 0.1}, {"num_ctx", c.vl_num_ctx}, {"num_predict", max_tokens}}}};
        if (c.vl_device == 1) req["options"]["num_gpu"] = 0;  // CPU only (e.g. while the GPU memory is taken)
        if (c.vl_json_mode) req["format"] = "json";
        // On the CPU a photo can take minutes on a busy machine; do not give up at the GPU-sized timeout.
        int timeout = c.vl_device == 1 ? std::max(c.vl_timeout, 600) : c.vl_timeout;
        HttpResponse h = http_post_json(server_root(c.vl_url) + "/api/chat", req.dump(), c.vl_key, timeout);
        if (!h.ok()) {
            err = h.error + (h.body.empty() ? "" : ": " + h.body.substr(0, 300));
            return "";
        }
        json j = json::parse(h.body, nullptr, false);
        if (j.is_object() && j.contains("message") && j["message"].contains("content") && j["message"]["content"].is_string())
            return j["message"]["content"].get<std::string>();
        err = "unexpected response: " + h.body.substr(0, 300);
        return "";
    }
    json content = prompt;
    if (jpeg)
        content = json::array({{{"type", "text"}, {"text", prompt}},
                               {{"type", "image_url"}, {"image_url", {{"url", "data:image/jpeg;base64," + b64}}}}});
    json req = {{"model", c.vl_model},
                {"temperature", 0.1},
                {"max_tokens", max_tokens},
                {"messages", json::array({{{"role", "user"}, {"content", content}}})}};
    if (c.vl_json_mode) req["response_format"] = {{"type", "json_object"}};
    HttpResponse h = http_post_json(url_join(c.vl_url, "/chat/completions"), req.dump(), c.vl_key, c.vl_timeout);
    if (!h.ok()) {
        err = h.error + (h.body.empty() ? "" : ": " + h.body.substr(0, 300));
        return "";
    }
    json j = json::parse(h.body, nullptr, false);
    if (!j.is_object() || !j.contains("choices") || j["choices"].empty()) {
        err = "unexpected response: " + h.body.substr(0, 300);
        return "";
    }
    const json& m = j["choices"][0]["message"];
    if (m.contains("content") && m["content"].is_string()) return m["content"].get<std::string>();
    err = "empty message";
    return "";
}

}  // namespace

std::string extract_json_object(const std::string& s_in) {
    std::string s = s_in;
    for (size_t a; (a = s.find("<think>")) != std::string::npos;) {
        size_t b = s.find("</think>", a);
        s.erase(a, b == std::string::npos ? std::string::npos : b + 8 - a);
    }
    size_t a = s.find('{');
    if (a == std::string::npos) return "";
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = a; i < s.size(); i++) {
        char ch = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') in_str = true;
        else if (ch == '{') depth++;
        else if (ch == '}' && --depth == 0) return s.substr(a, i - a + 1);
    }
    return "";
}

bool prepare_image(const std::string& path, int orientation, int max_side, std::string& jpeg, std::string& err) {
    Pixels px;
    if (!load_stb(path, px) && !load_preview(path, px) && !load_ffmpeg(path, max_side, px)) {
        err = "cannot decode image";
        return false;
    }
    if (std::max(px.w, px.h) > max_side) {
        double s = double(max_side) / std::max(px.w, px.h);
        Pixels r;
        r.w = std::max(1, int(px.w * s + 0.5));
        r.h = std::max(1, int(px.h * s + 0.5));
        r.rgb.resize(size_t(r.w) * r.h * 3);
        stbir_resize_uint8_srgb(px.rgb.data(), px.w, px.h, 0, r.rgb.data(), r.w, r.h, 0, STBIR_RGB);
        px = std::move(r);
    }
    px = orient(px, orientation);
    jpeg.clear();
    if (!stbi_write_jpg_to_func(jpeg_sink, &jpeg, px.w, px.h, 3, px.rgb.data(), 85)) {
        err = "jpeg encode failed";
        return false;
    }
    return true;
}

bool image_dhash(const std::string& path, int orientation, uint64_t& hash) {
    Pixels px;
    if (!load_stb(path, px) && !load_preview(path, px) && !load_ffmpeg(path, 256, px)) return false;
    // Shrink first (cheap to orient), then orient so rotated copies hash alike, then 9x8 grey.
    Pixels small;
    double s = std::min(1.0, 128.0 / std::max(px.w, px.h));
    small.w = std::max(9, int(px.w * s + 0.5));
    small.h = std::max(8, int(px.h * s + 0.5));
    small.rgb.resize(size_t(small.w) * small.h * 3);
    stbir_resize_uint8_linear(px.rgb.data(), px.w, px.h, 0, small.rgb.data(), small.w, small.h, 0, STBIR_RGB);
    small = orient(small, orientation);
    unsigned char tiny[9 * 8 * 3], grey[9 * 8];
    stbir_resize_uint8_linear(small.rgb.data(), small.w, small.h, 0, tiny, 9, 8, 0, STBIR_RGB);
    for (int i = 0; i < 72; i++) grey[i] = (unsigned char)((tiny[i * 3] * 299 + tiny[i * 3 + 1] * 587 + tiny[i * 3 + 2] * 114) / 1000);
    hash = dhash_from_gray9x8(grey);
    return true;
}

VisionResult vision_describe(const Config& c, const std::string& jpeg) {
    VisionResult r;
    std::string prompt =
        "You are cataloguing a personal photo library. Look at the photo and reply with JSON only, no prose:\n"
        "{\"caption\": \"one factual sentence\", \"scene\": \"short scene type, e.g. beach, city street, living room, "
        "restaurant, mountain trail, office, party\", \"objects\": [\"up to 10 important objects\"], "
        "\"tags\": [\"up to 10 search keywords: activities, season, weather, mood, event\"], \"people\": <number of people>, "
        "\"landmark\": \"name of a famous landmark if clearly recognisable, else empty\", "
        "\"text\": \"short visible text if any, else empty\"}\n"
        "Write caption, scene, objects and tags in " + c.vl_tag_lang + ". Do not guess names of people.";
    r.raw = ask(c, prompt, &jpeg, 500, r.error);
    if (r.raw.empty()) return r;
    json j = json::parse(extract_json_object(r.raw), nullptr, false);
    if (!j.is_object()) {
        r.error = "model did not return JSON";
        return r;
    }
    r.caption = str_of(j, "caption");
    r.scene = str_of(j, "scene");
    r.landmark = str_of(j, "landmark");
    r.text = str_of(j, "text");
    r.objects = list_of(j, "objects", 10);
    r.tags = list_of(j, "tags", 10);
    if (j.contains("people") && j["people"].is_number()) r.people = std::max(0, j["people"].get<int>());
    for (auto* s : {&r.landmark, &r.text})
        if (util::lower(*s) == "none" || util::lower(*s) == "empty" || util::lower(*s) == "n/a") s->clear();
    r.ok = !r.caption.empty() || !r.objects.empty() || !r.tags.empty();
    if (!r.ok) r.error = "empty description";
    return r;
}

VlPlace vl_place_from_names(const Config& c, const std::vector<std::string>& parts) {
    VlPlace r;
    std::string joined;
    for (auto& p : parts) joined += (joined.empty() ? "" : " ; ") + p;
    std::string prompt = "Photo folder/file name parts: " + joined +
                         "\nWhich part names a geographic place (city, country, region, park, lake, mountain, landmark) where "
                         "the photos were taken? Copy it exactly. People, events, devices, apps and generic words are not places.\n"
                         "Reply JSON only: {\"place\": \"<exact part or empty string>\", \"confidence\": <0..1>}";
    std::string text = ask(c, prompt, nullptr, 80, r.error);
    if (text.empty()) return r;
    json j = json::parse(extract_json_object(text), nullptr, false);
    if (!j.is_object()) {
        r.error = "model did not return JSON: " + text.substr(0, 200);
        return r;
    }
    r.place = str_of(j, "place");
    r.confidence = j.contains("confidence") && j["confidence"].is_number() ? std::clamp(j["confidence"].get<double>(), 0.0, 1.0) : 0.7;
    r.ok = true;
    return r;
}

bool& vl_loaded_flag() {
    static bool v = false;
    return v;
}

bool vl_health(const Config& c, std::string& detail) {
    HttpResponse h = http_get(url_join(c.vl_url, "/models"), c.vl_key, 5);
    if (!h.ok()) {
        detail = h.error;
        return false;
    }
    json j = json::parse(h.body, nullptr, false);
    bool found = false;
    if (j.contains("data"))
        for (auto& m : j["data"]) found |= m.value("id", "") == c.vl_model;
    detail = found ? "ok" : "model not listed by server";
    if (found && is_ollama(c)) {  // tell whether it is already loaded (first photo is slow otherwise)
        HttpResponse ps = http_get(server_root(c.vl_url) + "/api/ps", c.vl_key, 5);
        bool loaded = ps.ok() && ps.body.find("\"" + c.vl_model + "\"") != std::string::npos;
        vl_loaded_flag() = loaded;  // "ok" either way; the model loads on first use
    }
    return true;
}

bool vl_preload(const Config& c, std::string& err) {
    if (!is_ollama(c)) return true;  // other servers load on their own terms
    json req = {{"model", c.vl_model}, {"prompt", ""}, {"keep_alive", c.vl_keep_alive}, {"options", {{"num_ctx", c.vl_num_ctx}}}};
    if (c.vl_device == 1) req["options"]["num_gpu"] = 0;
    HttpResponse h = http_post_json(server_root(c.vl_url) + "/api/generate", req.dump(), c.vl_key, 300);
    if (!h.ok()) {
        err = h.error + (h.body.empty() ? "" : ": " + h.body.substr(0, 300));
        return false;
    }
    return true;
}
