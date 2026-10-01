#include "genmeta.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <regex>
#include <set>

#include "nlohmann/json.hpp"
#include "stb_image.h"
#include "util.h"

using json = nlohmann::json;

std::string GenInfo::to_json() const {
    json j = {{"tool", tool}, {"field", field}, {"prompt", prompt}, {"negative", negative}, {"model", model}, {"sampler", sampler},
              {"scheduler", scheduler}, {"seed", seed}, {"size", size}, {"steps", steps}, {"cfg", cfg}, {"sources", sources}};
    json l = json::array();
    for (auto& [n, s] : loras) l.push_back({n, s});
    j["loras"] = l;
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

GenInfo GenInfo::from_json(const std::string& s) {
    GenInfo g;
    json j = json::parse(s.empty() ? "{}" : s, nullptr, false);
    if (!j.is_object()) return g;
    auto str = [&](const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
    g.tool = str("tool"); g.field = str("field"); g.prompt = str("prompt"); g.negative = str("negative"); g.model = str("model");
    g.sampler = str("sampler"); g.scheduler = str("scheduler"); g.seed = str("seed"); g.size = str("size");
    g.steps = j.value("steps", 0);
    g.cfg = j.value("cfg", 0.0);
    if (j.contains("sources") && j["sources"].is_array())
        for (auto& x : j["sources"]) if (x.is_string()) g.sources.push_back(x.get<std::string>());
    if (j.contains("loras") && j["loras"].is_array())
        for (auto& x : j["loras"])
            if (x.is_array() && x.size() == 2 && x[0].is_string()) g.loras.push_back({x[0].get<std::string>(), x[1].is_number() ? x[1].get<double>() : 1.0});
    return g;
}

// ---------------------------------------------------------------------------
// PNG text chunks

std::map<std::string, std::string> png_text_chunks(const std::string& path) {
    std::map<std::string, std::string> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    unsigned char sig[8];
    if (fread(sig, 1, 8, f) != 8 || memcmp(sig, "\x89PNG\r\n\x1a\n", 8) != 0) { fclose(f); return out; }
    for (int guard = 0; guard < 100000; guard++) {
        unsigned char h[8];
        if (fread(h, 1, 8, f) != 8) break;
        uint32_t n = (uint32_t(h[0]) << 24) | (uint32_t(h[1]) << 16) | (uint32_t(h[2]) << 8) | h[3];
        std::string type(reinterpret_cast<char*>(h + 4), 4);
        if (type == "IEND") break;
        bool text = type == "tEXt" || type == "zTXt" || type == "iTXt";
        if (!text || n > (64u << 20)) {  // skip image data (and absurd chunks) without reading them
            if (fseek(f, long(n) + 4, SEEK_CUR) != 0) break;
            continue;
        }
        std::string body(n, '\0');
        if (n && fread(&body[0], 1, n, f) != n) break;
        fseek(f, 4, SEEK_CUR);  // CRC
        size_t z = body.find('\0');
        if (z == std::string::npos) continue;
        std::string key = body.substr(0, z), val;
        auto inflate = [](const std::string& data) {
            int len = 0;
            char* p = stbi_zlib_decode_malloc(data.data(), int(data.size()), &len);
            std::string s = p ? std::string(p, size_t(len)) : std::string();
            free(p);
            return s;
        };
        if (type == "tEXt") {
            val = body.substr(z + 1);
            // tEXt is Latin-1 by the spec, but every image tool writes UTF-8 there; keep the bytes.
        } else if (type == "zTXt") {
            if (z + 2 <= body.size()) val = inflate(body.substr(z + 2));
        } else {  // iTXt: keyword\0 flag method lang\0 translated\0 text
            if (z + 3 > body.size()) continue;
            bool compressed = body[z + 1] != 0;
            size_t lang_end = body.find('\0', z + 3);
            size_t tr_end = lang_end == std::string::npos ? std::string::npos : body.find('\0', lang_end + 1);
            if (tr_end == std::string::npos) continue;
            std::string data = body.substr(tr_end + 1);
            val = compressed ? inflate(data) : data;
        }
        if (!key.empty() && !val.empty()) out[key] = val;
    }
    fclose(f);
    return out;
}

// ---------------------------------------------------------------------------
// A1111 / Forge "parameters"

GenInfo parse_a1111(const std::string& text_in) {
    GenInfo g;
    std::string text = util::trim(text_in);
    if (text.empty()) return g;
    g.tool = "A1111";
    std::vector<std::string> lines = util::split(text, '\n');
    int params = -1;
    for (int i = int(lines.size()) - 1; i >= 0; i--)
        if (util::starts_with(util::trim(lines[size_t(i)]), "Steps: ")) { params = i; break; }
    std::string pos, neg;
    bool in_neg = false;
    for (int i = 0; i < (params < 0 ? int(lines.size()) : params); i++) {
        std::string l = lines[size_t(i)];
        if (util::starts_with(l, "Negative prompt:")) { in_neg = true; l = l.substr(16); }
        std::string& dst = in_neg ? neg : pos;
        dst += (dst.empty() ? "" : "\n") + l;
    }
    g.prompt = util::trim(pos);
    g.negative = util::trim(neg);
    if (params >= 0) {
        // "Key: value, Key: "quoted, value", ..." (values may be quoted and contain commas)
        std::string p;
        for (size_t i = size_t(params); i < lines.size(); i++) p += (p.empty() ? "" : ", ") + lines[i];
        std::map<std::string, std::string> kv;
        size_t i = 0;
        while (i < p.size()) {
            while (i < p.size() && (p[i] == ' ' || p[i] == ',')) i++;
            size_t colon = p.find(':', i);
            if (colon == std::string::npos) break;
            std::string k = util::trim(p.substr(i, colon - i));
            i = colon + 1;
            while (i < p.size() && p[i] == ' ') i++;
            std::string v;
            if (i < p.size() && p[i] == '"') {
                size_t e = i + 1;
                while (e < p.size() && !(p[e] == '"' && p[e - 1] != '\\')) e++;
                v = p.substr(i + 1, e - i - 1);
                i = e + 1;
            } else {
                size_t e = p.find(", ", i);
                v = p.substr(i, e == std::string::npos ? std::string::npos : e - i);
                i = e == std::string::npos ? p.size() : e + 2;
            }
            kv[k] = util::trim(v);
        }
        g.steps = atoi(kv["Steps"].c_str());
        g.sampler = kv["Sampler"];
        g.scheduler = kv["Schedule type"];
        g.cfg = atof(kv["CFG scale"].c_str());
        g.seed = kv["Seed"];
        g.size = kv["Size"];
        g.model = kv.count("Model") ? kv["Model"] : kv["Model hash"];
        for (auto& part : util::split(kv["Lora hashes"], ','))
            if (auto name = util::trim(part.substr(0, part.find(':'))); !name.empty()) g.loras.push_back({name, 1.0});
    }
    static const std::regex lora(R"(<lora:([^:>]+)(?::([0-9.]+))?[^>]*>)");
    std::set<std::string> have;
    for (auto& l : g.loras) have.insert(l.first);
    for (std::sregex_iterator it(g.prompt.begin(), g.prompt.end(), lora), end; it != end; ++it)
        if (have.insert((*it)[1].str()).second) g.loras.push_back({(*it)[1].str(), (*it)[2].matched ? atof((*it)[2].str().c_str()) : 1.0});
    return g;
}

// ---------------------------------------------------------------------------
// ComfyUI API graph: follow the sampler's positive / negative inputs back to the text that produced them.

GenInfo parse_comfy_prompt(const std::string& js) {
    GenInfo g;
    json G = json::parse(js, nullptr, false);
    if (!G.is_object() || G.empty()) return g;
    auto node = [&](const json& id) -> const json* {
        std::string k = id.is_string() ? id.get<std::string>() : id.is_number() ? std::to_string(id.get<long long>()) : "";
        return G.contains(k) && G[k].is_object() ? &G[k] : nullptr;
    };
    auto is_link = [](const json& v) { return v.is_array() && v.size() == 2 && (v[0].is_string() || v[0].is_number()) && v[1].is_number_integer(); };
    auto cls = [](const json* n) { return n && n->contains("class_type") && (*n)["class_type"].is_string() ? (*n)["class_type"].get<std::string>() : std::string(); };
    auto inputs = [](const json* n) -> const json* { return n && n->contains("inputs") && (*n)["inputs"].is_object() ? &(*n)["inputs"] : nullptr; };
    auto lower = [](std::string s) { return util::lower(s); };
    auto textish = [&](const std::string& key) {
        std::string k = lower(key);
        for (const char* w : {"text", "prompt", "string", "value", "positive", "negative", "t5xxl", "clip_l", "clip_g", "caption", "wildcard"})
            if (k.find(w) != std::string::npos) return true;
        return false;
    };
    auto negish = [&](const std::string& key) { std::string k = lower(key); return k.find("negative") != std::string::npos || k == "uc"; };

    // A literal (number or string) behind an input, following primitive/seed nodes.
    std::function<json(const json&, int)> literal = [&](const json& v, int depth) -> json {
        if (!is_link(v)) return v;
        if (depth > 6) return json();
        const json* in = inputs(node(v[0]));
        if (!in) return json();
        for (const char* k : {"seed", "noise_seed", "value", "int", "float", "number", "string"})
            if (in->contains(k)) return literal((*in)[k], depth + 1);
        return json();
    };
    // The text behind a conditioning/string link. role: 0 positive, 1 negative.
    std::set<std::string> seen;
    std::function<std::string(const json&, int, int)> text_of = [&](const json& link, int role, int depth) -> std::string {
        if (depth > 14 || !is_link(link)) return "";
        const json* n = node(link[0]);
        const json* in = inputs(n);
        std::string c = cls(n);
        if (!in || c.find("ZeroOut") != std::string::npos) return "";
        std::string key = link[0].dump() + "/" + std::to_string(role);
        if (!seen.insert(key).second) return "";
        bool has_neg = false, has_pos = false;
        for (auto& [k, v] : in->items())
            if (textish(k)) (negish(k) ? has_neg : has_pos) = true;
        std::vector<std::string> parts;
        for (auto& [k, v] : in->items()) {
            if (!textish(k)) continue;
            if (has_neg && has_pos && negish(k) != (role == 1)) continue;  // a node holding both: pick by role
            if (v.is_string() && !util::trim(v.get<std::string>()).empty()) parts.push_back(util::trim(v.get<std::string>()));
            else if (is_link(v)) {
                std::string t = text_of(v, negish(k) ? 1 : role, depth + 1);
                if (!t.empty()) parts.push_back(t);
            }
        }
        if (parts.empty())  // pass-through nodes (ControlNet apply, conditioning combine/concat, guiders)
            for (auto& [k, v] : in->items()) {
                std::string lk = lower(k);
                bool cond = lk.find("conditioning") != std::string::npos || lk == "positive" || lk == "negative" || lk == "guider";
                if (!cond || !is_link(v)) continue;
                if ((lk == "positive" && role == 1) || (lk == "negative" && role == 0)) continue;
                std::string t = text_of(v, role, depth + 1);
                if (!t.empty()) parts.push_back(t);
            }
        std::string out;
        for (auto& p : parts)
            if (out.find(p) == std::string::npos) out += (out.empty() ? "" : "\n") + p;
        return out;
    };

    // Sampler: the node that takes positive/negative (or a guider).
    std::vector<std::string> ids;
    for (auto& [k, v] : G.items()) ids.push_back(k);
    std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) { return atoi(a.c_str()) < atoi(b.c_str()); });
    const json* sampler = nullptr;
    for (auto& id : ids) {
        const json* in = inputs(&G[id]);
        std::string c = cls(&G[id]);
        if (in && (in->contains("positive") || in->contains("guider")) && (c.find("Sampler") != std::string::npos || c.find("Guider") != std::string::npos)) {
            sampler = &G[id];
            if (c.find("Sampler") != std::string::npos) break;
        }
    }
    if (sampler) {
        const json& in = *inputs(sampler);
        if (in.contains("positive")) g.prompt = text_of(in["positive"], 0, 0);
        if (in.contains("negative")) g.negative = text_of(in["negative"], 1, 0);
        if (g.prompt.empty() && in.contains("guider")) {
            const json* gin = inputs(node(in["guider"][0]));
            if (gin && gin->contains("positive")) { g.prompt = text_of((*gin)["positive"], 0, 0); if (gin->contains("negative")) g.negative = text_of((*gin)["negative"], 1, 0); }
            else if (gin && gin->contains("conditioning")) g.prompt = text_of((*gin)["conditioning"], 0, 0);
        }
        auto lit = [&](const char* k) { return in.contains(k) ? literal(in[k], 0) : json(); };
        json seed = lit("seed");
        if (seed.is_null()) seed = lit("noise_seed");
        if (seed.is_null() && in.contains("noise")) {  // SamplerCustomAdvanced: RandomNoise node
            const json* nin = inputs(node(in["noise"][0]));
            if (nin && nin->contains("noise_seed")) seed = literal((*nin)["noise_seed"], 0);
        }
        if (seed.is_number()) g.seed = std::to_string(seed.get<long long>());
        if (json s = lit("steps"); s.is_number()) g.steps = s.get<int>();
        if (json s = lit("cfg"); s.is_number()) g.cfg = s.get<double>();
        if (json s = lit("sampler_name"); s.is_string()) g.sampler = s.get<std::string>();
        if (json s = lit("scheduler"); s.is_string()) g.scheduler = s.get<std::string>();
    }
    // Without a recognisable sampler: every text encoder's text.
    if (g.prompt.empty())
        for (auto& id : ids) {
            std::string c = cls(&G[id]);
            const json* in = inputs(&G[id]);
            if (!in || c.find("TextEncode") == std::string::npos) continue;
            for (auto& [k, v] : in->items())
                if (textish(k) && !negish(k) && v.is_string() && !v.get<std::string>().empty())
                    g.prompt += (g.prompt.empty() ? "" : "\n") + v.get<std::string>();
        }
    // Model, LoRAs, size, input images.
    std::string unet, ckpt;
    for (auto& id : ids) {
        const json* in = inputs(&G[id]);
        std::string c = cls(&G[id]);
        if (!in) continue;
        auto s = [&](const char* k) { return in->contains(k) && (*in)[k].is_string() ? (*in)[k].get<std::string>() : std::string(); };
        if (ckpt.empty()) ckpt = s("ckpt_name");
        if (unet.empty()) unet = s("unet_name");
        if (unet.empty() && c.find("Loader") != std::string::npos && c.find("CLIP") == std::string::npos && c.find("VAE") == std::string::npos &&
            c.find("Lora") == std::string::npos && c.find("LoRA") == std::string::npos)
            unet = s("model_name");
        if (std::string l = s("lora_name"); !l.empty()) {
            json st = in->contains("strength_model") ? literal((*in)["strength_model"], 0) : in->contains("strength") ? literal((*in)["strength"], 0) : json(1.0);
            g.loras.push_back({util::basename(l), st.is_number() ? st.get<double>() : 1.0});
        }
        if (c.find("LatentImage") != std::string::npos && g.size.empty()) {
            json w = in->contains("width") ? literal((*in)["width"], 0) : json(), h = in->contains("height") ? literal((*in)["height"], 0) : json();
            if (w.is_number() && h.is_number()) g.size = std::to_string(w.get<int>()) + "x" + std::to_string(h.get<int>());
        }
        if (c == "LoadImage" || c == "LoadImageMask")
            if (std::string im = s("image"); !im.empty()) g.sources.push_back(im);
    }
    g.model = util::basename(!ckpt.empty() ? ckpt : unet);
    if (!g.prompt.empty() || !g.model.empty()) g.tool = "ComfyUI";
    return g;
}

// ---------------------------------------------------------------------------

static GenInfo parse_json_params(const std::string& text) {
    GenInfo g;
    json j = json::parse(text, nullptr, false);
    if (!j.is_object()) return g;
    auto str = [](const json& o, const char* k) {
        if (!o.contains(k)) return std::string();
        const json& v = o[k];
        return v.is_string() ? v.get<std::string>() : v.is_number() ? v.dump() : std::string();
    };
    if (j.contains("sui_image_params") && j["sui_image_params"].is_object()) {  // SwarmUI
        const json& p = j["sui_image_params"];
        g.tool = "SwarmUI";
        g.prompt = str(p, "prompt"); g.negative = str(p, "negativeprompt"); g.model = str(p, "model"); g.seed = str(p, "seed");
        g.sampler = str(p, "sampler"); g.scheduler = str(p, "scheduler");
        g.steps = atoi(str(p, "steps").c_str()); g.cfg = atof(str(p, "cfgscale").c_str());
        if (p.contains("width")) g.size = str(p, "width") + "x" + str(p, "height");
        return g;
    }
    if (j.contains("prompt") && j["prompt"].is_string()) {  // Fooocus and friends
        g.tool = j.contains("base_model") || j.contains("fooocus_scheme") || j.contains("styles") ? "Fooocus" : "JSON";
        g.prompt = str(j, "prompt"); g.negative = str(j, "negative_prompt"); g.model = str(j, "base_model");
        if (g.model.empty()) g.model = str(j, "model");
        g.seed = str(j, "seed"); g.sampler = str(j, "sampler"); g.scheduler = str(j, "scheduler");
        g.steps = atoi(str(j, "steps").c_str());
        g.cfg = atof(str(j, "guidance_scale").c_str());
        g.size = str(j, "resolution");
    }
    return g;
}

static GenInfo parse_invoke(const std::string& text) {
    GenInfo g;
    json j = json::parse(text, nullptr, false);
    if (!j.is_object()) return g;
    g.tool = "InvokeAI";
    auto str = [&](const char* k) { return j.contains(k) ? (j[k].is_string() ? j[k].get<std::string>() : j[k].is_number() ? j[k].dump() : "") : ""; };
    g.prompt = str("positive_prompt");
    g.negative = str("negative_prompt");
    if (j.contains("model") && j["model"].is_object()) g.model = j["model"].value("name", "");
    g.seed = str("seed"); g.scheduler = str("scheduler");
    g.steps = j.value("steps", 0);
    g.cfg = j.contains("cfg_scale") && j["cfg_scale"].is_number() ? j["cfg_scale"].get<double>() : 0;
    if (j.contains("width") && j.contains("height")) g.size = str("width") + "x" + str("height");
    return g;
}

GenInfo read_gen_info(const std::string& path, const MetaMap& meta) {
    GenInfo g;
    std::map<std::string, std::string> t = util::ext_lower(path) == "png" ? png_text_chunks(path) : std::map<std::string, std::string>{};
    auto field = [&](GenInfo x, const std::string& where) {
        if (x.found()) x.field = where;
        return x;
    };
    if (t.count("prompt")) g = field(parse_comfy_prompt(t["prompt"]), "PNG text: prompt (ComfyUI graph)");
    if (!g.found() && t.count("parameters")) {
        const std::string& p = t["parameters"];
        g = util::trim(p).rfind("{", 0) == 0 ? parse_json_params(p) : parse_a1111(p);
        g = field(g, "PNG text: parameters");
    }
    if (!g.found() && t.count("invokeai_metadata")) g = field(parse_invoke(t["invokeai_metadata"]), "PNG text: invokeai_metadata");
    if (!g.found() && t.count("Comment") && (t.count("Software") && t["Software"].find("NovelAI") != std::string::npos)) {
        json c = json::parse(t["Comment"], nullptr, false);
        if (c.is_object()) {
            g.tool = "NovelAI";
            g.field = "PNG text: Comment";
            g.prompt = c.value("prompt", t.count("Description") ? t["Description"] : "");
            g.negative = c.value("uc", "");
            g.steps = c.value("steps", 0);
            g.cfg = c.contains("scale") && c["scale"].is_number() ? c["scale"].get<double>() : 0;
            g.seed = c.contains("seed") ? c["seed"].dump() : "";
            g.sampler = c.value("sampler", "");
            g.model = t.count("Source") ? t["Source"] : "";
        }
    }
    if (g.found()) return g;
    // JPEG / WebP: EXIF text written by the tools (and kept by Civitai downloads).
    auto strip_charset = [](std::string s) {
        if (util::starts_with(s, "charset=")) {
            size_t sp = s.find(' ');
            s = sp == std::string::npos ? "" : s.substr(sp + 1);
        }
        return util::trim(s);
    };
    std::string make = meta.get("Exif.Image.Make"), model = meta.get("Exif.Image.Model");
    if (util::starts_with(model, "prompt:")) g = field(parse_comfy_prompt(model.substr(7)), "EXIF Model: prompt (ComfyUI)");
    else if (util::starts_with(make, "prompt:")) g = field(parse_comfy_prompt(make.substr(7)), "EXIF Make: prompt (ComfyUI)");
    if (g.found()) return g;
    for (const char* k : {"Exif.Photo.UserComment", "Exif.Image.ImageDescription", "Xmp.exif.UserComment"}) {
        std::string v = strip_charset(meta.get(k));
        if (v.empty()) continue;
        if (v.find("\nSteps: ") != std::string::npos || v.find("Negative prompt:") != std::string::npos || util::starts_with(v, "Steps: "))
            return field(parse_a1111(v), k);
        if (v[0] == '{') {
            GenInfo j = parse_json_params(v);
            if (j.found()) return field(j, k);
        }
    }
    // Midjourney: the prompt is the description, followed by "Job ID: ...".
    for (const char* k : {"Xmp.dc.description", "Exif.Image.ImageDescription", "Xmp.iptc.Description"}) {
        std::string v = meta.get(k);
        if (util::starts_with(v, "lang=")) v = v.substr(v.find(' ') == std::string::npos ? v.size() : v.find(' ') + 1);
        size_t job = v.find("Job ID:");
        if (job != std::string::npos) {
            g.tool = "Midjourney";
            g.field = k;
            g.prompt = util::trim(v.substr(0, job));
            return g;
        }
    }
    return g;
}

std::string clean_prompt(const std::string& p) {
    static const std::regex angle(R"(<[^>]*>)"), weight(R"(:\s*-?[0-9.]+\s*\))"), embed(R"(\bembedding:\S+)"), brk(R"(\bBREAK\b)"),
        brackets(R"([()\[\]{}])"), spaces(R"([ \t]+)"), commas(R"((\s*,\s*)+)"), nl(R"(\s*\n\s*)");
    std::string s = util::replace_all(util::replace_all(p, "\\(", "("), "\\)", ")");
    s = std::regex_replace(s, angle, " ");
    s = std::regex_replace(s, embed, " ");
    s = std::regex_replace(s, brk, ",");
    s = std::regex_replace(s, weight, ")");
    s = std::regex_replace(s, brackets, "");
    s = std::regex_replace(s, spaces, " ");
    s = std::regex_replace(s, nl, "\n");
    s = std::regex_replace(s, commas, ", ");
    s = util::trim(s);
    while (!s.empty() && (s.back() == ',' || s.back() == ' ')) s.pop_back();
    while (!s.empty() && (s[0] == ',' || s[0] == ' ')) s.erase(0, 1);
    return s;
}

std::vector<std::string> prompt_keywords(const std::string& prompt, size_t max_n) {
    std::vector<std::string> segs;
    std::string s = clean_prompt(prompt);
    for (const char* sep : {"\xEF\xBC\x8C", "\xE3\x80\x81", "\xEF\xBC\x9B", ";"}) s = util::replace_all(s, sep, ",");  // ，、；;
    for (auto& line : util::split(s, '\n'))
        for (auto& part : util::split(line, ',')) {
            std::string t = util::lower(util::trim(util::replace_all(part, "_", " ")));
            if (!t.empty()) segs.push_back(t);
        }
    if (segs.size() < 3) return {};
    size_t shortn = 0;
    for (auto& t : segs) shortn += std::count(t.begin(), t.end(), ' ') <= 3;
    if (shortn * 10 < segs.size() * 8) return {};  // prose (or prose mixed with tags): left to the LLM
    static const std::set<std::string> boiler = {"masterpiece", "best quality", "high quality", "highres", "absurdres", "ultra detailed",
                                                 "highly detailed", "8k", "4k", "uhd", "hdr", "amazing quality", "very aesthetic",
                                                 "newest", "score 9", "score 8 up", "score 7 up", "score 6 up", "score 5 up", "score 4 up",
                                                 "worst quality", "low quality", "detailed", "intricate details", "sharp focus", "realistic",
                                                 "photorealistic", "raw photo", "official art", "general", "sensitive", "safe"};
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (auto& t : segs) {
        if (boiler.count(t) || t.size() > 40 || std::count(t.begin(), t.end(), ' ') > 3 || util::starts_with(t, "score ") ||
            util::starts_with(t, "rating:"))
            continue;
        if (seen.insert(t).second) out.push_back(t);
        if (out.size() >= max_n) break;
    }
    return out;
}
