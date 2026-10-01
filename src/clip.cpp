#include "clip.h"

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>

#include "nlohmann/json.hpp"
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "util.h"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

using json = nlohmann::json;

float dot(const Vec& a, const Vec& b) {
    float s = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void normalise(Vec& v) {
    double s = 0;
    for (float x : v) s += double(x) * x;
    s = std::sqrt(s);
    if (s > 0)
        for (float& x : v) x = float(x / s);
}

// ---------------------------------------------------------------------------
// Tokenizer (byte-level BPE, as in openai/CLIP's simple_tokenizer.py)

static std::string utf8_of(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += char(cp);
    else if (cp < 0x800) { s += char(0xC0 | (cp >> 6)); s += char(0x80 | (cp & 0x3F)); }
    else { s += char(0xE0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    return s;
}

bool ClipTokenizer::load(const std::string& dir, std::string& err) {
    // bytes_to_unicode(): printable bytes map to themselves, the rest to 256+n.
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        byte_enc_[b] = utf8_of(keep ? uint32_t(b) : uint32_t(256 + n++));
    }
    std::string s;
    if (!util::read_file(dir + "/vocab.json", s)) { err = "missing vocab.json"; return false; }
    json v = json::parse(s, nullptr, false);
    if (!v.is_object()) { err = "bad vocab.json"; return false; }
    for (auto& [k, id] : v.items()) vocab_[k] = id.get<int64_t>();
    std::ifstream m(dir + "/merges.txt");
    if (!m) { err = "missing merges.txt"; return false; }
    std::string line;
    int rank = 0;
    while (std::getline(m, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
    }
    return !vocab_.empty() && !ranks_.empty();
}

// Split a UTF-8 string into characters.
static std::vector<std::string> chars_of(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

std::vector<std::string> ClipTokenizer::bpe(const std::string& word) const {
    auto hit = cache_.find(word);
    if (hit != cache_.end()) return hit->second;
    std::vector<std::string> parts;
    for (unsigned char b : word) parts.push_back(byte_enc_[b]);
    if (parts.empty()) return parts;
    parts.back() += "</w>";
    for (;;) {
        int best = -1, best_rank = 1 << 30;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = ranks_.find({parts[i], parts[i + 1]});
            if (it != ranks_.end() && it->second < best_rank) { best_rank = it->second; best = int(i); }
        }
        if (best < 0) break;
        // Merge every occurrence of this pair, left to right.
        std::string a = parts[size_t(best)], b = parts[size_t(best) + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < parts.size();) {
            if (i + 1 < parts.size() && parts[i] == a && parts[i + 1] == b) { merged.push_back(a + b); i += 2; }
            else merged.push_back(parts[i++]);
        }
        parts.swap(merged);
    }
    cache_[word] = parts;
    return parts;
}

std::vector<int64_t> ClipTokenizer::encode(const std::string& text_in, size_t max_len) const {
    // Lower-case ASCII, collapse whitespace; then the CLIP pre-tokenizer pattern: contractions, letter runs,
    // single digits, runs of other symbols. Non-ASCII characters count as letters (as \p{L} does for CJK etc.).
    std::string text;
    bool space = true;
    for (char ch : text_in) {
        char c = (ch >= 'A' && ch <= 'Z') ? char(ch - 'A' + 'a') : ch;
        if (isspace((unsigned char)c)) { if (!space) text += ' '; space = true; }
        else { text += c; space = false; }
    }
    std::vector<std::string> words;
    std::vector<std::string> cs = chars_of(util::trim(text));
    auto kind = [](const std::string& c) {
        unsigned char b = c[0];
        if (b >= 0x80 || isalpha(b)) return 1;  // letter
        if (isdigit(b)) return 2;
        if (b == ' ') return 0;
        return 3;  // symbol
    };
    for (size_t i = 0; i < cs.size();) {
        int k = kind(cs[i]);
        if (k == 0) { i++; continue; }
        if (cs[i] == "'" && i + 1 < cs.size()) {  // 's 't 're 've 'm 'll 'd
            for (const char* suf : {"s", "t", "re", "ve", "m", "ll", "d"}) {
                std::string sfx = suf;
                std::string cand;
                for (size_t j = i + 1; j < cs.size() && cand.size() < sfx.size(); j++) cand += cs[j];
                if (cand == sfx) { words.push_back("'" + sfx); i += 1 + sfx.size(); goto next; }
            }
        }
        if (k == 2) { words.push_back(cs[i++]); continue; }
        {
            std::string w;
            while (i < cs.size() && kind(cs[i]) == k) w += cs[i++];
            words.push_back(w);
        }
    next:;
    }
    std::vector<int64_t> ids = {49406};  // <|startoftext|>
    for (auto& w : words)
        for (auto& t : bpe(w)) {
            auto it = vocab_.find(t);
            if (it != vocab_.end()) ids.push_back(it->second);
            if (ids.size() >= max_len - 1) break;
        }
    ids.push_back(49407);  // <|endoftext|>
    return ids;
}

// ---------------------------------------------------------------------------
// Models

std::string models_dir() {
    const char* xdg = getenv("XDG_DATA_HOME");
    return (xdg && *xdg ? std::string(xdg) : util::home() + "/.local/share") + "/jev-photos/models";
}

// Full precision when downloaded (the int8 exports lose a lot of accuracy), else the quantized files.
static bool pick_files(const std::string& dir, std::string& vision, std::string& text) {
    if (!util::file_exists(dir + "/vocab.json") || !util::file_exists(dir + "/merges.txt")) return false;
    for (const char* sfx : {"", "_quantized"}) {
        std::string v = std::string("vision_model") + sfx + ".onnx", t = std::string("text_model") + sfx + ".onnx";
        if (util::file_exists(dir + "/" + v) && util::file_exists(dir + "/" + t)) {
            vision = v;
            text = t;
            return true;
        }
    }
    return false;
}

bool clip_files_present(const std::string& dir) {
    std::string v, t;
    return pick_files(dir, v, t);
}

static const OrtApi* ort() {
    static const OrtApi* a = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return a;
}

static OrtEnv* ort_env() {
    static OrtEnv* env = [] {
        OrtEnv* e = nullptr;
        if (OrtStatus* st = ort()->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "jev-photos", &e)) ort()->ReleaseStatus(st);
        return e;
    }();
    return env;
}

// Whether this ONNX Runtime build can run CUDA (the stock CPU tarball cannot).
static bool cuda_available() {
    static int cached = -1;
    static std::mutex mu;
    std::lock_guard<std::mutex> l(mu);
    if (cached >= 0) return cached == 1;
    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = ort()->CreateSessionOptions(&so)) { ort()->ReleaseStatus(st); cached = 0; return false; }
    OrtCUDAProviderOptionsV2* co = nullptr;
    bool ok = false;
    if (OrtStatus* st = ort()->CreateCUDAProviderOptions(&co)) ort()->ReleaseStatus(st);
    else {
        OrtStatus* st2 = ort()->SessionOptionsAppendExecutionProvider_CUDA_V2(so, co);
        ok = st2 == nullptr;
        if (st2) ort()->ReleaseStatus(st2);
        ort()->ReleaseCUDAProviderOptions(co);
    }
    ort()->ReleaseSessionOptions(so);
    cached = ok ? 1 : 0;
    return ok;
}

static ClipInfo model_info(const char* name, const char* label) {
    ClipInfo i{name, label, models_dir() + "/" + name, "", ""};
    // The id names the exact weights: embeddings from different files are not comparable.
    if (pick_files(i.dir, i.vision_file, i.text_file) && i.vision_file.find("quantized") == std::string::npos) i.id += "-fp32";
    else i.label += " int8";
    return i;
}

ClipInfo clip_choose(const Config& c) {
    ClipInfo b32 = model_info("clip-vit-base-patch32", "ViT-B/32");
    ClipInfo l14 = model_info("clip-vit-large-patch14", "ViT-L/14");
    if (c.clip_model == "l14") return l14;
    if (c.clip_model == "b32") return b32;
    // auto: the large model when its files are there (full precision it is ~0.2 s per photo on this CPU, and much
    // better at people, art and documents), else the base model.
    if (clip_files_present(l14.dir) && (l14.vision_file.find("quantized") == std::string::npos || cuda_available())) return l14;
    return b32;
}

bool clip_status(const Config& c, std::string& detail) {
    ClipInfo info = clip_choose(c);
    if (!clip_files_present(info.dir)) {
        detail = info.label + ": model files missing (run scripts/fetch-models.sh" + (info.id.find("large") != std::string::npos ? " l14" : "") + ")";
        return false;
    }
    bool gpu = c.clip_device != 1 && cuda_available();
    detail = info.label + " on " + (gpu ? "GPU" : "CPU") + (c.clip_device == 2 && !gpu ? " (no GPU provider in this ONNX Runtime build)" : "") +
             ", tags from " + tag_vocabulary_path();
    return true;
}

struct ClipModel::Impl {
    OrtSession* vision = nullptr;
    OrtSession* text = nullptr;
    OrtMemoryInfo* mem = nullptr;
    std::mutex mu;  // sessions are thread-safe, but keep batches from interleaving on a busy CPU
};

ClipModel::ClipModel() : impl_(new Impl) {}

ClipModel::~ClipModel() {
    if (impl_->vision) ort()->ReleaseSession(impl_->vision);
    if (impl_->text) ort()->ReleaseSession(impl_->text);
    if (impl_->mem) ort()->ReleaseMemoryInfo(impl_->mem);
}

static std::string status_msg(OrtStatus* st) {
    std::string m = ort()->GetErrorMessage(st);
    ort()->ReleaseStatus(st);
    return m;
}

bool ClipModel::load(const ClipInfo& info, int device, int threads, bool need_vision, bool need_text, std::string& err) {
    info_ = info;
    if (!clip_files_present(info.dir)) {
        err = "model files missing in " + info.dir + " (run scripts/fetch-models.sh)";
        return false;
    }
    if (!ort_env()) { err = "cannot start ONNX Runtime"; return false; }
    if (!tok_.load(info.dir, err)) return false;
    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = ort()->CreateSessionOptions(&so)) { err = status_msg(st); return false; }
    ort()->SetIntraOpNumThreads(so, std::max(1, threads));
    ort()->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    device_used_ = "CPU";
    if (device != 1 && cuda_available()) {
        OrtCUDAProviderOptionsV2* co = nullptr;
        if (!ort()->CreateCUDAProviderOptions(&co)) {
            if (OrtStatus* st = ort()->SessionOptionsAppendExecutionProvider_CUDA_V2(so, co)) ort()->ReleaseStatus(st);
            else device_used_ = "GPU (CUDA)";
            ort()->ReleaseCUDAProviderOptions(co);
        }
    }
    if (device == 2 && device_used_ == "CPU") device_used_ = "CPU (no GPU provider in this ONNX Runtime build)";
    auto open = [&](const char* f, OrtSession** s) {
        if (OrtStatus* st = ort()->CreateSession(ort_env(), (info.dir + "/" + f).c_str(), so, s)) {
            err = std::string(f) + ": " + status_msg(st);
            return false;
        }
        return true;
    };
    bool ok = (!need_vision || open(info.vision_file.c_str(), &impl_->vision)) && (!need_text || open(info.text_file.c_str(), &impl_->text));
    ort()->ReleaseSessionOptions(so);
    if (ok) {
        if (OrtStatus* st = ort()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &impl_->mem)) { err = status_msg(st); ok = false; }
    }
    vision_ = impl_->vision;
    text_ = impl_->text;
    return ok;
}

static bool run_one(OrtSession* s, OrtMemoryInfo* mem, const char* in_name, ONNXTensorElementDataType type, void* data, size_t bytes,
                    const std::vector<int64_t>& shape, const char* out_name, std::vector<Vec>& out, std::string& err) {
    OrtValue* in = nullptr;
    if (OrtStatus* st = ort()->CreateTensorWithDataAsOrtValue(mem, data, bytes, shape.data(), shape.size(), type, &in)) {
        err = status_msg(st);
        return false;
    }
    OrtValue* res = nullptr;
    const char* ins[] = {in_name};
    const char* outs[] = {out_name};
    OrtStatus* st = ort()->Run(s, nullptr, ins, &in, 1, outs, 1, &res);
    ort()->ReleaseValue(in);
    if (st) { err = status_msg(st); return false; }
    OrtTensorTypeAndShapeInfo* ti = nullptr;
    ort()->GetTensorTypeAndShape(res, &ti);
    size_t nd = 0;
    ort()->GetDimensionsCount(ti, &nd);
    std::vector<int64_t> dims(nd);
    ort()->GetDimensions(ti, dims.data(), nd);
    ort()->ReleaseTensorTypeAndShapeInfo(ti);
    float* p = nullptr;
    ort()->GetTensorMutableData(res, reinterpret_cast<void**>(&p));
    if (nd != 2) { ort()->ReleaseValue(res); err = "unexpected output rank"; return false; }
    out.clear();
    for (int64_t i = 0; i < dims[0]; i++) {
        Vec v(p + i * dims[1], p + (i + 1) * dims[1]);
        normalise(v);
        out.push_back(std::move(v));
    }
    ort()->ReleaseValue(res);
    return true;
}

bool ClipModel::encode_images(const std::vector<std::vector<float>>& images, std::vector<Vec>& out, std::string& err) {
    if (!impl_->vision) { err = "vision model not loaded"; return false; }
    if (images.empty()) { out.clear(); return true; }
    std::vector<float> batch;
    batch.reserve(images.size() * 3 * 224 * 224);
    for (auto& im : images) batch.insert(batch.end(), im.begin(), im.end());
    std::lock_guard<std::mutex> l(impl_->mu);
    return run_one(impl_->vision, impl_->mem, "pixel_values", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, batch.data(), batch.size() * sizeof(float),
                   {int64_t(images.size()), 3, 224, 224}, "image_embeds", out, err);
}

bool ClipModel::encode_text(const std::string& text, Vec& out, std::string& err) {
    if (!impl_->text) { err = "text model not loaded"; return false; }
    std::vector<int64_t> ids = tok_.encode(text);
    std::vector<Vec> res;
    std::lock_guard<std::mutex> l(impl_->mu);
    if (!run_one(impl_->text, impl_->mem, "input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ids.data(), ids.size() * sizeof(int64_t),
                 {1, int64_t(ids.size())}, "text_embeds", res, err))
        return false;
    out = res.empty() ? Vec() : res[0];
    return !out.empty();
}

bool ClipModel::encode_texts(const std::vector<std::string>& texts, std::vector<Vec>& out, std::string& err) {
    if (!impl_->text) { err = "text model not loaded"; return false; }
    out.clear();
    // Pad with <|endoftext|> (CLIP's pad token): the model pools at the first end token and attends causally, so
    // padding after it changes nothing.
    for (size_t at = 0; at < texts.size(); at += 64) {
        size_t n = std::min<size_t>(64, texts.size() - at);
        std::vector<std::vector<int64_t>> ids;
        size_t len = 0;
        for (size_t k = 0; k < n; k++) {
            ids.push_back(tok_.encode(texts[at + k]));
            len = std::max(len, ids.back().size());
        }
        std::vector<int64_t> flat;
        for (auto& v : ids) {
            v.resize(len, 49407);
            flat.insert(flat.end(), v.begin(), v.end());
        }
        std::vector<Vec> res;
        std::lock_guard<std::mutex> l(impl_->mu);
        if (!run_one(impl_->text, impl_->mem, "input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, flat.data(), flat.size() * sizeof(int64_t),
                     {int64_t(n), int64_t(len)}, "text_embeds", res, err))
            return false;
        for (auto& v : res) out.push_back(std::move(v));
    }
    return out.size() == texts.size();
}

bool clip_preprocess(const std::string& path, std::vector<float>& chw, std::string& err) {
    int w, h, n;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!px) { err = "cannot decode " + path; return false; }
    // Shorter side -> 224, then centre crop 224x224. Very wide or tall pictures (screenshots, panoramas, phone
    // captures) would lose most of their content to the crop, so those are letterboxed instead.
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    double aspect = double(std::max(w, h)) / std::max(1, std::min(w, h));
    bool pad = aspect > 2.0;  // 16:9 still crops (as CLIP was trained); phone screenshots and panoramas are padded
    double s = pad ? 224.0 / std::max(w, h) : 224.0 / std::min(w, h);
    int rw = std::max(1, int(std::lround(w * s))), rh = std::max(1, int(std::lround(h * s)));
    if (!pad) { rw = std::max(224, rw); rh = std::max(224, rh); }
    std::vector<unsigned char> r(size_t(rw) * rh * 3);
    stbir_resize_uint8_srgb(px, w, h, 0, r.data(), rw, rh, 0, STBIR_RGB);
    stbi_image_free(px);
    chw.assign(3 * 224 * 224, 0.0f);  // 0 = the mean colour after normalisation
    for (int y = 0; y < 224; y++)
        for (int x = 0; x < 224; x++) {
            int sx = pad ? x - (224 - rw) / 2 : x + (rw - 224) / 2, sy = pad ? y - (224 - rh) / 2 : y + (rh - 224) / 2;
            if (sx < 0 || sy < 0 || sx >= rw || sy >= rh) continue;
            for (int c = 0; c < 3; c++)
                chw[size_t(c) * 224 * 224 + size_t(y) * 224 + x] = (r[(size_t(sy) * rw + sx) * 3 + c] / 255.0f - mean[c]) / stdv[c];
        }
    return true;
}

// ---------------------------------------------------------------------------
// Tag vocabulary

// Categories matter: tags compete for probability only inside their category, and a category must first carry a
// real share of the whole vocabulary to count as present (a landscape has no "people" to name). "medium" is special:
// every picture gets exactly one (what kind of picture it is).
static const char* kDefaultTags = R"(# vocabulary-version: 2
# medium
photo
screenshot
scanned document
illustration
anime
cartoon
digital art
3d render
painting
drawing
pencil sketch
meme
black and white photo
old photo
# people
person
woman
man
girl
boy
baby
toddler
child
kids
teenager
elderly person
couple
family
group of people
crowd
friends
selfie
portrait
face close-up
wedding couple
# clothing
dress
swimsuit
bikini
lingerie
suit and tie
uniform
costume
cosplay
traditional clothing
kimono
hat
sunglasses
# sensitive
nudity
partial nudity
revealing clothes
# scene
beach
sea
lake
river
waterfall
mountain
snowy mountain
forest
park
garden
field
desert
countryside
farm
city street
city skyline
night city
old town
village
bridge
harbor
airport
train station
road trip
highway
shopping mall
market
supermarket
restaurant
cafe
bar
kitchen
living room
bedroom
bathroom
office
classroom
library
museum
art gallery
church
temple
castle
palace
stadium
concert
theater
hospital
gym
swimming pool
playground
zoo
aquarium
amusement park
campsite
ski resort
home interior
backyard
rooftop
underwater
sky
clouds
photo studio
outer space
fantasy landscape
# event
birthday party
wedding
graduation
christmas
chinese new year
halloween
festival
fireworks
party
dinner party
picnic
barbecue
camping
hiking
skiing
swimming
surfing
cycling
running
football
basketball
tennis
travel
vacation
business meeting
conference
performance
# time and weather
sunrise
sunset
night
snow
rain
fog
autumn leaves
spring flowers
summer
winter
# nature and animals
dog
cat
bird
horse
fish
flowers
tree
cherry blossom
insect
wild animal
pet
# objects
car
bicycle
motorcycle
boat
airplane
train
bus
phone
laptop
computer screen
camera
book
toy
gift
cake
balloons
musical instrument
guitar
piano
sculpture
furniture
shoes
bag
jewelry
watch
# food
food
dessert
fruit
vegetables
noodles
rice
sushi
pizza
hamburger
coffee
tea
wine
beer
breakfast
# document
website
app screen
chat conversation
text document
receipt
ticket
map
chart
diagram
whiteboard
handwriting
presentation slide
code
book page
qr code
# composition
close-up
aerial view
panorama
blurry photo
)";

std::string tag_vocabulary_path() { return util::config_dir() + "/tags.txt"; }

static std::vector<TagDef> parse_vocabulary(const std::string& s) {
    std::vector<TagDef> out;
    std::string cat = "object";
    std::set<std::string> seen;
    for (auto& line : util::split(s, '\n')) {
        std::string t = util::trim(line);
        if (t.empty()) continue;
        if (t[0] == '#') {
            std::string c = util::trim(t.substr(1));
            if (!c.empty() && c.find(':') == std::string::npos) cat = c == "style" ? "composition" : c;
            continue;
        }
        if (seen.insert(util::lower(t)).second) out.push_back({t, cat});
    }
    return out;
}

std::vector<TagDef> load_tag_vocabulary() {
    std::string s;
    std::string path = tag_vocabulary_path();
    static const std::string header = "# jev-photos tag vocabulary: one tag per line, \"# name\" starts a category.\n";
    if (!util::read_file(path, s) || util::trim(s).empty()) {
        s = kDefaultTags;
        util::write_file(path, header + s);
    } else if (s.find("# vocabulary-version: 2") == std::string::npos) {
        // Upgrade an older file: keep the user's tags (and any they added), add the new built-in ones and categories.
        std::vector<TagDef> mine = parse_vocabulary(s), builtin = parse_vocabulary(kDefaultTags);
        std::set<std::string> have;
        for (auto& t : mine) have.insert(util::lower(t.tag));
        std::map<std::string, std::string> style_moved = {{"illustration", "medium"}, {"cartoon", "medium"}, {"drawing", "medium"}, {"meme", "medium"},
                                                          {"old photo", "medium"}, {"black and white photo", "medium"}, {"painting", "medium"},
                                                          {"screenshot", "medium"}, {"clothes", ""}};
        std::vector<TagDef> merged;
        for (auto& t : builtin) merged.push_back(t);
        std::set<std::string> in;
        for (auto& t : merged) in.insert(util::lower(t.tag));
        for (auto& t : mine)
            if (!in.count(util::lower(t.tag)) && !style_moved.count(util::lower(t.tag))) merged.push_back(t);  // the user's own additions
        std::string out = "# vocabulary-version: 2\n", cat;
        for (auto& t : merged) {
            if (t.category != cat) out += "# " + (cat = t.category) + "\n";
            out += t.tag + "\n";
        }
        util::write_file(path, header + out);
        s = out;
    }
    return parse_vocabulary(s);
}

bool save_tag_vocabulary(const std::vector<TagDef>& tags) {
    // Keep each category together, in the order categories first appear.
    std::vector<std::string> cats;
    for (auto& t : tags)
        if (std::find(cats.begin(), cats.end(), t.category) == cats.end()) cats.push_back(t.category);
    std::string out = "# jev-photos tag vocabulary: one tag per line, \"# name\" starts a category.\n# vocabulary-version: 2\n";
    for (auto& c : cats) {
        out += "# " + c + "\n";
        for (auto& t : tags)
            if (t.category == c && !util::trim(t.tag).empty()) out += util::trim(t.tag) + "\n";
    }
    return util::write_file(tag_vocabulary_path(), out);
}

// Prompt ensembles (as in the CLIP paper): several phrasings per tag, averaged. Single prompts are noisy, and
// "a photo of ..." alone biases everything towards photographs.
static bool plural_like(const std::string& t) {
    static const std::set<std::string> mass = {"food", "rice", "tea", "coffee", "wine", "beer", "sushi", "fruit", "noodles", "breakfast", "snow",
                                               "rain", "fog", "sky", "clouds", "night", "sunset", "sunrise", "summer", "winter", "nudity",
                                               "partial nudity", "revealing clothes", "lingerie", "furniture", "jewelry", "code", "handwriting",
                                               "christmas", "halloween", "chinese new year", "travel", "vacation", "camping", "hiking", "skiing",
                                               "swimming", "surfing", "cycling", "running", "football", "basketball", "tennis", "outer space",
                                               "traditional clothing", "cosplay", "underwater", "countryside", "spring flowers", "autumn leaves"};
    return mass.count(t) || (t.size() > 2 && t.back() == 's' && t[t.size() - 2] != 's');
}

static std::string with_article(const std::string& t) {
    if (plural_like(t)) return t;
    return (std::string("aeiou").find(t[0]) != std::string::npos ? "an " : "a ") + t;
}

std::vector<std::string> tag_prompts(const std::string& tag, const std::string& category) {
    std::string a = with_article(tag);
    if (category == "medium" || category == "composition") return {a, "an image that is " + a, "this is " + a, a + " of something"};
    if (category == "document") return {"a screenshot of " + a, "an image of " + a, a, "a photo of " + a};
    return {"a photo of " + a, "a picture of " + a, "an image showing " + a, "an illustration of " + a, a};
}

// Tags are picked this way (bump when pick_tags changes, so stored tags are recomputed).
static const char* kPickVersion = "pick-3";

static std::string vocab_hash(const std::string& model_id, const std::vector<TagDef>& tags) {
    std::string key = model_id + "\n" + kPickVersion;
    for (auto& t : tags)
        for (auto& p : tag_prompts(t.tag, t.category)) key += "\n" + p;
    char hex[32];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)XXH3_64bits(key.data(), key.size()));
    return hex;
}

std::string tag_vocabulary_hash(const std::string& model_id) { return vocab_hash(model_id, load_tag_vocabulary()); }

bool build_tag_index(ClipModel& m, TagIndex& idx, std::string& err) {
    idx.model_id = m.info().id;
    idx.tags = load_tag_vocabulary();
    idx.vecs.clear();
    // Cache the text embeddings on disk: keyed by model and the exact prompt list.
    idx.vocab_hash = vocab_hash(idx.model_id, idx.tags);
    std::string cache = models_dir() + "/tagcache-" + idx.vocab_hash + ".bin";
    std::string blob;
    if (util::read_file(cache, blob) && blob.size() % sizeof(float) == 0 && !idx.tags.empty()) {
        size_t dim = blob.size() / sizeof(float) / idx.tags.size();
        if (dim * idx.tags.size() * sizeof(float) == blob.size() && dim > 0) {
            const float* f = reinterpret_cast<const float*>(blob.data());
            for (size_t i = 0; i < idx.tags.size(); i++) idx.vecs.emplace_back(f + i * dim, f + (i + 1) * dim);
            return true;
        }
    }
    std::vector<std::string> prompts;
    std::vector<size_t> owner;
    for (size_t i = 0; i < idx.tags.size(); i++)
        for (auto& p : tag_prompts(idx.tags[i].tag, idx.tags[i].category)) { prompts.push_back(p); owner.push_back(i); }
    std::vector<Vec> pv;
    if (!m.encode_texts(prompts, pv, err)) return false;
    idx.vecs.assign(idx.tags.size(), Vec());
    for (size_t k = 0; k < pv.size(); k++) {
        Vec& v = idx.vecs[owner[k]];
        if (v.empty()) v.assign(pv[k].size(), 0.0f);
        for (size_t d = 0; d < v.size(); d++) v[d] += pv[k][d];
    }
    for (auto& v : idx.vecs) normalise(v);
    std::string out;
    for (auto& v : idx.vecs) out.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    util::write_file(cache, out);
    return true;
}

bool encode_query(ClipModel& m, const std::string& text, Vec& out, std::string& err) {
    std::vector<Vec> pv;
    if (!m.encode_texts(tag_prompts(text, "object"), pv, err) || pv.empty()) return false;
    out.assign(pv[0].size(), 0.0f);
    for (auto& v : pv)
        for (size_t d = 0; d < out.size(); d++) out[d] += v[d];
    normalise(out);
    return true;
}

TagScores score_tags(const TagIndex& idx, const Vec& image) {
    TagScores s;
    size_t n = idx.vecs.size();
    s.prob.assign(n, 0.0);
    if (!n || image.empty()) return s;
    // CLIP's own scale: softmax(100 * cosine) over the whole vocabulary.
    std::vector<double> logit(n);
    double mx = -1e9, sum = 0;
    for (size_t i = 0; i < n; i++) mx = std::max(mx, logit[i] = 100.0 * dot(image, idx.vecs[i]));
    for (size_t i = 0; i < n; i++) sum += s.prob[i] = std::exp(logit[i] - mx);
    for (auto& p : s.prob) p /= sum;
    for (size_t i = 0; i < n; i++) s.mass[idx.tags[i].category] += s.prob[i];
    return s;
}

double TagScores::share(const TagIndex& idx, size_t i) const {
    auto it = mass.find(idx.tags[i].category);
    return it == mass.end() || it->second <= 0 ? 0.0 : prob[i] / it->second;
}

// How much of the vocabulary's probability a category must hold to count as fully present. Composition has only a
// few tags that match almost any picture a little, so it must hold much more.
double presence_scale(const std::string& category) { return category == "composition" ? 0.6 : 0.3; }

ImageTags pick_tags(const TagIndex& idx, const Vec& image, int max_tags) {
    ImageTags r;
    TagScores s = score_tags(idx, image);
    size_t n = s.prob.size();
    if (!n) return r;
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return s.prob[a] > s.prob[b]; });
    // Confidence of a tag = its share inside the category x how present the category is. A category holding 30%
    // of the whole vocabulary's probability counts as fully present.
    auto conf = [&](size_t i) {
        double m = s.mass.count(idx.tags[i].category) ? s.mass.at(idx.tags[i].category) : 0;
        return s.share(idx, i) * std::min(1.0, m / presence_scale(idx.tags[i].category));
    };
    std::map<std::string, int> per_cat;
    std::vector<std::pair<std::string, float>> picked;
    for (size_t i : order) {
        const std::string& cat = idx.tags[i].category;
        double c = conf(i), m = s.mass.count(cat) ? s.mass.at(cat) : 0;
        int limit = cat == "medium" || cat == "scene" ? 1 : cat == "people" ? 3 : 2;
        if (per_cat[cat] >= limit) continue;
        if (cat == "medium") {  // one per picture; the share alone decides
            if (s.share(idx, i) < 0.3) continue;
            c = s.share(idx, i);
        } else if (m < 0.08 || s.share(idx, i) < 0.25 || c < 0.12) {
            continue;
        }
        per_cat[cat]++;
        if (cat == "scene") r.scene = idx.tags[i].tag;
        picked.push_back({idx.tags[i].tag, float(c)});
    }
    std::stable_sort(picked.begin(), picked.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (auto& t : picked)
        if (int(r.tags.size()) < std::max(1, max_tags)) r.tags.push_back(t);
    return r;
}

double phrase_confidence(ClipModel& m, const TagIndex& idx, const std::string& phrase_in, const Vec& image, std::string& err) {
    std::string phrase = util::lower(util::trim(phrase_in));
    if (phrase.empty() || image.empty() || idx.vecs.empty()) return 0;
    int same = -1;
    for (size_t k = 0; k < idx.tags.size(); k++)
        if (util::lower(idx.tags[k].tag) == phrase) same = int(k);
    Vec q;
    std::string cat;
    if (same >= 0) {
        q = idx.vecs[size_t(same)];
        cat = idx.tags[size_t(same)].category;
    } else {
        if (!encode_query(m, phrase, q, err)) return 0;
        double best = -2;
        for (size_t k = 0; k < idx.vecs.size(); k++) {
            double c = dot(q, idx.vecs[k]);
            if (c > best) { best = c; cat = idx.tags[k].category; }
        }
    }
    double mx = -1e9;
    std::vector<double> l(idx.vecs.size());
    for (size_t k = 0; k < l.size(); k++) mx = std::max(mx, l[k] = 100.0 * dot(image, idx.vecs[k]));
    double lq = 100.0 * dot(image, q);
    mx = std::max(mx, lq);
    double all = 0, in_cat = 0;
    for (size_t k = 0; k < l.size(); k++) {
        if (int(k) == same) continue;
        double e = std::exp(l[k] - mx);
        all += e;
        if (idx.tags[k].category == cat) in_cat += e;
    }
    double eq = std::exp(lq - mx);
    double share = eq / (in_cat + eq), mass = (in_cat + eq) / (all + eq);
    if (cat == "medium") return std::clamp(share, 0.0, 1.0);
    return std::clamp(share * std::min(1.0, mass / presence_scale(cat)), 0.0, 1.0);
}
