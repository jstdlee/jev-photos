// jev-photos — organize a photo collection by recovered capture date, with jev decisions and VL tagging.
// Dear ImGui + GLFW + OpenGL 3 (same stack as gpu-hud), SQLite catalogue, exiv2 for metadata.
#include "backends/imgui_impl_opengl3_loader.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <sys/stat.h>
#ifndef _WIN32
#include <spawn.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "config.h"
#include "dates.h"
#include "genmeta.h"
#include "db.h"
#include "i18n.h"
#include "imgui.h"
#include "jev.h"
#include "location.h"
#include "meta.h"
#include "nlohmann/json.hpp"
#include "pipeline.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "util.h"
#include "clip.h"
#include "llm.h"
#include "search.h"
#include "thumbs.h"
#include "icon_png.h"
#include "icons.h"
#include "vision.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

static volatile sig_atomic_t g_quit = 0;

// Frame profiler: every frame is split into named sections; a frame whose work (not the idle wait) takes
// more than 100 ms is reported with its breakdown, so a UI stall can be pinned to the code that caused it.
struct FrameProf {
    double last = 0;
    std::vector<std::pair<std::string, double>> parts;
    void begin(double t) { last = t; parts.clear(); }
    void mark(const std::string& name) {
        double t = glfwGetTime();
        parts.push_back({name, t - last});
        last = t;
    }
    double total() const {
        double s = 0;
        for (auto& p : parts) s += p.second;
        return s;
    }
    std::string str() const {
        std::string o;
        for (auto& p : parts)
            if (p.second > 0.002) o += util::fmt("%s %.0fms  ", p.first.c_str(), p.second * 1000);
        return o;
    }
};
static FrameProf g_prof;
#ifndef _WIN32
extern char** environ;
#endif

// ---------------------------------------------------------------------------
// Fonts (as in gpu-hud): a Latin base plus the CJK face for the UI language, so Chinese paths render.
struct FontFace {
    std::string file;
    int index = 0;
};

static FontFace fc_match(const char* pattern) {
    FontFace f;
    util::ProcResult r = util::run({"fc-match", "-f", "%{file}|%{index}", pattern}, "", 5);
    size_t bar = r.out.rfind('|');
    if (bar != std::string::npos) {
        f.file = r.out.substr(0, bar);
        f.index = atoi(r.out.c_str() + bar + 1);
    }
    auto ends = [&](const char* e) { return util::ends_with(util::lower(f.file), e); };
    if (!util::file_exists(f.file) || !(ends(".ttf") || ends(".otf") || ends(".ttc"))) f.file.clear();
    return f;
}

// The Font Awesome icon font, linked into the binary.
#ifdef _WIN32  // PE/COFF: read-only data section, and no .previous
#define JEV_RODATA ".section .rdata,\"dr\"\n"
#define JEV_BACK ".text\n"
#else
#define JEV_RODATA ".section .rodata\n"
#define JEV_BACK ".previous\n"
#endif
__asm__(JEV_RODATA ".balign 16\n.global jev_icons_ttf\njev_icons_ttf:\n.incbin \"" ICON_TTF "\"\n"
        ".global jev_icons_ttf_end\njev_icons_ttf_end:\n" JEV_BACK);
extern "C" const unsigned char jev_icons_ttf[], jev_icons_ttf_end[];

static void build_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    FontFace base;
#ifdef _WIN32
    std::string windir = getenv("WINDIR") ? util::norm_path(getenv("WINDIR")) : "C:/Windows";
    for (const char* f : {"segoeui.ttf", "arial.ttf"})
        if (util::file_exists(windir + "/Fonts/" + f)) { base.file = windir + "/Fonts/" + f; break; }
#else
    for (const char* p : {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                          "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"})
        if (util::file_exists(p)) { base.file = p; break; }
    if (base.file.empty()) base = fc_match("sans-serif:lang=en");
#endif
    // Private-use codepoints belong to the icon font: some text fonts put old ligatures there (DejaVu's U+F001/F002).
    static const ImWchar no_private_use[] = {0xE000, 0xF8FF, 0};
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    cfg.GlyphExcludeRanges = no_private_use;
    bool have = false;
    if (!base.file.empty()) {
        cfg.FontNo = base.index;
        have = io.Fonts->AddFontFromFileTTF(base.file.c_str(), 0.0f, &cfg) != nullptr;
    }
    if (!have) io.Fonts->AddFontDefault();
    // Font Awesome icons (private-use codepoints), merged right after the base font so they win over any private-use
    // glyphs in the CJK fonts; a little smaller than the text, fixed width so labels line up, on the text baseline.
    {
        ImFontConfig ic;
        ic.MergeMode = true;
        ic.FontDataOwnedByAtlas = false;
        ic.ExtraSizeScale = 0.86f;
        ic.GlyphMinAdvanceX = 13.0f;
        ic.GlyphOffset = ImVec2(0, 1);
        io.Fonts->AddFontFromMemoryTTF((void*)jev_icons_ttf, int(jev_icons_ttf_end - jev_icons_ttf), 0.0f, &ic);
    }
    // Always merge a CJK face: file and folder names are often Chinese/Japanese even in an English UI.
    // Chinese, Japanese and Korean all merged (names in any of them show up), the interface language's own face
    // first: the same character is drawn the Japanese way in Japanese, the Chinese way in Chinese.
    std::vector<std::vector<std::string>> by_lang;  // zh, ja, ko
#ifdef _WIN32
    by_lang = {{"msyh.ttc", "simsun.ttc"}, {"YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"}, {"malgun.ttf", "gulim.ttc"}};
#else
    by_lang = {{"sans-serif:lang=zh-cn"}, {"sans-serif:lang=ja"}, {"sans-serif:lang=ko"}};
#endif
    int first = get_lang() == L_JA ? 1 : get_lang() == L_KO ? 2 : 0;
    std::vector<int> order = {first};
    for (int k = 0; k < 3; k++)
        if (k != first) order.push_back(k);
    std::set<std::string> merged{base.file};
    for (int k : order)
        for (auto& cand : by_lang[size_t(k)]) {
            FontFace cjk;
#ifdef _WIN32
            if (util::file_exists(windir + "/Fonts/" + cand)) cjk.file = windir + "/Fonts/" + cand;
#else
            cjk = fc_match(cand.c_str());
#endif
            if (cjk.file.empty()) continue;
            std::string key = cjk.file + "#" + std::to_string(cjk.index);
            if (merged.count(key) || merged.count(cjk.file + (cjk.index ? "" : "#0"))) break;
            merged.insert(key);
            ImFontConfig m;
            m.MergeMode = true;
            m.FontNo = cjk.index;
            m.ExtraSizeScale = 1.1f;  // CJK faces look smaller than Latin ones at the same pixel size
            m.GlyphExcludeRanges = no_private_use;
            m.GlyphOffset = ImVec2(0, 1);
            io.Fonts->AddFontFromFileTTF(cjk.file.c_str(), 0.0f, &m);
            break;  // one face per language
        }

}

// ---------------------------------------------------------------------------
// Photos are drawn smaller than they are: mipmaps (trilinear) and anisotropic filtering keep them smooth instead of
// jagged. glGenerateMipmap is not in ImGui's GL loader, so it is fetched once from GLFW.
typedef void (*PFN_GenerateMipmap)(GLenum);
static PFN_GenerateMipmap gl_generate_mipmap() {
    static PFN_GenerateMipmap f = (PFN_GenerateMipmap)glfwGetProcAddress("glGenerateMipmap");
    return f;
}
static float gl_max_anisotropy() {
    static float v = -1;
    if (v < 0) {
        v = 0;
        if (glfwExtensionSupported("GL_EXT_texture_filter_anisotropic") || glfwExtensionSupported("GL_ARB_texture_filter_anisotropic"))
            if (auto get = (void (*)(GLenum, float*))glfwGetProcAddress("glGetFloatv")) get(0x84FF /* GL_MAX_TEXTURE_MAX_ANISOTROPY */, &v);
    }
    return v;
}

struct Texture {
    GLuint tex = 0;
    int w = 0, h = 0;
    void unload() {
        if (tex) glDeleteTextures(1, &tex);
        tex = 0;
        w = h = 0;
    }
    bool upload(const unsigned char* rgba, int pw, int ph) {
        unload();
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pw, ph, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        if (auto gen = gl_generate_mipmap()) {
            gen(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 0x2703 /* GL_LINEAR_MIPMAP_LINEAR */);
            if (float an = gl_max_anisotropy(); an > 1) glTexParameteri(GL_TEXTURE_2D, 0x84FE /* GL_TEXTURE_MAX_ANISOTROPY */, int(std::min(an, 8.0f)));
        } else {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
        w = pw;
        h = ph;
        return true;
    }
};

struct Thumb {
    int64_t id = 0;
    std::vector<unsigned char> rgba;
    int w = 0, h = 0;
    std::string error;
};

enum Tab { TAB_PHOTOS, TAB_LOG, TAB_SETTINGS };
// What the main area shows (the sidebar's items).
enum Section {
    SEC_ALL, SEC_FAV, SEC_MONTH, SEC_PLACE, SEC_AI, SEC_FOLDER,            // library
    SEC_DUPES, SEC_DATES, SEC_SUGG, SEC_ORGANIZE, SEC_SAVE,                // to review
    SEC_TAGS, SEC_RECOG, SEC_ACTIVITY                                      // tools
};
static bool library_section(int s) { return s <= SEC_FOLDER || s == SEC_DATES; }

// One row of the duplicates table (group header or member).
struct DupLine {
    int group = -1;  // -1 = group header
    int64_t id = 0;
    bool keeper = false, similar = false;
    std::string text;  // header text
};

// Everything the Photos / Duplicates tabs show, built off the UI thread.
struct ViewRequest {
    std::string month, folder;
    bool review_only = false, dup_similar = false;
    SearchQuery sq;
    int gen = 0;     // catalogue generation: the search index reloads when it changes
    Config cfg;      // CLIP settings for the query encoder
    bool tags = false;  // the Tags & EXIF tab is open: build its rows too
    bool fav_only = false, ai_only = false;
    std::string place;  // sidebar Places: only this place
};
struct ViewData {
    std::vector<PhotoRow> rows;
    bool searching = false;
    std::string search_note, search_error;
    std::vector<MonthInfo> months;
    DbStats stats;
    std::vector<DupRow> dup_rows;
    std::vector<DupLine> dup_lines;
    std::string dup_summary;
    std::vector<TagRow> tag_rows;
    bool tags_built = false;
    std::vector<Decision> decisions;  // newest first
    DecisionStats decision_stats;
    std::vector<PhotoRow> favorites;
    std::vector<Correction> corrections;  // pending
    std::map<std::string, int> places;
    int ai_count = 0;
};

static ViewData build_view(Db& db, Searcher& se, const ViewRequest& r) {
    ViewData v;
    if (r.tags) {
        ClipInfo info = clip_choose(r.cfg);
        v.tag_rows = db.tag_rows(r.folder, info.id, tag_vocabulary_hash(info.id));
        v.tags_built = true;
    }
    if (r.sq.active()) {
        // Search: rank with the searcher, then show the rows in that order (month / review filters still apply).
        v.searching = true;
        se.load(db, r.folder, r.gen);
        SearchResult sr = se.run(r.cfg, r.sq);
        for (auto& d : sr.decisions) db.add_decision(d);  // jev's verdicts, for the decisions list
        v.search_note = sr.note;
        v.search_error = sr.error;
        std::map<int64_t, PhotoRow> by_id;
        for (auto& row : db.query("", r.review_only, r.month, 1000000, r.folder)) by_id[row.id] = std::move(row);
        for (auto& h : sr.hits) {
            auto it = by_id.find(h.id);
            if (it == by_id.end()) continue;
            it->second.score = h.score;
            it->second.why = h.why;
            v.rows.push_back(std::move(it->second));
        }
    } else {
        v.rows = db.query("", r.review_only, r.month, 20000, r.folder);
    }
    if (r.fav_only) v.rows.erase(std::remove_if(v.rows.begin(), v.rows.end(), [](const PhotoRow& x) { return !x.favorite; }), v.rows.end());
    if (r.ai_only) v.rows.erase(std::remove_if(v.rows.begin(), v.rows.end(), [](const PhotoRow& x) { return !x.ai; }), v.rows.end());
    if (!r.place.empty())
        v.rows.erase(std::remove_if(v.rows.begin(), v.rows.end(), [&](const PhotoRow& x) { return x.location != r.place; }), v.rows.end());

    v.months = db.months(r.folder);
    v.stats = db.stats(r.folder);
    v.decisions = db.decisions(r.folder, 300);
    v.corrections = db.corrections(r.folder, true);
    // One pass over every photo for the sidebar: favorites, places, AI images.
    for (auto& row : db.query("", false, "", 1000000, r.folder)) {
        if (!row.location.empty()) v.places[row.location]++;
        v.ai_count += row.ai;
        if (row.favorite) v.favorites.push_back(std::move(row));
    }
    v.decision_stats = db.decision_stats(r.folder);
    for (auto& d : db.dup_rows(false))  // this folder only
        if (r.folder.empty() || path_under(d.src_path, r.folder)) v.dup_rows.push_back(std::move(d));
    std::map<int64_t, size_t> at;
    for (size_t i = 0; i < v.dup_rows.size(); i++) at[v.dup_rows[i].id] = i;
    std::map<int64_t, std::vector<int64_t>> exact, similar;
    for (auto& d : v.dup_rows) {
        if (d.dup_of && at.count(d.dup_of)) exact[d.dup_of].push_back(d.id);
        if (d.similar_to && at.count(d.similar_to)) similar[d.similar_to].push_back(d.id);
    }
    int64_t reclaim = 0, extra = 0, sim_n = 0, total = 0;
    for (auto& [k, m] : exact) {
        extra += int64_t(m.size());
        for (int64_t id : m) reclaim += v.dup_rows[at[id]].size;
    }
    for (auto& [k, m] : similar) sim_n += int64_t(m.size()) + 1;
    for (auto& d : v.dup_rows) total += d.size;
    v.dup_summary = util::fmt("%zu %s · %s   |   %s: %zu %s, %lld %s, %s %s   |   %s: %zu %s, %lld %s", v.dup_rows.size(), tr("files"),
                              util::human_size(total).c_str(), tr("Exact"), exact.size(), tr("groups"), (long long)extra, tr("extra copies"),
                              util::human_size(reclaim).c_str(), tr("reclaimable"), tr("Similar"), similar.size(), tr("groups"),
                              (long long)sim_n, tr("photos"));
    auto& src = r.dup_similar ? similar : exact;
    // Largest groups first: that is where the space is.
    std::vector<std::pair<int64_t, std::vector<int64_t>>> gs(src.begin(), src.end());
    std::stable_sort(gs.begin(), gs.end(), [&](auto& x, auto& y) {
        return v.dup_rows[at[x.first]].size * int64_t(x.second.size()) > v.dup_rows[at[y.first]].size * int64_t(y.second.size());
    });
    int g = 0;
    for (auto& [keeper, members] : gs) {
        const DupRow& k = v.dup_rows[at[keeper]];
        DupLine h;
        h.id = g + 1;
        h.text = r.dup_similar
                     ? util::fmt("%zu %s · %s %d x %d", members.size() + 1, tr("photos"), tr("largest"), k.width, k.height)
                     : util::fmt("%zu %s · %s %s · %s %s", members.size() + 1, tr("copies"), util::human_size(k.size).c_str(),
                                 tr("each"), util::human_size(k.size * int64_t(members.size())).c_str(), tr("reclaimable"));
        v.dup_lines.push_back(h);
        DupLine kl;
        kl.group = g; kl.id = keeper; kl.keeper = true; kl.similar = r.dup_similar;
        v.dup_lines.push_back(kl);
        for (int64_t id : members) {
            DupLine ml;
            ml.group = g; ml.id = id; ml.similar = r.dup_similar;
            v.dup_lines.push_back(ml);
        }
        g++;
    }
    return v;
}

// Owns its own SQLite connection (WAL readers never wait for the pipeline's writer). Runs UI writes and view
// rebuilds in order; the newest view request wins, so a burst of typing costs one query.
class ViewWorker {
public:
    bool start(const std::string& db_file, std::string& err) {
        if (!db_.open(db_file, err)) return false;
        {
            std::lock_guard<std::mutex> l(mu_);
            quit_ = false;  // a previous stop() (also called before the first start) must not end the new worker
        }
        th_ = std::thread([this] { loop(); });
        return true;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> l(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    void request(const ViewRequest& r) {
        {
            std::lock_guard<std::mutex> l(mu_);
            req_ = r;
            has_req_ = true;
            loading_ = true;
        }
        cv_.notify_all();
    }
    void write(std::function<void(Db&)> fn) {
        {
            std::lock_guard<std::mutex> l(mu_);
            writes_.push_back(std::move(fn));
        }
        cv_.notify_all();
    }
    bool take(ViewData& out) {
        std::lock_guard<std::mutex> l(mu_);
        if (!has_ready_) return false;
        out = std::move(ready_);
        has_ready_ = false;
        return true;
    }
    bool loading() {
        std::lock_guard<std::mutex> l(mu_);
        return loading_;
    }

private:
    void loop() {
        std::unique_lock<std::mutex> l(mu_);
        for (;;) {
            cv_.wait(l, [&] { return quit_ || has_req_ || !writes_.empty(); });
            if (quit_) return;
            auto writes = std::move(writes_);
            writes_.clear();
            bool do_view = has_req_;
            ViewRequest r = req_;
            has_req_ = false;
            l.unlock();
            for (auto& w : writes) w(db_);
            ViewData v;
            if (do_view) v = build_view(db_, searcher_, r);
            l.lock();
            if (do_view) {
                ready_ = std::move(v);
                has_ready_ = true;
                loading_ = has_req_;
                glfwPostEmptyEvent();
            }
        }
    }
    Db db_;
    Searcher searcher_;  // keeps the search index and the CLIP text encoder warm
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, has_req_ = false, has_ready_ = false, loading_ = false;
    ViewRequest req_;
    ViewData ready_;
    std::deque<std::function<void(Db&)>> writes_;
};

// Thumbnails for grids: decoded on a few background threads, uploaded a few per frame, least recently used dropped.
struct GridThumbs {
    struct Done { int64_t id; Thumb t; };
    std::mutex mu;
    std::deque<Done> done;
    std::set<int64_t> pending, failed;  // failed: no thumbnail could be made (not retried)
    std::map<int64_t, Texture> tex;
    std::map<int64_t, double> shown_at;  // when each thumbnail arrived (it fades in)
    std::map<int64_t, int> used;  // id -> frame last drawn
    int frame = 0;
};

// Thumbnail decoding on detached threads; only the newest request is kept (no blocking std::future).
struct ThumbSlot {
    std::mutex mu;
    std::atomic<int64_t> want{0};
    bool ready = false;
    Thumb result;
};

// Keyboard navigation for the lists: arrows or vim keys, Enter to view, Space to toggle, Esc to close.
enum NavKey { NK_NONE, NK_UP, NK_DOWN, NK_LEFT, NK_RIGHT, NK_ENTER, NK_SPACE, NK_ESC, NK_FAV };
static NavKey nav_key(bool allowed) {
    if (!allowed || ImGui::GetIO().WantTextInput || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) return NK_NONE;
    bool ctrl = ImGui::GetIO().KeyCtrl;
    auto p = [ctrl](ImGuiKey k) { return ImGui::IsKeyPressed(k, true) && (!ctrl || k < ImGuiKey_A || k > ImGuiKey_Z); };
    if (p(ImGuiKey_DownArrow) || p(ImGuiKey_J)) return NK_DOWN;
    if (p(ImGuiKey_UpArrow) || p(ImGuiKey_K)) return NK_UP;
    if (p(ImGuiKey_LeftArrow) || p(ImGuiKey_H)) return NK_LEFT;
    if (p(ImGuiKey_RightArrow) || p(ImGuiKey_L)) return NK_RIGHT;
    if (p(ImGuiKey_Enter) || p(ImGuiKey_KeypadEnter)) return NK_ENTER;
    if (p(ImGuiKey_Space)) return NK_SPACE;
    if (p(ImGuiKey_Escape) || p(ImGuiKey_Q)) return NK_ESC;
    if (p(ImGuiKey_F)) return NK_FAV;
    return NK_NONE;
}

// Large photo viewer: the cached thumbnail appears at once, then a sharper 2048 px version replaces it.
static double g_wake_at = 0;

// ---- Motion: a few short, purposeful animations (feedback, where things come from, no jarring pops). Keyboard
// actions and things done 100+ times a day (palette, rename, delete confirm, list navigation, pane toggles) do not
// animate. Curves are the CSS ones: ease-out cubic-bezier(0.23, 1, 0.32, 1), ease-in-out (0.77, 0, 0.175, 1).
namespace anim {
static bool g_reduce = false;  // reduced motion: fades stay (shorter), movement and zoom tweens go

// y of a CSS cubic-bezier(x1, y1, x2, y2) at x (Newton, then bisection), as browsers evaluate it
static float bezier(float x1, float y1, float x2, float y2, float x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    auto bx = [&](float t) { return ((1 - 3 * x2 + 3 * x1) * t + (3 * x2 - 6 * x1)) * t * t + 3 * x1 * t; };
    auto by = [&](float t) { return ((1 - 3 * y2 + 3 * y1) * t + (3 * y2 - 6 * y1)) * t * t + 3 * y1 * t; };
    auto dx = [&](float t) { return 3 * (1 - 3 * x2 + 3 * x1) * t * t + 2 * (3 * x2 - 6 * x1) * t + 3 * x1; };
    float t = x;
    for (int i = 0; i < 8; i++) {
        float e = bx(t) - x, d = dx(t);
        if (std::fabs(e) < 1e-5f) return by(t);
        if (std::fabs(d) < 1e-6f) break;
        t -= e / d;
    }
    float lo = 0, hi = 1;
    t = x;
    for (int i = 0; i < 30; i++) {
        float v = bx(t);
        if (std::fabs(v - x) < 1e-5f) break;
        (v < x ? lo : hi) = t;
        t = (lo + hi) / 2;
    }
    return by(t);
}
static float ease_out(float x) { return bezier(0.23f, 1.0f, 0.32f, 1.0f, x); }
static float ease_in_out(float x) { return bezier(0.77f, 0.0f, 0.175f, 1.0f, x); }

static double now() { return glfwGetTime(); }
static void keep_drawing() { g_wake_at = std::max(g_wake_at, now() + 0.012); }

// Progress 0..1 of something that started at t0 and lasts dur seconds (eased); keeps frames coming until done.
static float progress(double t0, float dur, float (*curve)(float) = ease_out) {
    float x = dur <= 0 ? 1.0f : float((now() - t0) / dur);
    if (x < 1) keep_drawing();
    return curve(std::clamp(x, 0.0f, 1.0f));
}

// A value that eases towards its target and can be retargeted mid-flight (it continues from where it is, like a
// CSS transition, never restarting from zero).
struct Tween {
    float from = 0, to = 0, dur = 0.18f;
    double t0 = -1;
    float value() const {
        if (t0 < 0) return to;
        float x = float((now() - t0) / dur);
        if (x >= 1) return to;
        keep_drawing();
        return from + (to - from) * ease_out(std::max(0.0f, x));
    }
    void set(float target, bool instant = false) {
        if (target == to && t0 >= 0) return;
        if (instant || g_reduce) { from = to = target; t0 = -1; return; }
        from = value();
        to = target;
        t0 = now();
    }
    void jump(float v) { from = to = v; t0 = -1; }
};

// Popovers opened by a click: a 150 ms ease-out fade (opacity only; they appear where ImGui places them).
static std::map<ImGuiID, double> g_popup_open;
static bool begin_popup(const char* id, ImGuiWindowFlags flags = 0) {
    ImGuiID key = ImGui::GetID(id);
    if (!ImGui::IsPopupOpen(id)) { g_popup_open.erase(key); return false; }
    auto it = g_popup_open.find(key);
    if (it == g_popup_open.end()) it = g_popup_open.emplace(key, now()).first;
    float a = progress(it->second, g_reduce ? 0.08f : 0.15f);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.02f, a));
    bool open = ImGui::BeginPopup(id, flags);
    if (!open) ImGui::PopStyleVar();
    return open;
}
static void end_popup() {
    ImGui::EndPopup();
    ImGui::PopStyleVar();
}

}  // namespace anim

static void apply_motion(const Config& c) {
    static int system = -1;  // asked once
    if (c.motion == 0 && system < 0) system = util::system_reduce_motion() ? 1 : 0;
    anim::g_reduce = c.motion == 2 || (c.motion == 0 && system == 1);
}

struct ViewerSlot {
    std::mutex mu;
    std::atomic<int64_t> want{0};
    std::vector<unsigned char> rgba;
    int w = 0, h = 0, stage = 0;
    int64_t id = 0;
    bool ready = false;
    int64_t failed_id = 0;  // the photo that could not be read, and why
    std::string error;
};
struct Viewer {
    bool open = false;
    std::vector<int64_t> ids;
    int idx = 0;
    int64_t loaded_id = 0;
    int shown_stage = 0;
    int opened_frame = -1;  // the key that opened the viewer must not also move it
    float zoom = 1;         // 1 = fit the window; the wheel zooms around the pointer
    ImVec2 pan{0, 0};       // offset of the image centre from the window centre (pixels)
    anim::Tween dz, dx, dy; // what is on screen, easing towards zoom / pan
    double shown_at = 0;    // when the first picture arrived (it fades in)
    Photo p;
    std::shared_ptr<ViewerSlot> slot = std::make_shared<ViewerSlot>();
};


// ---- Tasks: long work waits in a queue and runs one at a time (the pipeline does one run at once).
enum TaskState { TQ_QUEUED, TQ_RUNNING, TQ_PAUSED, TQ_DONE, TQ_STOPPED, TQ_FAILED };
struct QTask {
    int id = 0;
    std::string name;
    Config cfg;
    RunOptions o;
    int last_stages = 0, plan_op = -1;  // applied when it starts: which list the pipeline will hold
    bool planned = false, show_plan = false;
    bool resumable = true;  // analysis stages skip finished work, so a paused one picks up where it stopped
    int state = TQ_QUEUED;
    double t_start = 0, t_end = 0;
    int done = 0, total = 0, errors = 0;
    std::string note;
};

struct App {
    GLFWwindow* win = nullptr;
    Config cfg;
    std::string config_file;
    Db db;
    Log log;
    Pipeline* pipe = nullptr;
    RunOptions opts;

    // photos view + search
    char search[256] = "";
    bool in_name = true, in_exif = true, in_desc = true, in_prompt = true;
    std::string date_from, date_to;  // "YYYY-MM-DD" or ""
    bool searching = false;
    std::string search_note, search_error;
    std::atomic<int> catalog_gen{0};  // bumps after every run (and Trash moves) so the search index reloads
    int last_stages = 0;   // stages of the current run (for "step 3 of 5")
    bool show_log = false;
    bool review_only = false;
    std::string month;  // "" all, "undated", "2019-05"
    std::vector<PhotoRow> rows;
    std::vector<MonthInfo> months;
    DbStats stats;
    bool rows_dirty = true;
    double last_refresh = 0;
    int64_t selected = 0;
    Photo sel;
    bool sel_loaded = false;
    char manual_buf[64] = "";

    Texture thumb;
    int64_t thumb_id = 0, thumb_failed = 0;
    double thumb_shown_at = 0;
    std::shared_ptr<ThumbSlot> thumb_slot = std::make_shared<ThumbSlot>();
    ViewWorker view;
    std::atomic<bool> want_dupes_run{false};  // set by the worker once a keeper pin is stored
    int view_gen = 0;                          // bumps whenever fresh view data arrives
    Db ui_db;  // point lookups for the detail panes (own connection, never waits on the pipeline)

    // health
    std::atomic<int> jev_ok{-1}, vl_ok{-1}, clip_ok{-1}, llm_ok{-1};
    std::mutex health_mu;
    std::string jev_detail, vl_detail, clip_detail, llm_detail;
    std::atomic<bool> health_busy{false};
    double health_at = -100;

    // settings edit buffers
    char lib_buf[1024] = "", src_buf[1024] = "";
    char jev_url[512] = "", jev_model[256] = "", jev_key[256] = "";
    char vl_url[512] = "", vl_model[512] = "", vl_key[256] = "", vl_lang[64] = "";
    char llm_url[512] = "", llm_model[256] = "", llm_key[256] = "";
    bool settings_loaded = false;
    std::string toast;
    double toast_until = 0;
    std::atomic<bool> dlg_running{false};
    std::mutex dlg_mu;
    std::string dlg_result;
    int dlg_target = 0;  // 1 library, 2 source
    bool want_font_rebuild = false;

    // path + preview
    char folder_buf[1024] = "";
    bool last_run_planned = false, show_plan_tab = false;
    std::vector<PlanItem> plan;
    double plan_at = -1;
    int plan_sel = -1;
    bool plan_review_only = false, plan_decide_only = false;
    char plan_filter[128] = "";

    // duplicates view
    std::vector<DupRow> dup_rows;
    std::vector<DupLine> dup_lines;
    bool dups_dirty = true, dup_show_similar = false;
    int64_t dup_sel = 0;
    std::string dup_summary;
    std::string start_tab;  // --tab: photos | preview | dupes | log | settings
    std::string gl_renderer;

    // viewer, navigation, home
    Viewer viewer;
    std::shared_ptr<GridThumbs> grid = std::make_shared<GridThumbs>();
    int64_t fav_sel = 0;
    Texture viewer_tex;
    bool scroll_sel = false;       // bring the selected row into view (after keyboard navigation)
    bool show_overview = false;    // switch to the Overview tab (after Analyze)
    bool on_overview = false;
    // Tags & EXIF tab
    std::vector<TagRow> tag_rows;
    bool on_tags = false, tags_built = false;
    int tags_filter = 0;
    char tags_search[128] = "";
    int64_t tags_sel = 0, tags_edit_id = 0;
    char tags_edit[512] = "";
    struct MetaPreview {
        std::mutex mu;
        int64_t id = 0;
        bool ready = false;
        std::string file;
        WriteResult r;
    };
    std::shared_ptr<MetaPreview> meta_preview = std::make_shared<MetaPreview>();
    int64_t meta_preview_id = 0;
    // favorites, corrections, search translation
    std::vector<PhotoRow> favorites;
    std::string fav_tag_filter;
    bool fav_only = false;   // Photos: only starred photos
    // layout
    int section = SEC_ALL;
    std::string section_folder, place;
    bool sidebar_on = true, inspector_on = true, show_settings = false, ai_only = false;
    float sidebar_w = 220, inspector_w = 340;
    std::map<std::string, int> places;
    int ai_count = 0;
    // search: tokens (chips) + free text; Enter on a sentence asks the LLM
    std::vector<std::pair<std::string, std::string>> tokens;  // label, query
    bool ask_go = false, regex = false, sugg_open = false;
    int sugg_sel = -1;
    std::vector<std::string> sugg_tags;  // every tag seen in the catalog (for suggestions)
    int sugg_tags_gen = -1;
    std::string undo_run, undo_text;  // the last applied run (Undo banner)
    bool focus_search = false;        // Ctrl+F: put the cursor in the search field
    int help_tab = -1;                // open Help on this tab (Ctrl+/ -> shortcuts)
    bool status_open_req = false;     // open the tasks and logs popover (Ctrl+J)
    // Rename (F2, context menus, inspector, viewer)
    int64_t rename_id = 0;
    bool rename_open = false;
    char rename_buf[512] = "";
    std::string rename_ext, rename_err;
    double undo_until = 0;
    std::set<int64_t> multi;  // Photos: selected photos (Ctrl/Shift+click); `selected` is the one in focus
    int64_t anchor = 0;
    bool photo_trash_confirm = false, trash_with_copies = false;
    // CLIP tab
    struct SmartItem {
        int64_t id = 0;
        std::string path, action, reason, by;  // action: reanalyse | retag
        double worth = 0;
        bool tick = false;
    };
    struct SmartState {
        std::mutex mu;
        bool done = false;
        std::vector<SmartItem> items;
        std::string note;
    };
    std::shared_ptr<SmartState> smart;
    bool smart_open = false, corr_confirm = false, ana_check = false;
    ClipCounts clip_counts, clip_counts_quick;
    int clip_counts_gen = -1;
    // tag list editor
    bool tag_editor_open = false, ed_loaded = false, ed_dirty = false;
    std::vector<TagDef> ed_tags;
    char ed_search[96] = "", ed_name[96] = "", ed_newcat[64] = "";
    int ed_cat = 0, ed_page = 0, ed_edit = -1, ed_form_cat = 0;
    int plan_op = OP_COPY;   // the file operation the current organize list was built for (OP_METADATA = Metadata's list)
    std::string meta_sub;    // Metadata inner tab to open next
    // pane sizes (splitters); 0 = not set yet (a share of the window)
    float months_w = 190, photo_detail_w = 0, plan_detail_h = 230, tags_detail_w = 0, dup_pane_w = 360;
    bool show_help = false;
    std::string action_sub;  // Actions sub-tab to open next
    // Analyze dialog: how much image analysis (CLIP) and tagging to do
    bool analyze_dlg = false;
    ClipCounts ana_counts, ana_counts_quick;
    int ana_clip = 0, ana_tags = 0;
    bool photo_grid = true;  // the library: thumbnails (default) or the table
    float grid_cell = 168;   // thumbnail cell size (Ctrl+wheel)
    int auto_meta_step = 0;  // Auto set meta after Analyze: 1 = list being built, 2 = being applied
    // Screenshot (top bar): 1 = whole window on the next frames, 2 = choosing an area, 3 = area chosen, capture next
    bool show_model_tests = false, palette_open = false;
    std::deque<QTask> tasks;          // running (front), queued, paused
    std::vector<QTask> task_done;     // finished, newest first
    int next_task_id = 1, batch_total = 0, batch_done = 0;
    int task_stop_as = -1;
    double status_seen_at = 0;        // errors after this time put a red dot on the status button            // what the running task becomes when its stop request lands (TQ_PAUSED / TQ_STOPPED)
    int shot_mode = 0, shot_frames = 0;
    ImVec2 shot_a, shot_b;
    std::string shot_last;
    std::vector<Correction> corrections;
    std::map<int64_t, bool> corr_sel;  // correction id -> ticked (default: confidence >= 70%)
    int64_t corr_focus = 0;
    double search_edit_at = -1;   // when the search text last changed (translations wait for a pause)
    bool search_translate = false;
    // jev decisions
    std::vector<Decision> decisions;
    DecisionStats decision_stats;
    bool show_decisions = false;
    // Photos table sorting
    int sorted_gen = -1;
    // moving duplicate copies to the Trash
    bool trash_confirm = false;
    std::atomic<bool> trash_busy{false};
    std::atomic<int> trash_done{0}, trash_failed{0};
    std::string trash_what;
};

static ImVec4 g_accent(0.30f, 0.78f, 0.47f, 1);
// Theme colours for meaning (set by apply_style): danger, warning, a warning about originals, success, error, info.
static ImVec4 kDanger(0.86f, 0.30f, 0.27f, 1);
static ImVec4 g_warn(0.95f, 0.72f, 0.4f, 1), g_warn_red(1.0f, 0.62f, 0.55f, 1), g_ok(0.5f, 0.9f, 0.6f, 1), g_err(1, 0.5f, 0.4f, 1),
    g_info(0.6f, 0.75f, 1, 1), g_star(1, 0.8f, 0.25f, 1), g_text_dim(0.55f, 0.57f, 0.62f, 1), g_sidebar(0.09f, 0.10f, 0.12f, 1),
    g_panel(0.11f, 0.12f, 0.15f, 1), g_seg_on(1, 1, 1, 0.14f);  // cards, and the chosen pill of a segmented control
static bool g_seg_shadow = false;                                // the pill casts a soft shadow (light theme)

// Themes: Dark, Tokyo Night (the editor theme: #1a1b26 night, #7aa2f7 blue) and Light. The accent comes from the
// settings (choosing a theme sets its own accent first).
struct ThemeColors {
    ImVec4 bg, sidebar, panel, frame, frame_hover, text, dim, border, popup, danger, warn, warn_red, ok, err, info, star;
};
static ThemeColors theme_colors(int t) {
    auto rgb = [](int hex, float a = 1) { return ImVec4(((hex >> 16) & 255) / 255.0f, ((hex >> 8) & 255) / 255.0f, (hex & 255) / 255.0f, a); };
    if (t == THEME_TOKYO)
        return {rgb(0x1a1b26), rgb(0x16161e), rgb(0x1f2335), rgb(0x292e42), rgb(0x343a55), rgb(0xc0caf5), rgb(0x8089b3), rgb(0x292e42),
                rgb(0x1f2335, 0.98f), rgb(0xf7768e), rgb(0xe0af68), rgb(0xff9e64), rgb(0x9ece6a), rgb(0xf7768e), rgb(0x7dcfff), rgb(0xe0af68)};
    if (t == THEME_LIGHT)
        return {rgb(0xf5f5f7), rgb(0xe9e9ec), rgb(0xffffff), ImVec4(0, 0, 0, 0.06f), ImVec4(0, 0, 0, 0.10f), rgb(0x1d1d1f), rgb(0x6e6e73),
                ImVec4(0, 0, 0, 0.12f), rgb(0xffffff, 0.99f), rgb(0xd70015), rgb(0xa05a00), rgb(0xc93400), rgb(0x248a3d), rgb(0xd70015),
                rgb(0x0058d0), rgb(0xb25000)};
    return {ImVec4(0.07f, 0.08f, 0.10f, 1), ImVec4(0.09f, 0.10f, 0.12f, 1), ImVec4(0.11f, 0.12f, 0.15f, 1), ImVec4(1, 1, 1, 0.08f),
            ImVec4(1, 1, 1, 0.14f), ImVec4(0.92f, 0.93f, 0.95f, 1), ImVec4(0.55f, 0.57f, 0.62f, 1), ImVec4(1, 1, 1, 0.08f),
            ImVec4(0.08f, 0.09f, 0.11f, 0.98f), ImVec4(0.86f, 0.30f, 0.27f, 1), ImVec4(0.95f, 0.72f, 0.4f, 1), ImVec4(1.0f, 0.62f, 0.55f, 1),
            ImVec4(0.5f, 0.9f, 0.6f, 1), ImVec4(1, 0.5f, 0.4f, 1), ImVec4(0.6f, 0.75f, 1, 1), ImVec4(1, 0.8f, 0.25f, 1)};
}
static void theme_default_accent(int t, float* acc) {
    const float d[3][3] = {{0.30f, 0.78f, 0.47f}, {0.478f, 0.635f, 0.969f}, {0.0f, 0.478f, 1.0f}};  // dark: the app's green; Tokyo blue; system blue
    for (int k = 0; k < 3; k++) acc[k] = d[std::clamp(t, 0, 2)][k];
}

// The theme in effect: System follows the desktop (asked once per start, and when chosen again).
static int g_system_dark = -1;
static int theme_kind(const Config& c) {
    if (c.theme != THEME_SYSTEM) return std::clamp(c.theme, 0, 2);
    if (g_system_dark < 0) g_system_dark = util::system_dark_mode() ? 1 : 0;
    return g_system_dark ? THEME_DARK : THEME_LIGHT;
}

static void apply_style(const Config& c) {
    ImGuiStyle& st = ImGui::GetStyle();
    int tk = theme_kind(c);
    ThemeColors th = theme_colors(tk);
    if (tk == THEME_LIGHT) ImGui::StyleColorsLight(&st);
    else ImGui::StyleColorsDark(&st);
    st.WindowRounding = 8.0f;
    st.FrameRounding = 6.0f;
    st.PopupRounding = 8.0f;
    st.GrabRounding = 6.0f;
    st.ChildRounding = 8.0f;
    st.TabRounding = 6.0f;
    st.ScrollbarRounding = 6.0f;
    st.WindowPadding = ImVec2(12, 10);
    st.FramePadding = ImVec2(8, 5);
    st.ItemSpacing = ImVec2(8, 6);
    st.WindowBorderSize = 1;  // dialogs and floating windows get a thin frame (the full-window views turn it off)
    st.ChildBorderSize = 1;
    st.PopupBorderSize = 1;
    st.ScrollbarSize = 12;
    st.DisabledAlpha = 0.45f;
    ImVec4 acc(c.accent[0], c.accent[1], c.accent[2], 1);
    auto A = [&](float a) { return ImVec4(acc.x, acc.y, acc.z, a); };
    ImVec4* k = st.Colors;
    k[ImGuiCol_WindowBg] = th.bg;
    k[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    k[ImGuiCol_PopupBg] = th.popup;
    k[ImGuiCol_Text] = th.text;
    k[ImGuiCol_TextDisabled] = th.dim;
    k[ImGuiCol_Border] = tk == THEME_LIGHT ? ImVec4(0, 0, 0, 0.18f) : th.border;  // white on white needs a visible edge
    k[ImGuiCol_Separator] = th.border;
    k[ImGuiCol_CheckMark] = acc;
    k[ImGuiCol_SliderGrab] = acc;
    k[ImGuiCol_SliderGrabActive] = acc;
    k[ImGuiCol_PlotHistogram] = acc;
    k[ImGuiCol_Button] = th.frame;
    k[ImGuiCol_ButtonHovered] = th.frame_hover;
    k[ImGuiCol_ButtonActive] = A(0.55f);
    k[ImGuiCol_FrameBg] = th.frame;
    k[ImGuiCol_FrameBgHovered] = th.frame_hover;
    k[ImGuiCol_FrameBgActive] = A(0.30f);
    k[ImGuiCol_Header] = A(0.28f);
    k[ImGuiCol_HeaderHovered] = A(0.18f);
    k[ImGuiCol_HeaderActive] = A(0.40f);
    k[ImGuiCol_Tab] = th.frame;
    k[ImGuiCol_TabHovered] = A(0.35f);
    k[ImGuiCol_TabSelected] = A(0.30f);
    k[ImGuiCol_TableHeaderBg] = th.frame;
    k[ImGuiCol_TableRowBgAlt] = ImVec4(th.text.x, th.text.y, th.text.z, 0.025f);
    k[ImGuiCol_TableBorderLight] = th.border;
    k[ImGuiCol_TableBorderStrong] = th.border;
    k[ImGuiCol_TitleBg] = th.sidebar;
    k[ImGuiCol_TitleBgActive] = th.panel;
    k[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    k[ImGuiCol_NavCursor] = A(0.6f);  // keyboard focus ring
    k[ImGuiCol_TextSelectedBg] = A(0.35f);
    k[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, tk == THEME_LIGHT ? 0.25f : 0.55f);
    g_accent = acc;
    kDanger = th.danger;
    g_warn = th.warn;
    g_warn_red = th.warn_red;
    g_ok = th.ok;
    g_err = th.err;
    g_info = th.info;
    g_star = th.star;
    g_text_dim = th.dim;
    g_sidebar = th.sidebar;
    g_panel = th.panel;
    g_seg_on = tk == THEME_LIGHT ? ImVec4(1, 1, 1, 1) : th.frame_hover;
    g_seg_shadow = tk == THEME_LIGHT;
}

static std::string fmt_eta(double s) {
    if (s < 0 || !std::isfinite(s)) return "--";
    int t = int(s + 0.5);
    return t >= 3600 ? util::fmt("%dh%02dm", t / 3600, t / 60 % 60) : util::fmt("%dm%02ds", t / 60, t % 60);
}

static bool open_db(App& a) {
    a.db.close();
    std::string err;
    if (!a.db.open(db_path(a.cfg), err)) {
        a.log.add(2, "cannot open database: " + err);
        return false;
    }
    if (a.win) {  // GUI: separate connections for the UI so it never waits on the pipeline
        a.view.stop();
        a.ui_db.close();
        if (!a.view.start(db_path(a.cfg), err) || !a.ui_db.open(db_path(a.cfg), err)) a.log.add(2, "cannot open database: " + err);
    }
    a.rows_dirty = true;
    return true;
}

static void check_health(App& a) {
    if (a.health_busy.exchange(true)) return;
    Config c = a.cfg;
    std::thread([&a, c] {
        std::string jd = "disabled", vd = "disabled";
        int j = -1, v = -1;
        if (c.jev_enabled) j = jev_health(c, jd) ? 1 : 0;
        if (c.vl_enabled) v = vl_health(c, vd) && vd == "ok" ? 1 : 0;
        std::string cd = "disabled", ld = "disabled";
        int k = -1, m = -1;
        if (c.clip_enabled) k = clip_status(c, cd) ? 1 : 0;
        if (c.llm_enabled) m = llm_health(c, ld) ? 1 : 0;
        {
            std::lock_guard<std::mutex> l(a.health_mu);
            a.jev_detail = jd;
            a.vl_detail = vd;
            a.clip_detail = cd;
            a.llm_detail = ld;
        }
        a.clip_ok = k;
        a.llm_ok = m;
        a.jev_ok = j;
        a.vl_ok = v;
        a.health_busy = false;
        glfwPostEmptyEvent();
    }).detach();
}

static void start_dir_dialog(App& a, int target) {
    if (a.dlg_running.exchange(true)) return;
    a.dlg_target = target;
    std::thread([&a] {
        std::string chosen = util::choose_folder(tr("Choose folder"));
        {
            std::lock_guard<std::mutex> l(a.dlg_mu);
            a.dlg_result = chosen;
        }
        a.dlg_running = false;
        glfwPostEmptyEvent();
    }).detach();
}

static void load_settings_buffers(App& a) {
    auto cp = [](char* d, size_t n, const std::string& s) { snprintf(d, n, "%s", s.c_str()); };
    cp(a.lib_buf, sizeof a.lib_buf, a.cfg.output);
    cp(a.jev_url, sizeof a.jev_url, a.cfg.jev_url);
    cp(a.jev_model, sizeof a.jev_model, a.cfg.jev_model);
    cp(a.jev_key, sizeof a.jev_key, a.cfg.jev_key);
    cp(a.vl_url, sizeof a.vl_url, a.cfg.vl_url);
    cp(a.vl_model, sizeof a.vl_model, a.cfg.vl_model);
    cp(a.vl_key, sizeof a.vl_key, a.cfg.vl_key);
    cp(a.vl_lang, sizeof a.vl_lang, a.cfg.vl_tag_lang);
    cp(a.llm_url, sizeof a.llm_url, a.cfg.llm_url);
    cp(a.llm_model, sizeof a.llm_model, a.cfg.llm_model);
    cp(a.llm_key, sizeof a.llm_key, a.cfg.llm_key);
    a.settings_loaded = true;
}

// View all folders at once (or go back to one).
static void set_view_all(App& a, bool all) {
    a.cfg.view_all = all && a.cfg.folders.size() > 1;
    save_config(a.cfg, a.config_file);
    a.month.clear();
    a.selected = 0;
    a.rows_dirty = a.dups_dirty = true;
}

static void set_folder(App& a, const std::string& f_in) {
    std::error_code ec;
    std::string f = util::canonical(util::trim(f_in));
    if (f.empty()) return;
    if (!util::dir_exists(f)) {
        a.log.add(2, "folder not found: " + f);
        a.toast = std::string(tr("Folder not found")) + ": " + f;
        a.toast_until = glfwGetTime() + 3;
        return;
    }
    remember_folder(a.cfg, f);
    a.cfg.view_all = false;
    snprintf(a.folder_buf, sizeof a.folder_buf, "%s", f.c_str());
    save_config(a.cfg, a.config_file);
    a.month.clear();
    a.selected = 0;
    a.rows_dirty = a.dups_dirty = true;
}

// op >= 0: build the organize list with this file operation (Metadata's "write into files" uses OP_METADATA without
// changing the Organization choice).
static void launch_task(App& a, QTask& t) {
    a.last_stages = t.last_stages;
    a.last_run_planned = t.planned;
    if (t.plan_op >= 0) a.plan_op = t.plan_op;
    if (t.show_plan) a.show_plan_tab = true;
    if (!t.o.undo_run.empty()) a.undo_until = 0;
    t.state = TQ_RUNNING;
    t.t_start = glfwGetTime();
    a.task_stop_as = -1;
    a.pipe->start(t.cfg, t.o);
    a.log.add(0, "task started: " + t.name);
}

// Start the next queued task when the pipeline is free (paused ones wait for Resume).
static void start_next_task(App& a) {
    if (a.pipe->running()) return;
    for (auto& t : a.tasks)
        if (t.state == TQ_RUNNING) return;
    for (auto& t : a.tasks)
        if (t.state == TQ_QUEUED) { launch_task(a, t); return; }
}

// Every long run goes through here: now when nothing runs, else after what is queued.
static void submit_task(App& a, const std::string& name, const Config& cfg, const RunOptions& o, int last_stages, bool planned, int plan_op, bool show_plan) {
    QTask t;
    t.id = a.next_task_id++;
    t.name = name;
    t.cfg = cfg;
    t.o = o;
    t.last_stages = last_stages;
    t.planned = planned;
    t.plan_op = plan_op;
    t.show_plan = show_plan;
    t.resumable = !o.apply_plan && o.undo_run.empty() && o.reanalyse_ids.empty() && o.retag_ids.empty();
    bool idle = a.tasks.empty();
    if (idle) a.batch_total = a.batch_done = 0;
    a.batch_total++;
    a.tasks.push_back(t);
    if (!idle || a.pipe->running()) {
        a.toast = std::string(tr("Queued: ")) + tr(name.c_str());
        a.toast_until = glfwGetTime() + 2.5;
    }
    start_next_task(a);
}

// The pipeline finished a run: record it on the running task, then go on with the queue.
static void task_finished(App& a) {
    for (auto it = a.tasks.begin(); it != a.tasks.end(); ++it) {
        if (it->state != TQ_RUNNING) continue;
        QTask t = *it;
        t.done = a.pipe->progress.done;
        t.total = a.pipe->progress.total;
        t.errors = a.pipe->progress.errors;
        t.t_end = glfwGetTime();
        if (a.task_stop_as == TQ_PAUSED && t.resumable) {  // stays in the queue, waiting for Resume
            it->state = TQ_PAUSED;
            it->note = util::fmt("%d / %d", t.done, t.total);
            break;
        }
        t.state = a.task_stop_as == TQ_STOPPED ? TQ_STOPPED : TQ_DONE;
        if (t.errors) t.note = util::fmt("%d %s", t.errors, tr("errors"));
        a.tasks.erase(it);
        a.task_done.insert(a.task_done.begin(), t);
        if (a.task_done.size() > 30) a.task_done.pop_back();
        a.batch_done++;
        break;
    }
    a.task_stop_as = -1;
    start_next_task(a);
}

// Waiting work survives a restart: resumable tasks come back paused.
static std::string queue_path() { return util::data_dir() + "/queue.json"; }
static void save_queue(App& a) {
    json arr = json::array();
    for (auto& t : a.tasks) {
        if (!t.resumable) continue;
        arr.push_back({{"name", t.name}, {"stages", t.o.stages}, {"last_stages", t.last_stages}, {"planned", t.planned}, {"plan_op", t.plan_op},
                       {"plan_only", t.o.plan_only}, {"redecide_all", t.o.redecide_all}, {"retag_all", t.o.retag_all}, {"recheck_all", t.o.recheck_all},
                       {"clip_mode", t.o.clip_mode}, {"tag_mode", t.o.tag_mode}, {"file_op", t.cfg.file_op}});
    }
    if (arr.empty()) remove(queue_path().c_str());
    else util::write_file(queue_path(), arr.dump(1));
}
static void load_queue(App& a) {
    std::string s;
    if (!util::read_file(queue_path(), s)) return;
    json arr = json::parse(s, nullptr, false);
    if (!arr.is_array()) return;
    for (auto& j : arr) {
        QTask t;
        t.id = a.next_task_id++;
        t.name = j.value("name", "Run");
        t.cfg = effective(a.cfg);
        t.cfg.file_op = j.value("file_op", t.cfg.file_op);
        t.o = a.opts;
        t.o.stages = j.value("stages", 0);
        t.o.scope = all_scope(a.cfg);
        t.o.plan_only = j.value("plan_only", false);
        t.o.redecide_all = j.value("redecide_all", false);
        t.o.retag_all = j.value("retag_all", false);
        t.o.recheck_all = j.value("recheck_all", false);
        t.o.clip_mode = j.value("clip_mode", 0);
        t.o.tag_mode = j.value("tag_mode", 0);
        t.last_stages = j.value("last_stages", 0);
        t.planned = j.value("planned", false);
        t.plan_op = j.value("plan_op", -1);
        t.state = TQ_PAUSED;
        t.note = tr("from the last session");
        a.tasks.push_back(t);
        a.batch_total++;
    }
}

static void start_run(App& a, int stages, bool force_preview = false, int op = -1) {
    if (a.cfg.folders.empty()) {  // nothing chosen yet: ask for the folder first
        start_dir_dialog(a, 3);
        return;
    }
    if (!a.db.is_open() && !open_db(a)) return;
    for (auto& f : a.cfg.folders)
        if (!util::dir_exists(f)) a.log.add(1, "folder not found (skipped): " + f);
    RunOptions o = a.opts;
    o.stages = stages;
    o.scope = all_scope(a.cfg);  // every folder in the list, together
    o.plan_only = (stages & ST_ORGANIZE) && (a.cfg.preview_first || force_preview || a.cfg.dry_run);
    Config run = effective(a.cfg);
    if (op >= 0) run.file_op = op;
    const char* name = (stages & ST_SCAN) && (stages & ST_DECIDE) ? "Analyze"
                     : stages == ST_ORGANIZE ? (run.file_op == OP_METADATA ? "List what to write into files" : "Build the organize list")
                     : stages == ST_SCAN ? "Scan for new or changed files" : stages == ST_DUPES ? "Find duplicates"
                     : stages == ST_DECIDE ? "Work out dates and places" : stages == ST_TAG ? "Image recognition"
                     : stages == ST_VISION ? "Describe with the vision model" : stages == ST_FIX ? "Check tags" : "Run";
    submit_task(a, name, run, o, stages, o.plan_only, (stages & ST_ORGANIZE) ? run.file_op : -1, false);
}

static int analyze_stages(const Config& c) {
    // The correction check uses the LLM (GPU server): only when chosen in the Analyze dialog.
    return ST_SCAN | ST_DUPES | ST_DECIDE | ST_ORGANIZE | (c.clip_enabled ? ST_TAG : 0) | (c.vl_enabled ? ST_VISION : 0);
}

static void apply_plan(App& a) {
    RunOptions o = a.opts;
    o.stages = ST_ORGANIZE;
    o.apply_plan = true;
    o.scope = all_scope(a.cfg);  // every folder in the list, together
    submit_task(a, a.plan_op == OP_METADATA ? "Write info into files" : "Apply the organize list", effective(a.cfg), o, ST_ORGANIZE, false, -1, true);
}

// ---------------------------------------------------------------------------
// UI pieces

// Tooltips and button styles used everywhere: the accent colour for the main action of a screen, red for anything
// that changes or removes original files.
// Tooltips appear after the mouse rests on an item for 2 seconds. The main loop is asked to draw a frame then
// (it otherwise sleeps until the next event).

static void tip(const std::string& text);
// The hover delay before a tooltip shows (one constant); after one has shown, the next shows at once for 0.5 s
// ("warm"), so moving along a toolbar reads every button without waiting again.
static const double kTooltipDelay = 2.0;
static double g_tip_warm_until = 0;

// Keyboard keys as small chips: "Ctrl+P" -> [Ctrl] [P]
static void key_chips(const char* keys) {
    if (!keys || !*keys) return;
    std::string k = keys;
    std::vector<std::string> parts;
    size_t st = 0;
    for (size_t i = 1; i <= k.size(); i++)  // split on '+' but keep a key that is itself '+' ("Ctrl++")
        if (i == k.size() || (k[i] == '+' && i > st)) { parts.push_back(k.substr(st, i - st)); st = i + 1; }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < parts.size(); i++) {
        ImGui::SameLine(0, i ? 3.0f : 10.0f);
        ImVec2 ts = ImGui::CalcTextSize(parts[i].c_str());
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImVec2 q(p.x + ts.x + 10, p.y + ts.y + 2);
        dl->AddRectFilled(ImVec2(p.x, p.y - 1), q, ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
        dl->AddRect(ImVec2(p.x, p.y - 1), q, ImGui::GetColorU32(ImGuiCol_Border), 4.0f);
        dl->AddText(ImVec2(p.x + 5, p.y), ImGui::GetColorU32(ImGuiCol_Text), parts[i].c_str());
        ImGui::Dummy(ImVec2(ts.x + 10, ts.y));
    }
}

static void tip_impl(const char* text, const char* keys) {
    static ImVec2 last_min(-1, -1), last_max(-1, -1);
    static double since = 0;
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) return;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) return;  // hides on press
    ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
    double now = glfwGetTime();  // same clock as the main loop's wake-up
    if (mn.x != last_min.x || mn.y != last_min.y || mx.x != last_max.x || mx.y != last_max.y) {
        last_min = mn;
        last_max = mx;
        since = now < g_tip_warm_until ? now - kTooltipDelay : now;  // warm: show at once
    }
    if (now - since < kTooltipDelay) {
        g_wake_at = since + kTooltipDelay + 0.02;
        return;
    }
    g_tip_warm_until = now + 0.5;
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    key_chips(keys);
    ImGui::EndTooltip();
}
static void tip(const char* text) { tip_impl(text, nullptr); }
// A tooltip that names the action and shows its keys as chips.
static void tipk(const char* text, const char* keys) { tip_impl(text, keys); }
static void tip(const std::string& text) { tip(text.c_str()); }

// Splitters: a thin bar between two panes that can be dragged. vsplit sits between a left and a right pane
// (call it after the left one, then SameLine); hsplit between a top and a bottom pane. `size` is the size of the
// pane the bar resizes; `grow_right` = the pane is on the right/bottom side (dragging right makes it smaller).
// Splitters: follow the pointer 1:1 from where it was grabbed, clamp to the panes' limits, accent line while hot,
// double-click restores the default size, and (when `collapsed` is given) dragging 40 px past the minimum collapses
// the pane.
static void split_drag(ImGuiID id, float& size, float min, float max, float delta_sign, float axis_delta, float def, bool* collapsed) {
    static ImGuiID active = 0;
    static float start = 0, moved = 0;
    if (ImGui::IsItemActivated()) { active = id; start = size; moved = 0; }
    if (ImGui::IsItemActive() && active == id) {
        moved += delta_sign * axis_delta;
        float want = start + moved;
        if (collapsed && want < min - 40) { *collapsed = true; size = start; return; }
        size = std::clamp(want, min, max);
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && def > 0) size = def;
}
static void vsplit(const char* id, float& size, float min, float max, bool grow_right, float def = 0, bool* collapsed = nullptr) {
    ImGui::SameLine(0, 0);
    ImGui::InvisibleButton(id, ImVec2(7, std::max(1.0f, ImGui::GetContentRegionAvail().y)));
    bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
    if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    split_drag(ImGui::GetItemID(), size, min, max, grow_right ? -1.0f : 1.0f, ImGui::GetIO().MouseDelta.x, def, collapsed);
    if (hot) tip(def > 0 ? tr("Drag to resize; double-click for the default size") : tr("Drag to resize"));
    ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    float x = std::floor((a.x + b.x) / 2) + 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(x, a.y + 2), ImVec2(x, b.y - 2), ImGui::GetColorU32(hot ? g_accent : ImGui::GetStyleColorVec4(ImGuiCol_Border)),
                                        hot ? 2.0f : 1.0f);
    ImGui::SameLine(0, 0);
}
static void hsplit(const char* id, float& size, float min, float max, bool grow_down, float def = 0) {
    ImGui::InvisibleButton(id, ImVec2(std::max(1.0f, ImGui::GetContentRegionAvail().x), 7));
    bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
    if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    split_drag(ImGui::GetItemID(), size, min, max, grow_down ? -1.0f : 1.0f, ImGui::GetIO().MouseDelta.y, def, nullptr);
    ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    float y = std::floor((a.y + b.y) / 2) + 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(a.x + 2, y), ImVec2(b.x - 2, y), ImGui::GetColorU32(hot ? g_accent : ImGui::GetStyleColorVec4(ImGuiCol_Border)),
                                        hot ? 2.0f : 1.0f);
}

static bool styled_button(const char* label, const ImVec4& col, const ImVec2& size, const char* tip_text) {
    // Solid fill; the label is dark or light, whichever reads better on the colour.
    float lum = 0.299f * col.x + 0.587f * col.y + 0.114f * col.z;
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(col.x, col.y, col.z, 0.88f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(col.x, col.y, col.z, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(col.x * 0.85f, col.y * 0.85f, col.z * 0.85f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, lum > 0.62f ? ImVec4(0.06f, 0.07f, 0.09f, 1) : ImVec4(1, 1, 1, 1));
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(4);
    if (tip_text) tip(tip_text);
    return clicked;
}
static bool primary_button(const char* label, const char* tip_text, const ImVec2& size = ImVec2(0, 0)) { return styled_button(label, g_accent, size, tip_text); }
static bool danger_button(const char* label, const char* tip_text, const ImVec2& size = ImVec2(0, 0)) { return styled_button(label, kDanger, size, tip_text); }
// Apply of the organize list: red when it moves or renames the originals (or writes into them).
static bool apply_button(const Config& c, const char* label, const ImVec2& size = ImVec2(0, 0), int op = -1) {
    if (op < 0) op = c.file_op;
    bool originals = op == OP_MOVE || op == OP_RENAME || (op == OP_METADATA && c.write_mode == WRITE_EMBED);
    const char* t = op == OP_MOVE ? "Moves your original files into the month folders and renames them. No copies are kept."
                  : op == OP_RENAME ? "Renames your original files in their folders."
                  : op == OP_METADATA ? "Adds the listed fields into the files themselves (existing values are never changed)."
                  : "Copies the photos into the organized folder with date names; your originals are not touched.";
    return originals ? danger_button(label, tr(t), size) : primary_button(label, tr(t), size);
}

// A path as shown: relative to the photo folder it is in (prefixed with that folder's name when there are several).
static std::string rel_path(const Config& c, const std::string& p) {
    for (auto& f : c.folders)
        if (util::starts_with(p, f + "/")) return (c.folders.size() > 1 ? util::basename(f) + "/" : std::string()) + p.substr(f.size() + 1);
    return p;
}

static void status_dot(int ok, const char* label, const std::string& detail) {
    // green = answered the last automatic test, yellow = down / not ready, gray = not tested yet (or disabled)
    ImVec4 col = ok == 1 ? g_ok : ok == 0 ? g_warn : g_text_dim;
    ImGui::BeginGroup();
    ImGui::TextColored(col, "\xE2\x97\x8F");
    ImGui::SameLine(0, 4);
    ImGui::TextUnformatted(label);
    ImGui::EndGroup();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s\n%s", ok == 1 ? tr("up (checked automatically every 20 s)") : ok == 0 ? tr("down or not ready") : tr("checking..."),
                          detail.c_str());
}

static void draw_log(App& a);
static void draw_photo_grid(App& a);
static void request_analyze(App& a);
// The search as the searcher sees it: the chips' queries, then the typed text.
static std::string compose_query(const App& a) {
    std::string q;
    for (auto& [label, query] : a.tokens) q += (q.empty() ? "" : " ") + query;
    std::string typed = util::trim(a.search);
    if (!typed.empty()) q += (q.empty() ? "" : " ") + typed;
    return q;
}
static void select_photo(App& a, int idx);
static void draw_date_actions(App& a, Photo& p);
static void begin_rename(App& a, int64_t id);
static void apply_theme(App& a, int t);
namespace prefs {
static bool seg(const char* id, int* v, const std::vector<std::string>& labels);
}
static void trash_selected(App& a);
static void trash_selected_now(App& a);
static void copy_paths(App& a);

static std::string fmt_duration(double s) {
    if (s < 1) return "< 1 s";
    int t = int(s + 0.5);
    if (t < 60) return util::fmt("%d s", t);
    if (t < 3600) return util::fmt("%d min %02d s", t / 60, t % 60);
    return util::fmt("%d h %02d min", t / 3600, t / 60 % 60);
}

// Rough time for applying a plan: copying reads and writes every byte (~150 MB/s on a local disk); renames and
// moves on the same disk are instant; each metadata write runs exiv2 (~40 ms).
static double estimate_plan(const std::vector<PlanItem>& plan, const Config& c, int& n, int64_t& bytes) {
    n = 0;
    bytes = 0;
    double t = 0;
    for (auto& it : plan) {
        if (it.action == "duplicate" || !it.include || !it.result.empty()) continue;
        n++;
        bytes += it.size;
        if (it.action == "copy") t += double(it.size) / 150e6;
        t += c.write_mode == WRITE_DB_ONLY ? 0.005 : 0.04;
    }
    return t;
}
static int count_plan(const std::vector<PlanItem>& plan, bool review) {
    int n = 0;
    for (auto& it : plan) n += it.action != "duplicate" && it.include && it.result.empty() && (!review || it.needs_review);
    return n;
}

// Top of the window: the folder, one main button, and a status line. Everything else lives in the tabs.
static void request_thumb(App& a, const Photo& p) {
    if (a.thumb_id == p.id) return;
    a.thumb_id = p.id;
    a.thumb.unload();
    ThumbKey key{p.src_path, p.size, p.mtime};
    int orient = meta_orientation(MetaMap::from_json(p.meta_json));
    int side = thumb_side_for(p.src_path, a.cfg.thumb_side, a.cfg.vl_max_side);
    int64_t id = p.id;
    auto slot = a.thumb_slot;
    slot->want = id;
    std::thread([slot, key, orient, side, id] {
        Thumb t;
        t.id = id;
        std::string jpeg, err, tp = ensure_thumb(key, orient, side, err);  // cached: usually just a small file read
        if ((!tp.empty() && util::read_file(tp, jpeg)) || prepare_image(key.src_path, orient, side, jpeg, err)) {
            int n;
            unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(jpeg.data()), int(jpeg.size()), &t.w, &t.h, &n, 4);
            if (px) {
                t.rgba.assign(px, px + size_t(t.w) * t.h * 4);
                stbi_image_free(px);
            }
        }
        if (slot->want != id) return;  // the user moved on
        std::lock_guard<std::mutex> l(slot->mu);
        slot->result = std::move(t);
        slot->ready = true;
        glfwPostEmptyEvent();
    }).detach();
}

static void upload_thumb(App& a) {
    std::lock_guard<std::mutex> l(a.thumb_slot->mu);
    if (!a.thumb_slot->ready) return;
    a.thumb_slot->ready = false;
    Thumb& t = a.thumb_slot->result;
    if (t.id == a.thumb_id && !t.rgba.empty()) {
        a.thumb.upload(t.rgba.data(), t.w, t.h);
        a.thumb_shown_at = glfwGetTime();
    }
    else if (t.id == a.thumb_id) a.thumb_failed = t.id;
    t.rgba.clear();
}

static void viewer_load(App& a) {
    Viewer& v = a.viewer;
    if (v.idx < 0 || v.idx >= int(v.ids.size())) return;
    int64_t id = v.ids[size_t(v.idx)];
    if (v.loaded_id == id) return;
    v.loaded_id = id;
    v.shown_stage = 0;
    v.zoom = 1;
    v.pan = ImVec2(0, 0);
    v.dz.jump(1);
    v.dx.jump(0);
    v.dy.jump(0);
    a.viewer_tex.unload();
    if (!a.ui_db.load(id, v.p)) return;
    ThumbKey key{v.p.src_path, v.p.size, v.p.mtime};
    std::string dest = v.p.dest_path;
    int orient = meta_orientation(MetaMap::from_json(v.p.meta_json));
    int side = thumb_side_for(v.p.src_path, a.cfg.thumb_side, a.cfg.vl_max_side);
    auto slot = v.slot;
    slot->want = id;
    std::thread([slot, key, dest, orient, side, id] {
        bool shown = false;
        auto publish = [&](const std::string& jpeg, int stage) {
            int w, h, n;
            unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(jpeg.data()), int(jpeg.size()), &w, &h, &n, 4);
            if (!px) return;
            shown = true;
            if (slot->want == id) {
                std::lock_guard<std::mutex> l(slot->mu);
                slot->rgba.assign(px, px + size_t(w) * h * 4);
                slot->w = w;
                slot->h = h;
                slot->stage = stage;
                slot->id = id;
                slot->ready = true;
                glfwPostEmptyEvent();
            }
            stbi_image_free(px);
        };
        std::string jpeg, err, tp = ensure_thumb(key, orient, side, err);
        if (!tp.empty() && util::read_file(tp, jpeg)) publish(jpeg, 1);
        if (slot->want != id) return;
        std::string src = util::file_exists(key.src_path) ? key.src_path : dest;
        if (prepare_image(src, orient, 2048, jpeg, err)) publish(jpeg, 2);
        if (!shown && slot->want == id) {  // say why instead of leaving the viewer blank
            std::lock_guard<std::mutex> l(slot->mu);
            slot->failed_id = id;
            slot->error = !util::file_exists(src) ? "file not found: " + src : err.empty() ? "cannot decode " + src : err;
            glfwPostEmptyEvent();
        }
    }).detach();
}

static void open_viewer(App& a, std::vector<int64_t> ids, int idx) {
    if (ids.empty()) return;
    a.viewer.ids = std::move(ids);
    a.viewer.idx = std::clamp(idx, 0, int(a.viewer.ids.size()) - 1);
    a.viewer.loaded_id = 0;
    a.viewer.open = true;
    a.viewer.opened_frame = ImGui::GetFrameCount();
    viewer_load(a);
}

// Star / unstar a photo everywhere it is shown; stored by the worker.
static void set_favorite(App& a, int64_t id, bool on) {
    for (auto& r : a.rows)
        if (r.id == id) r.favorite = on;
    if (a.sel.id == id) a.sel.favorite = on;
    a.view.write([id, on](Db& db) { db.set_favorite(id, on); });
    a.catalog_gen++;  // "is:fav" and the favorite word are part of the search index
    a.rows_dirty = true;
}
static bool is_favorite(const App& a, int64_t id) {
    for (auto& r : a.rows)
        if (r.id == id) return r.favorite;
    for (auto& r : a.favorites)
        if (r.id == id) return true;
    return a.sel.id == id && a.sel.favorite;
}

// Starred tags as one-click filters. Clicking one searches tag:<name> (again: clears it).
static void draw_fav_tag_chips(App& a) {
    if (a.cfg.fav_tags.empty()) return;
    ImGui::TextDisabled("\xE2\x98\x85");
    for (auto& t : a.cfg.fav_tags) {
        ImGui::SameLine();
        std::string q = "tag:\"" + t + "\"";
        bool on = std::string(a.search) == q;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 0.6f));
        if (ImGui::SmallButton(t.c_str())) {
            snprintf(a.search, sizeof a.search, "%s", on ? "" : q.c_str());
            a.ask_go = a.regex = false;
            a.rows_dirty = true;
        }
        tip(tr("Show only photos with this tag (click again to clear); right-click to unstar"));
        if (on) ImGui::PopStyleColor();
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem(tr("Unstar this tag"))) {
                std::string keep = t;
                a.cfg.fav_tags.erase(std::remove(a.cfg.fav_tags.begin(), a.cfg.fav_tags.end(), keep), a.cfg.fav_tags.end());
                save_config(a.cfg, a.config_file);
                ImGui::EndPopup();
                break;
            }
            tip(tr("Remove this tag from the chips"));
            ImGui::EndPopup();
        }
    }
}

// A star button for a tag (in tag lists): starred tags become filter chips.
static void tag_star(App& a, const std::string& tag) {
    auto& f = a.cfg.fav_tags;
    bool on = std::find(f.begin(), f.end(), tag) != f.end();
    ImGui::PushID(tag.c_str());
    ImGui::PushStyleColor(ImGuiCol_Text, on ? g_star : g_text_dim);
    if (ImGui::SmallButton(on ? "\xE2\x98\x85" : "\xE2\x98\x86")) {
        if (on) f.erase(std::remove(f.begin(), f.end(), tag), f.end());
        else f.push_back(tag);
        save_config(a.cfg, a.config_file);
    }
    ImGui::PopStyleColor();
    tip(on ? tr("Unstar this tag") : tr("Star this tag: it becomes a one-click filter"));
    ImGui::PopID();
}

static void draw_viewer(App& a) {
    Viewer& v = a.viewer;
    if (!v.open) return;
    bool dialog = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    if (!dialog && v.idx >= 0 && v.idx < int(v.ids.size())) {
        if (ImGui::IsKeyPressed(ImGuiKey_F2, false)) begin_rename(a, v.ids[size_t(v.idx)]);
        bool ctrl = ImGui::GetIO().KeyCtrl;
        if ((ImGui::IsKeyPressed(ImGuiKey_Delete, false) || (!ctrl && ImGui::IsKeyPressed(ImGuiKey_D, false))) && !a.trash_busy) {
            int64_t id = v.ids[size_t(v.idx)];  // asks first; then the viewer shows the next photo
            a.multi = {id};
            a.selected = id;
            trash_selected(a);
        }
        if (!ctrl && ImGui::IsKeyPressed(ImGuiKey_C, false)) copy_paths(a);
        if (ImGui::IsKeyPressed(ImGuiKey_0, false) && !ImGui::GetIO().KeyCtrl) { v.zoom = 1; v.pan = ImVec2(0, 0); }
    }
    switch (dialog ? NK_NONE : nav_key(ImGui::GetFrameCount() != v.opened_frame)) {
        case NK_LEFT: case NK_UP: if (v.idx > 0) v.idx--; break;
        case NK_RIGHT: case NK_DOWN: case NK_SPACE: if (v.idx + 1 < int(v.ids.size())) v.idx++; break;
        case NK_ESC: v.open = false; return;
        case NK_FAV:
            if (v.idx >= 0 && v.idx < int(v.ids.size())) set_favorite(a, v.ids[size_t(v.idx)], !is_favorite(a, v.ids[size_t(v.idx)]));
            break;
        default: break;
    }
    viewer_load(a);
    {
        std::lock_guard<std::mutex> l(v.slot->mu);
        if (v.slot->ready && v.slot->id == v.loaded_id && v.slot->stage > v.shown_stage) {
            a.viewer_tex.upload(v.slot->rgba.data(), v.slot->w, v.slot->h);
            if (v.shown_stage == 0) v.shown_at = glfwGetTime();  // the sharper version replaces it in place, no fade
            v.shown_stage = v.slot->stage;
            v.slot->ready = false;
        }
    }
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    if (!dialog) ImGui::SetNextWindowFocus();
    ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);  // the theme's background, nearly opaque
    bg.w = 0.98f;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("##viewer", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    const Photo& p = v.p;
    if (ImGui::Button("<")) { if (v.idx > 0) v.idx--; }
    tip(tr("Previous photo (Left, k)"));
    ImGui::SameLine();
    if (ImGui::Button(">")) { if (v.idx + 1 < int(v.ids.size())) v.idx++; }
    tip(tr("Next photo (Right, j, Space)"));
    ImGui::SameLine();
    std::string name = p.src_path;
    name = rel_path(a.cfg, name);
    ImGui::Text("%d / %zu   %s   ·   %s   ·   %s", v.idx + 1, v.ids.size(), name.c_str(),
                p.date_value.empty() ? tr("undated") : util::parse_db_datetime(p.date_value + "|" + p.date_prec).pretty().c_str(),
                util::human_size(p.size).c_str());
    {
        ImGuiStyle& st = ImGui::GetStyle();
        float w = ImGui::CalcTextSize(tr("Rename...")).x + ImGui::CalcTextSize(tr("Open file")).x + ImGui::CalcTextSize(tr("Open folder")).x +
                  ImGui::CalcTextSize(tr("Close")).x + st.FramePadding.x * 8 + st.ItemSpacing.x * 3 + st.WindowPadding.x;
        ImGui::SameLine(ImGui::GetWindowWidth() - w);
    }
    if (ImGui::Button(tr("Rename..."))) begin_rename(a, p.id);
    tip(tr("Rename this file on disk (F2)  ·  c copies its path, d moves it to the Trash, f stars it"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Open file"))) util::open_path(p.src_path);
    tip(tr("Open the photo in the default viewer"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Open folder"))) util::open_path(util::dirname(p.src_path));
    tip(tr("Open the folder that holds this photo"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Close"))) v.open = false;
    tip(tr("Close the viewer (Esc or q)"));
    std::string info = p.location.empty() ? "" : p.location + "   ";
    json ct = json::parse(p.clip_tags.empty() ? "[]" : p.clip_tags, nullptr, false);
    if (ct.is_array())
        for (auto& t : ct)
            if (t.is_array() && !t.empty()) info += t[0].get<std::string>() + "  ";
    ImGui::TextDisabled("%s", info.c_str());
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= ImGui::GetTextLineHeightWithSpacing();
    if (a.viewer_tex.tex) {
        // Fit, then the wheel zooms around the pointer (up to 8x), drag pans, double-click toggles fit / 2x, 0 fits again.
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##img", avail);
        bool hov = ImGui::IsItemHovered();
        ImVec2 c0(p0.x + avail.x / 2, p0.y + avail.y / 2);
        float fit = std::min(avail.x / a.viewer_tex.w, avail.y / a.viewer_tex.h);
        ImGuiIO& io = ImGui::GetIO();
        auto zoom_at = [&](ImVec2 m, float z1) {
            z1 = std::clamp(z1, 1.0f, 8.0f);
            // from what is on screen now (an animation may be under way), so the point under the pointer stays there
            float s0 = fit * v.dz.value(), s1 = fit * z1;
            ImVec2 c(c0.x + v.dx.value(), c0.y + v.dy.value());
            ImVec2 u((m.x - c.x) / s0, (m.y - c.y) / s0);  // image point under the pointer stays there
            v.pan = ImVec2(m.x - u.x * s1 - c0.x, m.y - u.y * s1 - c0.y);
            v.zoom = z1;
        };
        if (hov && io.MouseWheel != 0) zoom_at(io.MousePos, v.zoom * std::pow(1.2f, io.MouseWheel));
        if (hov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) zoom_at(io.MousePos, v.zoom > 1.01f ? 1.0f : 2.0f);
        bool dragging = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1) && v.zoom > 1;
        if (dragging) {  // the picture follows the pointer 1:1, no easing
            v.pan.x += io.MouseDelta.x;
            v.pan.y += io.MouseDelta.y;
        }
        {
            ImVec2 tsz(a.viewer_tex.w * fit * v.zoom, a.viewer_tex.h * fit * v.zoom);
            if (v.zoom <= 1.0f) v.pan = ImVec2(0, 0);
            else {  // keep the picture on screen: no panning past its edges
                float mx = std::max(0.0f, (tsz.x - avail.x) / 2), my = std::max(0.0f, (tsz.y - avail.y) / 2);
                v.pan.x = std::clamp(v.pan.x, -mx, mx);
                v.pan.y = std::clamp(v.pan.y, -my, my);
            }
        }
        // Zoom (wheel, double-click) eases to the new size in 180 ms; a new step mid-way continues from where it is.
        v.dz.set(v.zoom, dragging);
        v.dx.set(v.pan.x, dragging);
        v.dy.set(v.pan.y, dragging);
        float sc = fit * v.dz.value();
        ImVec2 sz(a.viewer_tex.w * sc, a.viewer_tex.h * sc);
        ImVec2 q0(c0.x + v.dx.value() - sz.x / 2, c0.y + v.dy.value() - sz.y / 2);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->PushClipRect(p0, ImVec2(p0.x + avail.x, p0.y + avail.y), true);
        float fade = anim::progress(v.shown_at, 0.15f);  // the first picture fades in instead of popping
        dl->AddImage(ImTextureRef((ImTextureID)(intptr_t)a.viewer_tex.tex), q0, ImVec2(q0.x + sz.x, q0.y + sz.y), ImVec2(0, 0), ImVec2(1, 1),
                     IM_COL32(255, 255, 255, int(255 * fade)));
        dl->PopClipRect();
        if (v.zoom > 1.01f) {
            std::string zl = util::fmt("%.0f%%", sc * 100 * a.viewer_tex.w / std::max(1, v.p.width ? v.p.width : a.viewer_tex.w));
            ImVec2 ts = ImGui::CalcTextSize(zl.c_str());
            ImVec2 b0(p0.x + avail.x - ts.x - 22, p0.y + 8);
            dl->AddRectFilled(b0, ImVec2(b0.x + ts.x + 14, b0.y + ts.y + 8), ImGui::GetColorU32(ImGuiCol_PopupBg), 6.0f);
            dl->AddText(ImVec2(b0.x + 7, b0.y + 4), ImGui::GetColorU32(ImGuiCol_Text), zl.c_str());
        }
        if (hov) ImGui::SetMouseCursor(v.zoom > 1 ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_Arrow);
    } else {
        // Not there yet (or unreadable): say so in the middle instead of an empty screen.
        std::string msg, why;
        {
            std::lock_guard<std::mutex> l(v.slot->mu);
            if (v.slot->failed_id == v.loaded_id) why = v.slot->error;
        }
        msg = why.empty() ? tr("Loading...") : tr("This photo cannot be shown");
        ImVec2 c = ImGui::GetCursorPos();
        ImVec2 ts = ImGui::CalcTextSize(msg.c_str());
        ImGui::SetCursorPos(ImVec2(c.x + (avail.x - ts.x) / 2, c.y + avail.y / 2 - ts.y));
        if (why.empty()) ImGui::TextDisabled("%s", msg.c_str());
        else ImGui::TextColored(g_warn, "%s", msg.c_str());
        if (!why.empty()) {
            ImVec2 ws = ImGui::CalcTextSize(why.c_str());
            ImGui::SetCursorPosX(c.x + std::max(0.0f, (avail.x - ws.x) / 2));
            ImGui::TextDisabled("%s", why.c_str());
        }
        ImGui::SetCursorPos(c);
        ImGui::Dummy(avail);
    }
    ImGui::TextDisabled("%s", tr("Left/Right: previous/next  ·  wheel: zoom, drag: move, double-click or 0: fit  ·  F2: rename  ·  c: copy path  ·  d / Del: Trash  ·  f: star  ·  Esc: close"));
    ImGui::End();
    ImGui::PopStyleColor();
}

// Move files to the desktop Trash (restorable); without gio they go to <folder>/jev-duplicates instead of being
// deleted. Each (photo id, paths): the photo leaves the catalog once its first path is gone.
static void trash_files(App& a, const std::vector<std::pair<int64_t, std::vector<std::string>>>& items, const char* what);

static void trash_duplicates(App& a) {
    // Only exact, verified duplicates inside this folder; the kept copy stays.
    std::vector<std::pair<int64_t, std::vector<std::string>>> todo;
    for (auto& r : a.dup_rows)
        if (r.dup_of && path_under(r.src_path, view_scope(a.cfg))) todo.push_back({r.id, {r.src_path}});
    trash_files(a, todo, "duplicate copies");
}

static void trash_files(App& a, const std::vector<std::pair<int64_t, std::vector<std::string>>>& todo, const char* what_c) {
    std::string what = what_c;
    if (todo.empty() || a.trash_busy.exchange(true)) return;
    a.trash_done = a.trash_failed = 0;
    a.trash_what = what;
    std::vector<std::string> folders = a.cfg.folders;
    std::thread([&a, todo, folders, what] {
        bool gio = util::have_trash();
        std::vector<int64_t> gone;
        for (auto& [id, paths] : todo) {
          bool first_ok = false;
          for (size_t pi = 0; pi < paths.size(); pi++) {
            const std::string& path = paths[pi];
            bool ok = false;
            if (!util::file_exists(path)) ok = true;
            else if (gio) ok = util::trash(path);
            else {
                std::string folder = util::dirname(path);
                for (auto& f : folders)
                    if (path_under(path, f)) folder = f;
                std::string dest = folder + "/jev-duplicates/" + path.substr(folder.size() + 1);
                util::mkdirs(util::dirname(dest));
                ok = !util::file_exists(dest) && rename(path.c_str(), dest.c_str()) == 0;
            }
            if (pi == 0) first_ok = ok;
            if (ok) a.trash_done++;
            else a.trash_failed++;
            glfwPostEmptyEvent();
          }
          if (first_ok) gone.push_back(id);
        }
        a.log.add(0, util::fmt("moved %d %s to the %s (%d failed)", a.trash_done.load(), what.c_str(), gio ? "Trash" : "jev-duplicates folder",
                               a.trash_failed.load()));
        a.view.write([gone, &a](Db& db) {
            for (int64_t id : gone) db.remove_photo(id);
            a.catalog_gen++;
            a.trash_busy = false;
            glfwPostEmptyEvent();
        });
    }).detach();
}

// Home: what the analysis found and the next step for each part, with its button.
static void draw_trash_modal(App& a) {
    if (a.trash_confirm) {
        ImGui::OpenPopup("###trash");
        a.trash_confirm = false;
    }
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(std::min(1100.0f, vp->WorkSize.x - 40), std::min(640.0f, vp->WorkSize.y - 40)), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal((std::string(tr("Remove extra copies")) + "###trash").c_str(), nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
    bool gio = util::have_trash();
    std::map<int64_t, const DupRow*> by;
    for (auto& r : a.dup_rows) by[r.id] = &r;
    std::map<int64_t, int> group_no;
    int n = 0;
    int64_t bytes = 0;
    for (auto& r : a.dup_rows)
        if (r.dup_of && path_under(r.src_path, view_scope(a.cfg))) {
            n++;
            bytes += r.size;
            if (!group_no.count(r.dup_of)) group_no[r.dup_of] = int(group_no.size()) + 1;
        }
    ImGui::Text("%s: %zu %s  ·  %d %s  ·  %s  ·  %s %s", gio ? tr("Move to Trash") : tr("Move to jev-duplicates"), group_no.size(), tr("groups"), n,
                tr("files"), util::human_size(bytes).c_str(), tr("about"), fmt_duration(n * 0.05).c_str());
    ImGui::TextWrapped("%s", gio ? tr("They go to the desktop Trash, so they can be restored. The kept copy of every group stays where it is. Each file was confirmed identical byte by byte.")
                                 : tr("No desktop Trash is available: they are moved into jev-duplicates inside the folder instead. Nothing is deleted."));
    if (!a.cfg.verify_dupes) ImGui::TextColored(g_err, "%s", tr("Byte-verify is off: turn it on and check duplicates again first."));
    auto rel = [&](const std::string& p) { return rel_path(a.cfg, p); };
    if (ImGui::BeginTable("trashlist", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable,
                          ImVec2(0, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() - 8))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(tr("Group"), ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn(tr("Moves to Trash"), ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(tr("Kept copy (stays)"), ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(tr("Size"), ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableHeadersRow();
        std::vector<const DupRow*> rows;
        for (auto& r : a.dup_rows)
            if (r.dup_of && path_under(r.src_path, view_scope(a.cfg))) rows.push_back(&r);
        std::stable_sort(rows.begin(), rows.end(), [&](const DupRow* x, const DupRow* y) { return group_no[x->dup_of] < group_no[y->dup_of]; });
        for (auto* r : rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", group_no[r->dup_of]);
            ImGui::TableNextColumn();
            ImGui::TextColored(g_warn, "%s", rel(r->src_path).c_str());
            ImGui::TableNextColumn();
            auto k = by.find(r->dup_of);
            ImGui::TextColored(g_ok, "%s", k != by.end() ? rel(k->second->src_path).c_str() : "?");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(util::human_size(r->size).c_str());
        }
        ImGui::EndTable();
    }
    ImGui::BeginDisabled(!a.cfg.verify_dupes || n == 0);
    if (danger_button(util::fmt("%s (%d)", gio ? tr("Move to Trash") : tr("Move to jev-duplicates"), n).c_str(),
                      tr("Moves the listed extra copies. They can be restored from the Trash; the kept copy of each group stays."), ImVec2(220, 0))) {
        trash_duplicates(a);
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), ImVec2(120, 0))) ImGui::CloseCurrentPopup();
    tip(tr("Close without changing anything"));
    ImGui::EndPopup();
}

// A label whose text can be copied: click it (left or right) and it goes to the clipboard.
static void copy_text(App& a, const std::string& text, const std::string& shown = "", const ImVec4* col = nullptr, bool dim = false) {
    const std::string& t = shown.empty() ? text : shown;
    if (col) ImGui::PushStyleColor(ImGuiCol_Text, *col);
    else if (dim) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", t.c_str());
    if (col || dim) ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left) || ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        ImGui::SetClipboardText(text.c_str());
        a.toast = std::string(tr("Copied")) + ": " + (text.size() > 60 ? text.substr(0, 57) + "..." : text);
        a.toast_until = glfwGetTime() + 1.5;
    }
    tip(tr("Click to copy"));
}

// The generation metadata of an AI image: tool, model, LoRAs, seed and settings, the prompt and its keywords.
static void draw_gen_info(App& a, const Photo& p) {
    (void)a;
    GenInfo g = GenInfo::from_json(p.gen_json);
    if (!g.found()) return;
    if (!ImGui::CollapsingHeader(util::fmt("%s (%s)###gen", tr("AI generation"), g.tool.c_str()).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (!g.model.empty()) copy_text(a, g.model, std::string(tr("Model")) + ": " + g.model);
    for (auto& [l, w] : g.loras) ImGui::TextWrapped("LoRA: %s (%.2f)", l.c_str(), w);
    std::string settings;
    if (!g.seed.empty()) settings += "seed " + g.seed + "  ";
    if (g.steps) settings += util::fmt("%d steps  ", g.steps);
    if (g.cfg > 0) settings += util::fmt("cfg %.1f  ", g.cfg);
    if (!g.sampler.empty()) settings += g.sampler + (g.scheduler.empty() ? "" : "/" + g.scheduler) + "  ";
    if (!g.size.empty()) settings += g.size;
    if (!settings.empty()) copy_text(a, settings, "", nullptr, true);
    for (auto& src : g.sources) ImGui::TextDisabled("%s %s", tr("made from"), src.c_str());
    if (!g.prompt.empty()) {
        ImGui::TextUnformatted(tr("Prompt"));
        ImGui::SameLine();
        if (ImGui::SmallButton((std::string(tr("Copy")) + "##prompt").c_str())) ImGui::SetClipboardText(g.prompt.c_str());
        tip(tr("Copy the prompt to the clipboard"));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.85f, 1, 1));
        copy_text(a, g.prompt);
        ImGui::PopStyleColor();
    }
    if (!g.negative.empty() && ImGui::TreeNode(tr("Negative prompt"))) {
        copy_text(a, g.negative);
        ImGui::TreePop();
    }
    std::string kw;
    for (auto& [k, c] : parse_tag_list(p.gen_keywords)) kw += (kw.empty() ? "" : ", ") + k;
    if (!kw.empty()) copy_text(a, kw, std::string(tr("Prompt keywords")) + ": " + kw);
    ImGui::TextDisabled("%s %s (%s)", tr("found in"), g.field.c_str(), tr("read-only: never changed"));
}

// Remove a tag from one photo (your own list, or the generated tags), or block it everywhere.
static void remove_tag(App& a, Photo& p, const std::string& tag, bool block) {
    if (block) {
        set_tag_blocked(tag, true);
        a.toast = util::fmt("%s \"%s\"", tr("Blocked"), tag.c_str());
    } else if (!p.user_tags.empty()) {
        json keep = json::array();
        for (auto& [t, c] : parse_tag_list(p.user_tags))
            if (util::lower(t) != util::lower(tag)) keep.push_back(t);
        std::string js = keep.empty() ? "" : keep.dump();
        int64_t id = p.id;
        a.view.write([id, js](Db& db) { db.set_user_tags(id, js); });
        p.user_tags = js;
        a.toast = util::fmt("%s \"%s\"", tr("Removed"), tag.c_str());
    } else {
        Correction c;
        c.photo_id = p.id;
        c.field = "tag";
        c.action = "remove";
        c.current = tag;
        a.view.write([c](Db& db) {
            std::string err;
            apply_correction(db, c, err);
        });
        a.toast = util::fmt("%s \"%s\"", tr("Removed"), tag.c_str());
    }
    a.toast_until = glfwGetTime() + 2.5;
    a.sel_loaded = false;
    a.rows_dirty = true;
    a.catalog_gen++;
}

// ---- rename one photo's file, from anywhere a photo is shown (undoable from Activity)

// c: the selected photos' paths to the clipboard, one per line (paste them into a file manager, a chat, a terminal).
static void copy_paths(App& a) {
    std::string out;
    int n = 0;
    if (a.viewer.open) { out = a.viewer.p.src_path; n = 1; }
    else {
        std::set<int64_t> ids = a.multi;
        if (ids.empty() && a.selected) ids = {a.selected};
        for (auto& r : a.rows)
            if (ids.count(r.id)) { out += (out.empty() ? "" : "\n") + r.src_path; n++; }
    }
    if (!n) return;
    glfwSetClipboardString(a.win, out.c_str());
    a.toast = n == 1 ? std::string(tr("Copied: ")) + util::basename(out) : util::fmt("%d %s", n, tr("paths copied"));
    a.toast_until = glfwGetTime() + 2.5;
}

static void begin_rename(App& a, int64_t id) {
    Photo p;
    if (!id || !a.ui_db.load(id, p)) return;
    a.rename_id = id;
    std::string base = util::basename(p.src_path), ext = util::extension(base);
    a.rename_ext = ext;
    snprintf(a.rename_buf, sizeof a.rename_buf, "%s", base.substr(0, base.size() - ext.size()).c_str());
    a.rename_err.clear();
    a.rename_open = true;
}

static bool do_rename(App& a, std::string& err) {
    Photo p;
    if (!a.ui_db.load(a.rename_id, p)) { err = tr("This photo is no longer in the catalog."); return false; }
    std::string stem = util::trim(a.rename_buf);
    if (stem.empty()) { err = tr("The name is empty."); return false; }
    if (stem.find_first_of("/\\:*?\"<>|") != std::string::npos) { err = tr("A name cannot contain / \\ : * ? \" < > |"); return false; }
    std::string to = util::dirname(p.src_path) + "/" + stem + a.rename_ext;
    if (to == p.src_path) return true;
    if (util::file_exists(to)) { err = tr("Another file already has this name here."); return false; }
    if (!util::file_exists(p.src_path)) { err = tr("The file is not there (is its drive connected?)."); return false; }
    if (rename(p.src_path.c_str(), to.c_str()) != 0) { err = std::string(tr("Could not rename: ")) + strerror(errno); return false; }
    std::string side_from = sidecar_path(p.src_path), side_to = sidecar_path(to);
    if (util::file_exists(side_from) && !util::file_exists(side_to)) rename(side_from.c_str(), side_to.c_str());
    // the catalog, and a journal entry so Activity can undo it
    JournalEntry e;
    e.run = util::now_iso();
    e.kind = "rename";
    e.photo_id = p.id;
    e.src = p.src_path;
    e.dst = to;
    e.prev_dest = p.dest_path;
    bool renamed_dest = p.dest_path == p.src_path;  // an organized file renamed in place is also the copy
    int64_t id = p.id;
    a.view.write([id, to, e, renamed_dest](Db& db) {
        db.move_src(id, to);
        if (renamed_dest) db.clear_dest(id, to);
        db.journal_add(e);
    });
    a.log.add(0, "renamed " + p.src_path + " -> " + util::basename(to));
    a.toast = std::string(tr("Renamed to ")) + util::basename(to);
    a.toast_until = glfwGetTime() + 2.5;
    a.sel_loaded = false;
    a.thumb_id = 0;
    a.rows_dirty = true;
    a.catalog_gen++;
    return true;
}

static void draw_rename_modal(App& a) {
    if (a.rename_open) {
        ImGui::OpenPopup(tr("Rename###rename"));
        a.rename_open = false;
    }
    ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(tr("Rename###rename"), nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(a.rename_ext.c_str()).x - 12);
    bool go = ImGui::InputText("##rn", a.rename_buf, sizeof a.rename_buf, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
    ImGui::SameLine(0, 4);
    ImGui::TextDisabled("%s", a.rename_ext.c_str());
    if (!a.rename_err.empty()) ImGui::TextColored(kDanger, "%s", a.rename_err.c_str());
    else ImGui::TextDisabled("%s", tr("The file is renamed on disk; Activity can undo it."));
    if (primary_button(tr("Rename"), tr("Rename the file (Enter)")) || go) {
        if (do_rename(a, a.rename_err)) ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel")) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ---- Delete: straight to the Trash / Recycle Bin (restorable), then the next photo is selected
// Delete / d: asks first (Enter confirms, Esc keeps them); the confirmation then calls trash_selected_now.
static void trash_selected(App& a) {
    if (a.multi.empty() && a.selected) a.multi = {a.selected};
    if (a.multi.empty() || a.trash_busy || a.pipe->running()) return;
    a.photo_trash_confirm = true;
}

static void trash_selected_now(App& a) {
    if (a.multi.empty() || a.trash_busy) return;
    if (a.viewer.open) {  // the viewer goes on with the next photo
        auto& ids = a.viewer.ids;
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int64_t id) { return a.multi.count(id) > 0; }), ids.end());
        if (ids.empty()) a.viewer.open = false;
        else a.viewer.idx = std::min(a.viewer.idx, int(ids.size()) - 1);
        a.viewer.loaded_id = 0;
    }
    std::vector<std::pair<int64_t, std::vector<std::string>>> items;
    int64_t next = 0;
    bool after = false;
    for (auto& r : a.rows) {
        if (a.multi.count(r.id)) { items.push_back({r.id, {r.src_path}}); after = true; }
        else if (after && !next) next = r.id;
    }
    if (items.empty() && a.multi.count(a.viewer.p.id)) items.push_back({a.viewer.p.id, {a.viewer.p.src_path}});
    if (!next)
        for (auto it = a.rows.rbegin(); it != a.rows.rend(); ++it)
            if (!a.multi.count(it->id)) { next = it->id; break; }
    trash_files(a, items, "photos");
    a.multi.clear();
    a.selected = next;
    if (next) a.multi = {next};
    a.scroll_sel = true;
}

static void draw_detail(App& a) {
    if (!a.selected) return;
    if (!a.sel_loaded || a.sel.id != a.selected) {
        a.sel_loaded = a.ui_db.load(a.selected, a.sel);
        snprintf(a.manual_buf, sizeof a.manual_buf, "%s", a.sel.manual_date.substr(0, a.sel.manual_date.find('|')).c_str());
    }
    if (!a.sel_loaded) return;
    Photo& p = a.sel;
    request_thumb(a, p);
    upload_thumb(a);
    float avail = ImGui::GetContentRegionAvail().x;
    if (a.thumb.tex) {
        float s = std::min(avail / a.thumb.w, 300.0f / a.thumb.h);
        float fade = anim::progress(a.thumb_shown_at, 0.15f);  // arrives without a pop
        ImGui::ImageWithBg(ImTextureRef((ImTextureID)(intptr_t)a.thumb.tex), ImVec2(a.thumb.w * s, a.thumb.h * s), ImVec2(0, 0), ImVec2(1, 1),
                           ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, fade));
    } else {
        ImVec2 c = ImGui::GetCursorPos();
        ImGui::Dummy(ImVec2(avail, 60));
        ImGui::SetCursorPos(ImVec2(c.x + 8, c.y + 20));
        if (a.thumb_failed == p.id) ImGui::TextColored(g_warn, "%s", util::file_exists(p.src_path) ? tr("No preview: the file cannot be decoded") : tr("No preview: the file is not there any more"));
        else ImGui::TextDisabled("%s", tr("Loading..."));
        ImGui::SetCursorPos(ImVec2(c.x, c.y + 64));
    }
    copy_text(a, p.src_path);
    if (!p.dest_path.empty()) copy_text(a, p.dest_path, "-> " + p.dest_path, &g_ok);
    std::string open_target = p.dest_path.empty() ? p.src_path : p.dest_path;
    ImGui::TextDisabled("%s  ·  %d x %d  ·  #%lld", util::human_size(p.size).c_str(), p.width, p.height, (long long)p.id);
    if (p.dup_of) ImGui::TextColored(g_warn, "%s #%lld", tr("duplicate of"), (long long)p.dup_of);
    if (p.similar_to) ImGui::TextColored(g_info, "%s #%lld (dHash %d)", tr("similar to"), (long long)p.similar_to, p.similar_dist);
    ImGui::PushStyleColor(ImGuiCol_Text, p.favorite ? g_star : ImGui::GetStyleColorVec4(ImGuiCol_Text));
    if (ImGui::SmallButton(p.favorite ? "\xE2\x98\x85 Favorite" : "\xE2\x98\x86 Favorite")) set_favorite(a, p.id, !p.favorite);
    ImGui::PopStyleColor();
    tip(tr("f also stars the selected photo"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Find similar"))) {  // CLIP image-to-image: like:#id in the search box
    tip(tr("Photos that look like this one (CLIP); also like:#id in the search box"));
        snprintf(a.search, sizeof a.search, "like:#%lld", (long long)p.id);
        a.ask_go = a.regex = false;
        a.rows_dirty = true;
        if (!library_section(a.section)) a.start_tab = "photos";
    }
    tip(tr("Photos that look like this one (CLIP); also: like:#id or like:<file name> in the search box"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Rename..."))) begin_rename(a, p.id);
    tip(tr("Rename this file on disk (F2; undoable from Activity)"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Open file"))) util::open_path(open_target);
    tip(tr("Open the photo in the default viewer"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Open folder"))) util::open_path(util::dirname(open_target));
    tip(tr("Open the folder that holds this photo"));
    ImGui::Separator();

    json ev = json::parse(p.date_evidence.empty() ? "{}" : p.date_evidence, nullptr, false);
    copy_text(a, p.date_value, std::string(tr("Date")) + ": " + (p.date_value.empty() ? tr("undated") : p.date_value));
    ImGui::Text("%s: %s  ·  %s: %.2f  ·  %s: %s", tr("Precision"), p.date_prec.c_str(), tr("Conf"), p.date_conf, tr("Decider"),
                p.date_decider.c_str());
    copy_text(a, p.date_source, std::string(tr("Source")) + ": " + p.date_source);
    draw_date_actions(a, p);  // unsure: confirm it, or pick another candidate
    if (ev.is_object() && ev.contains("jev")) ImGui::TextDisabled("%s", ev["jev"].get<std::string>().c_str());
    // jev's part in this photo: its date question, its folder's place question, searches that judged it.
    for (auto& d : a.decisions) {
        bool mine = (d.kind == "date" && d.photo_id == p.id) || (d.kind == "place" && d.subject == util::dirname(p.src_path)) ||
                    (d.kind == "search" && d.photo_id == p.id);
        if (!mine) continue;
        ImVec4 col = d.changed ? g_warn : g_info;
        std::string what = d.kind == "search" ? util::fmt("%s \"%s\"", tr("search"), d.subject.c_str()) : tr(d.kind.c_str());
        ImGui::TextColored(col, "jev · %s: %s %s, jev %s -> %s%s", what.c_str(), d.kind == "search" ? "" : tr("rules"), d.rules_pick.c_str(),
                           d.jev_pick.substr(0, 60).c_str(), d.final_pick.c_str(), d.changed ? util::fmt(" (%s)", tr("changed")).c_str() : "");
    }

    if (ImGui::CollapsingHeader(tr("Date evidence"), ImGuiTreeNodeFlags_DefaultOpen) && ev.is_object() && ev.contains("candidates")) {
        std::vector<int> alts;
        if (ev.contains("alternatives")) alts = ev["alternatives"].get<std::vector<int>>();
        if (ImGui::BeginTable("ev", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH)) {
            ImGui::TableSetupColumn(tr("Candidate"), ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn(tr("Value"), ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn(tr("Weight"), ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn(tr("Support"), ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableHeadersRow();
            int i = 0;
            for (auto& c : ev["candidates"]) {
                ImGui::TableNextRow();
                bool winner = !alts.empty() && alts[0] == i;
                bool rejected = c.value("weight", 0.0) <= 0;
                ImVec4 col = winner ? g_ok : rejected ? g_text_dim : ImGui::GetStyleColorVec4(ImGuiCol_Text);
                ImGui::TableNextColumn();
                ImGui::TextColored(col, "%s%s", winner ? "\xE2\x96\xB6 " : "", c.value("src", "").c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(col, "%s", c.value("when", "").c_str());
                tip(c.value("raw", "").c_str());
                if (c.contains("notes"))
                    for (auto& n : c["notes"]) {
                        ImGui::PushStyleColor(ImGuiCol_Text, g_warn);
                        ImGui::TextWrapped("%s", n.get<std::string>().c_str());
                        ImGui::PopStyleColor();
                    }
                ImGui::TableNextColumn();
                ImGui::Text("%.2f/%.2f", c.value("weight", 0.0), c.value("base", 0.0));
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", c.value("support", 0.0));
                i++;
            }
            ImGui::EndTable();
        }
    }
    ImGui::SetNextItemWidth(220);
    ImGui::InputTextWithHint("##manual", "YYYY-MM-DD HH:MM:SS", a.manual_buf, sizeof a.manual_buf);
    tip(tr("Manual date (YYYY-MM-DD HH:MM:SS, YYYY-MM-DD or YYYY-MM)"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Set date"))) {
        util::Civil c = util::parse_exif_datetime(a.manual_buf);
        if (c.valid()) {
            std::string v = c.str() + "|" + util::precision_name(c.prec);
            int64_t id = p.id;
            a.view.write([id, v](Db& db) { db.set_manual_date(id, v); });
            a.sel.manual_date = v;
            a.toast = tr("Re-run Decide and Organize to apply.");
            a.toast_until = glfwGetTime() + 3;
        }
    }
    tip(tr("Use this date for the photo; it wins over everything else on the next run"));
    if (!p.manual_date.empty()) {
        ImGui::SameLine();
        if (ImGui::Button(tr("Clear manual date"))) {
            int64_t id = p.id;
            a.view.write([id](Db& db) { db.set_manual_date(id, ""); });
            a.sel.manual_date.clear();
        }
        tip(tr("Forget the date you typed and use the decided one again"));
    }

    ImGui::Separator();
    copy_text(a, p.location, std::string(tr("Location")) + ": " + (p.location.empty() ? "-" : p.location));
    ImGui::SameLine();
    ImGui::TextDisabled("(%s, %.2f)", p.location_source.c_str(), p.location_conf);
    if (p.has_gps) copy_text(a, util::fmt("%.6f, %.6f", p.gps_lat, p.gps_lon), util::fmt("GPS: %.6f, %.6f", p.gps_lat, p.gps_lon));
    if (ImGui::CollapsingHeader(tr("Location evidence"))) {
        DirPlace d = a.ui_db.get_dir(util::dirname(p.src_path));
        ImGui::TextWrapped("%s", d.evidence.c_str());
    }
    draw_gen_info(a, p);
    if (!p.clip_model.empty() && ImGui::CollapsingHeader(tr("Tags (CLIP)"), ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!p.clip_scene.empty()) ImGui::Text("%s: %s", tr("Scene"), p.clip_scene.c_str());
        std::string drop;
        bool block = false;
        for (auto& [t, conf] : effective_tag_list(p)) {  // corrections applied; star a tag to make it a filter chip
            tag_star(a, t);
            ImGui::SameLine();
            ImGui::PushID(t.c_str());
            ImGui::PushStyleColor(ImGuiCol_Text, g_text_dim);
            if (ImGui::SmallButton(ICON_X)) ImGui::OpenPopup("tagdel");
            ImGui::PopStyleColor();
            tip(tr("Remove this tag"));
            if (ImGui::BeginPopup("tagdel")) {
                if (ImGui::MenuItem(tr("Remove from this photo"))) drop = t;
                tip(tr("Only this photo loses it; recognition will not put it back here"));
                ImGui::PushStyleColor(ImGuiCol_Text, g_warn_red);
                if (ImGui::MenuItem(tr("Block everywhere (never use this tag)"))) { drop = t; block = true; }
                ImGui::PopStyleColor();
                tip(tr("Removed from every photo and never suggested or written into files again; unblock it in the tag list"));
                ImGui::EndPopup();
            }
            ImGui::PopID();
            ImGui::SameLine();
            copy_text(a, t, util::fmt("%s  %.0f%%", t.c_str(), conf * 100));
        }
        if (!drop.empty()) remove_tag(a, p, drop, block);
    }
    if (p.vision_status == "done" && ImGui::CollapsingHeader(tr("Vision result"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("%s %s", p.vision_status.empty() ? "-" : p.vision_status.c_str(), p.vision_error.c_str());
        if (!p.caption.empty()) ImGui::TextWrapped("%s: %s", tr("Caption"), p.caption.c_str());
        if (!p.scene.empty()) ImGui::TextWrapped("%s: %s", tr("Scene"), p.scene.c_str());
        auto list = [](const std::string& j) {
            std::string out;
            json a = json::parse(j.empty() ? "[]" : j, nullptr, false);
            if (a.is_array())
                for (auto& x : a) out += (out.empty() ? "" : ", ") + x.get<std::string>();
            return out;
        };
        if (!p.objects.empty()) ImGui::TextWrapped("%s: %s", tr("Objects"), list(p.objects).c_str());
        if (!p.tags.empty()) ImGui::TextWrapped("%s: %s", tr("Tags"), list(p.tags).c_str());
        if (!p.landmark.empty()) ImGui::TextWrapped("%s: %s", tr("Landmark"), p.landmark.c_str());
        if (!p.vision_text.empty()) ImGui::TextWrapped("%s: %s", tr("Text"), p.vision_text.c_str());
        if (p.vision_status == "done") ImGui::Text("%s: %d", tr("People"), p.people);
    }
    ImGui::TextDisabled("%s: %s %s", tr("Metadata"), p.meta_state.c_str(), p.write_error.c_str());
    if (ImGui::CollapsingHeader(tr("Original metadata (read-only)"))) {
        MetaMap m = MetaMap::from_json(p.meta_json);
        if (!m.error.empty()) ImGui::TextColored(g_warn_red, "%s", m.error.c_str());
        for (auto& [k, vs] : m.kv)
            for (auto& v : vs) ImGui::TextWrapped("%s = %s", k.c_str(), v.c_str());
    }
}

// A date button with a small calendar popup. Returns true when the value changed ("" = no limit).
static bool date_picker(const char* id, std::string& value, const char* empty_label) {
    bool changed = false;
    ImGui::PushID(id);
    std::string label = (value.empty() ? std::string(empty_label) : value) + "  \xE2\x96\xBE";
    if (ImGui::Button(label.c_str(), ImVec2(140, 0))) ImGui::OpenPopup("cal");
    tip(tr("Pick a date from the calendar"));
    if (ImGui::BeginPopup("cal")) {
        ImGuiStorage* st = ImGui::GetStateStorage();
        ImGuiID ky = ImGui::GetID("y"), km = ImGui::GetID("m");
        int y = st->GetInt(ky, 0), m = st->GetInt(km, 0);
        if (!y) {  // start on the current value, or this month
            util::Civil c = util::parse_exif_datetime(value.empty() ? util::now_iso() : value);
            y = c.valid() ? c.y : 2020;
            m = c.valid() ? c.mo : 1;
        }
        if (ImGui::SmallButton("<<")) y--;
        tip(tr("Previous year"));
        ImGui::SameLine();
        if (ImGui::SmallButton("<")) { if (--m < 1) { m = 12; y--; } }
        tip(tr("Previous month"));
        ImGui::SameLine();
        ImGui::Text("  %04d-%02d  ", y, m);
        ImGui::SameLine();
        if (ImGui::SmallButton(">")) { if (++m > 12) { m = 1; y++; } }
        tip(tr("Next month"));
        ImGui::SameLine();
        if (ImGui::SmallButton(">>")) y++;
        tip(tr("Next year"));
        static const char* wd[] = {"Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"};
        if (ImGui::BeginTable("days", 7, ImGuiTableFlags_SizingFixedSame)) {
            for (auto* d : wd) { ImGui::TableNextColumn(); ImGui::TextDisabled("%s", d); }
            int first = int((util::days_from_civil(y, m, 1) + 3) % 7);  // 1970-01-01 was a Thursday; Monday = 0
            if (first < 0) first += 7;
            for (int k = 0; k < first; k++) ImGui::TableNextColumn();
            for (int d = 1; util::valid_date(y, m, d); d++) {
                ImGui::TableNextColumn();
                std::string v = util::fmt("%04d-%02d-%02d", y, m, d);
                if (ImGui::Selectable(util::fmt("%2d", d).c_str(), v == value, 0, ImVec2(24, 0))) {
                    value = v;
                    changed = true;
                    ImGui::CloseCurrentPopup();
                }
            }
            ImGui::EndTable();
        }
        if (ImGui::SmallButton(tr("Any date"))) { value.clear(); changed = true; ImGui::CloseCurrentPopup(); }
        tip(tr("No limit on this side"));
        st->SetInt(ky, y);
        st->SetInt(km, m);
        ImGui::EndPopup();
    }
    ImGui::PopID();
    return changed;
}

// The photos as a sortable list (the library's other view is the grid).
static void draw_photo_table(App& a) {
    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable |
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit;
    bool srch = a.searching;
    tf |= ImGuiTableFlags_Sortable;
    // Separate table ids: browsing sorts by date by default, a search by relevance.
    if (ImGui::BeginTable(srch ? "photos_s" : "photos", 6, tf)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("\xE2\x98\x85", ImGuiTableColumnFlags_PreferSortDescending, 22);
        ImGui::TableSetupColumn(tr("Date"), srch ? 0 : ImGuiTableColumnFlags_DefaultSort, 150);
        ImGui::TableSetupColumn(srch ? tr("Why it matched") : tr("Tags"),
                                srch ? ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending : 0, srch ? 360 : 300);
        ImGui::TableSetupColumn(tr("Place"), 0, 90);
        ImGui::TableSetupColumn(tr("Size"), ImGuiTableColumnFlags_PreferSortDescending, 64);
        ImGui::TableSetupColumn(tr("File"), 0, 420);
        ImGui::TableHeadersRow();
        if (ImGuiTableSortSpecs* ss = ImGui::TableGetSortSpecs())
            if ((ss->SpecsDirty || a.sorted_gen != a.view_gen) && ss->SpecsCount > 0) {
                const ImGuiTableColumnSortSpecs& sp = ss->Specs[0];
                bool desc = sp.SortDirection == ImGuiSortDirection_Descending;
                int col = sp.ColumnIndex;
                std::stable_sort(a.rows.begin(), a.rows.end(), [&](const PhotoRow& x, const PhotoRow& y) {
                    int c = 0;
                    switch (col) {
                        case 0: c = int(x.favorite) - int(y.favorite); break;
                        case 1:  // undated always last
                            if (x.date_value.empty() != y.date_value.empty()) return y.date_value.empty();
                            c = x.date_value.compare(y.date_value);
                            break;
                        case 2: c = srch ? (x.score < y.score ? -1 : x.score > y.score) : x.tags.compare(y.tags); break;
                        case 3: c = x.location.compare(y.location); break;
                        case 4: c = x.size < y.size ? -1 : x.size > y.size; break;
                        default: c = x.src_path.compare(y.src_path); break;
                    }
                    return desc ? c > 0 : c < 0;
                });
                ss->SpecsDirty = false;
                a.sorted_gen = a.view_gen;
            }
        // Keyboard: j/k or arrows move, h/l or left/right change month, Enter opens the viewer.
        int sel_idx = -1;
        for (int i = 0; i < int(a.rows.size()); i++)
            if (a.rows[size_t(i)].id == a.selected) { sel_idx = i; break; }
        NavKey nk = nav_key(!a.viewer.open);
        if ((nk == NK_DOWN || nk == NK_UP) && !a.rows.empty()) {
            sel_idx = std::clamp(sel_idx < 0 ? 0 : sel_idx + (nk == NK_DOWN ? 1 : -1), 0, int(a.rows.size()) - 1);
            a.selected = a.rows[size_t(sel_idx)].id;
            a.scroll_sel = true;
        } else if (nk == NK_LEFT || nk == NK_RIGHT) {
            std::vector<std::string> ms = {""};
            for (auto& mi : a.months) ms.push_back(mi.month);
            int mi = int(std::find(ms.begin(), ms.end(), a.month) - ms.begin());
            mi = std::clamp(mi + (nk == NK_RIGHT ? 1 : -1), 0, int(ms.size()) - 1);
            if (ms[size_t(mi)] != a.month) {
                a.section = ms[size_t(mi)].empty() ? SEC_ALL : SEC_MONTH;
                a.month = ms[size_t(mi)];
                a.rows_dirty = true;
                a.selected = 0;
            }
        } else if ((nk == NK_ENTER || nk == NK_SPACE) && sel_idx >= 0) {
            std::vector<int64_t> ids;
            for (auto& r : a.rows) ids.push_back(r.id);
            open_viewer(a, ids, sel_idx);
        } else if (nk == NK_FAV && sel_idx >= 0) {
            set_favorite(a, a.rows[size_t(sel_idx)].id, !a.rows[size_t(sel_idx)].favorite);
        }
        if ((nk == NK_DOWN || nk == NK_UP) && sel_idx >= 0) { a.multi = {a.selected}; a.anchor = a.selected; }  // keys move a single selection
        ImGuiListClipper clip;
        clip.Begin(int(a.rows.size()));
        if (a.scroll_sel && sel_idx >= 0) clip.IncludeItemByIndex(sel_idx);
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                const PhotoRow& r = a.rows[size_t(i)];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID(int(r.id));
                ImGui::PushStyleColor(ImGuiCol_Text, r.favorite ? g_star : g_text_dim);
                if (ImGui::SmallButton(r.favorite ? "\xE2\x98\x85" : "\xE2\x98\x86")) set_favorite(a, r.id, !r.favorite);
                tip(tr("Star or unstar this photo (f)"));
                ImGui::PopStyleColor();
                ImGui::TableNextColumn();
                std::string d = r.date_value.empty() ? tr("undated") : util::parse_db_datetime(r.date_value + "|" + r.date_prec).pretty();
                // Uncertain dates are shown in amber; details in the side panel.
                ImGui::PushStyleColor(ImGuiCol_Text, r.needs_review ? g_warn : ImGui::GetStyleColorVec4(ImGuiCol_Text));
                if (ImGui::Selectable(d.c_str(), a.selected == r.id || a.multi.count(r.id), ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
                    select_photo(a, i);
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        std::vector<int64_t> ids;
                        for (auto& x : a.rows) ids.push_back(x.id);
                        open_viewer(a, ids, i);
                    }
                }
                if (ImGui::BeginPopupContextItem("rowmenu")) {
                    if (ImGui::MenuItem(r.favorite ? tr("Remove from favorites") : tr("Add to favorites"), "f")) set_favorite(a, r.id, !r.favorite);
                    if (ImGui::MenuItem(tr("Rename..."), "F2")) begin_rename(a, r.id);
                    tip(tr("Rename this file on disk (undoable from Activity)"));
                    if (ImGui::MenuItem(tr("Open folder"))) util::open_path(util::dirname(r.src_path));
                    if (ImGui::MenuItem(tr("Move to Trash"), "Del")) {
                        if (!a.multi.count(r.id)) a.multi = {r.id};
                        trash_selected(a);
                    }
                    ImGui::EndPopup();
                }
                if (a.scroll_sel && i == sel_idx) { ImGui::SetScrollHereY(0.5f); a.scroll_sel = false; }
                ImGui::PopStyleColor();
                ImGui::PopID();
                ImGui::TableNextColumn();
                if (srch) ImGui::TextUnformatted(r.why.c_str());
                else ImGui::TextUnformatted(r.tags.empty() ? r.scene.c_str() : r.tags.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.location.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(util::human_size(r.size).c_str());
                if (ImGui::IsItemHovered() && r.width) ImGui::SetTooltip("%d x %d px, %lld bytes", r.width, r.height, (long long)r.size);
                ImGui::TableNextColumn();
                std::string shown = r.src_path;
                shown = rel_path(a.cfg, shown);
                if (r.dup_of) ImGui::TextDisabled("%s  (%s)", shown.c_str(), tr("duplicate"));
                else ImGui::TextUnformatted(shown.c_str());
                if (ImGui::IsItemHovered() && !r.dest_path.empty()) ImGui::SetTooltip("-> %s", r.dest_path.c_str());
            }
        ImGui::EndTable();
    }
}

// Example of the chosen naming for the Review header and the Tags & EXIF tab.
static std::string example_name(const Config& c, const std::string& src_stem, const std::string& ext, const std::string& day = "20190512") {
    std::string prefix = c.name_style == NAME_KEEP_PREFIX ? name_prefix(src_stem) : "";
    return (prefix.empty() ? "" : prefix + c.name_sep) + day + c.name_sep + util::fmt("%0*d", c.sn_digits, 1) + ext;
}

// What the Review list does: one row of choices. Changing any of them rebuilds the list.
static void draw_plan_controls(App& a, bool meta_view) {
    Config& c = a.cfg;
    bool busy = a.pipe->running();
    bool changed = false;
    ImGui::BeginDisabled(busy);
    if (!meta_view) {
        // Organize path: what happens to the files themselves.
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(tr("Do"));
        ImGui::SameLine();
        if (c.file_op == OP_METADATA) c.file_op = OP_RENAME;  // metadata-only lives in Auto set meta
        static const int order[] = {OP_RENAME, OP_COPY, OP_MOVE};  // rename first: it is the default
        int sel = c.file_op == OP_COPY ? 1 : c.file_op == OP_MOVE ? 2 : 0;
        if (prefs::seg("fileop", &sel, {tr("Rename in place"), tr("Copy into month folders"), tr("Move into month folders")})) {
            c.file_op = order[sel];
            changed = true;
        }
        tip(c.file_op == OP_RENAME ? tr("Each photo gets a date name in the folder it is in (your files are renamed; Activity can undo it).")
            : c.file_op == OP_COPY ? tr("Each photo is copied into <folder>/jev-organized/<year>/<month> with a date name; your originals keep their names.")
                                   : tr("Each photo is moved into month folders with a date name; no second copy uses disk space (Activity can undo it)."));
        ImGui::SameLine(0, 24);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(tr("Names"));
        ImGui::SameLine();
        if (prefs::seg("names", &c.name_style, {tr("Date only"), tr("Keep original name")})) changed = true;
        tip(tr("Keep the original name: its date, time, serial number and copy markers are removed, then the date and serial are added. A long name is shortened to its key words."));
        // Example: a real photo from the list when there is one, else a typical camera name.
        std::string ex_from = "IMG_20190512_123456.jpg", ex_to;
        for (auto& it : a.plan)
            if ((it.action == "rename" || it.action == "copy" || it.action == "move") && !it.dest.empty()) {
                ex_from = util::basename(it.src);
                ex_to = util::basename(it.dest);  // exactly what the list will do
                break;
            }
        if (ex_to.empty()) {
            std::string stem = ex_from.substr(0, ex_from.size() - util::extension(ex_from).size());
            ex_to = example_name(c, stem, util::lower(util::extension(ex_from)));
        }
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", tr("Example:"));
        ImGui::SameLine();
        ImGui::TextUnformatted(ex_from.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("\xE2\x86\x92");
        ImGui::SameLine();
        ImGui::TextColored(g_accent, "%s", ex_to.c_str());
        ImGui::SameLine(0, 28);
        bool meta = c.write_mode != WRITE_DB_ONLY;
        if (ImGui::Checkbox(c.file_op == OP_COPY ? tr("Also write metadata into the copies") : tr("Also write metadata into the files"), &meta)) {
            c.write_mode = meta ? WRITE_EMBED : WRITE_DB_ONLY;
            changed = true;
        }
        tip(tr("Dates, tags, descriptions and places are added to the organized files where they have none. Existing values are never changed."));
    } else {
        // Metadata: what is written into the files (where they are now).
        bool sidecar = c.write_mode == WRITE_SIDECAR;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(tr("Write"));
        ImGui::SameLine();
        if (ImGui::RadioButton(tr("inside the file"), !sidecar)) { c.write_mode = WRITE_EMBED; changed = true; }
        tip(tr("Into the photo file itself (its own values are never changed; only missing fields are added)"));
        ImGui::SameLine();
        if (ImGui::RadioButton(tr("in a .xmp sidecar"), sidecar)) { c.write_mode = WRITE_SIDECAR; changed = true; }
        tip(tr("Into a small .xmp file next to the photo; the photo file itself is not touched"));
        ImGui::SameLine(0, 20);
        if (ImGui::Checkbox(tr("Replace tags written earlier"), &c.rewrite_tags)) changed = true;
        tip(tr("Keywords jev-photos added on an earlier run are updated to the current tags. Keywords the file had on its own are kept."));
        ImGui::SameLine(0, 20);
        if (ImGui::Checkbox(tr("Write a description"), &c.write_description)) changed = true;
        tip(tr("The AI prompt (for generated images) and 'Shows: <tags>' as the file's description"));
        ImGui::SameLine();
        const char* dp[] = {tr("only if empty"), tr("LLM + jev decide, else you"), tr("LLM decides"), tr("always ask me")};
        ImGui::SetNextItemWidth(220);
        bool llm_down = a.llm_ok == 0;
        if (ImGui::Combo("##descpol", &c.desc_policy, dp, DESC_COUNT)) changed = true;
        tip(llm_down ? tr("When a file already has a description. The LLM is down: those photos are left for you to decide.")
                     : tr("When a file already has a description: keep it, append ours or replace it (the old text is backed up). Placeholders like 'OLYMPUS DIGITAL CAMERA' are always replaced."));
    }
    ImGui::EndDisabled();
    if (changed) {
        save_config(c, a.config_file);
        if (!busy && !a.cfg.folders.empty()) start_run(a, ST_ORGANIZE, true, meta_view ? OP_METADATA : c.file_op);  // the new list right away
    }
}

static void draw_plan(App& a, bool meta_view) {
    bool busy = a.pipe->running();
    if (!busy && glfwGetTime() - a.plan_at > 1.0) {  // pick up include/exclude and apply results
        a.plan = a.pipe->plan();
        a.plan_at = glfwGetTime();
    }
    if (meta_view) {
        ImGui::TextWrapped("%s", tr("All in one go: missing dates, keywords (tags), descriptions and places are written into the files where they are. "
                                    "What a file already has stays; every write can be undone from Activity."));
        int am = a.cfg.auto_meta ? 1 : 0;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(tr("After each Analyze"));
        ImGui::SameLine();
        if (prefs::seg("autometa", &am, {tr("Ask me"), tr("Write automatically")})) {
            a.cfg.auto_meta = am == 1;
            save_config(a.cfg, a.config_file);
        }
        tip(tr("Write automatically: when Analyze finishes, this list is built and applied by itself (undoable from Activity)"));
        ImGui::Spacing();
    }
    draw_plan_controls(a, meta_view);
    ImGui::Separator();
    int want_op = meta_view ? OP_METADATA : a.cfg.file_op;
    // The pipeline holds one list: it belongs to this view only if it was built for it.
    bool mine = meta_view ? a.plan_op == OP_METADATA : a.plan_op != OP_METADATA;
    if (a.plan.empty() || !mine) {
        if (mine && a.pipe->progress.finished_runs > 0 && a.last_stages == ST_ORGANIZE)
            ImGui::TextWrapped("%s", meta_view ? tr("Nothing to write: every file already has what jev-photos would add.")
                                               : util::fmt(tr("Nothing to do: all %d photos are already organized."), a.stats.organized).c_str());
        else
            ImGui::TextWrapped("%s", meta_view ? tr("Build the list to see every date, keyword, description and place that would be added to the files. Nothing changes until you press Apply.")
                                               : tr("Build the list to see every copy, move or rename, one per row. Nothing changes until you press Apply."));
        ImGui::BeginDisabled(busy || a.cfg.folders.empty());
        if (primary_button(tr("Build the list"), tr("List every change that would be made, to review before anything happens")))
            start_run(a, ST_ORGANIZE, true, want_op);
        ImGui::SameLine();
        ImGui::BeginDisabled();
        ImGui::Button(util::fmt("%s (0)", tr("Apply all")).c_str());
        ImGui::EndDisabled();
        tip(tr("Build the list first: Apply makes exactly the changes listed"));
        ImGui::EndDisabled();
        return;
    }
    int n_place = 0, n_refile = 0, n_meta = 0, n_dup = 0, n_rev = 0, n_inc = 0, n_ok = 0, n_err = 0, n_decide = 0;
    for (auto& it : a.plan) {
        n_decide += it.desc_action == "decide" && it.include;
        if (it.action == "copy" || it.action == "move" || it.action == "rename") n_place++;
        else if (it.action == "refile") n_refile++;
        else if (it.action == "metadata") n_meta++;
        else n_dup++;
        if (it.action != "duplicate") {
            n_rev += it.needs_review;
            n_inc += it.include && it.result.empty();  // what Apply would still do
            n_ok += it.result == "ok";
            n_err += !it.result.empty() && it.result != "ok";
        }
    }
    int n_todo = 0;
    int64_t bytes = 0;
    double secs = estimate_plan(a.plan, a.cfg, n_todo, bytes);
    const char* opname = a.plan_op == OP_MOVE ? "Move and rename" : a.plan_op == OP_RENAME ? "Rename in place"
                       : a.plan_op == OP_METADATA ? "Add missing metadata" : "Copy and rename";
    ImGui::Text("%s:  %d %s  ·  %s  ·  %s %s", tr(opname), n_todo, tr("files"), util::human_size(bytes).c_str(), tr("about"), fmt_duration(secs).c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("   (%d %s · %d %s · %d %s · %d %s)", n_refile, tr("to re-file"), n_meta, tr("metadata updates"), n_dup,
                        tr("duplicates skipped"), n_rev, tr("need review"));
    if (n_ok || n_err) {
        ImGui::SameLine();
        ImGui::TextColored(n_err ? g_err : g_ok, "  ·  %d %s, %d %s", n_ok, tr("applied"), n_err, tr("failed"));
    }
    if (a.plan_op == OP_METADATA)
        ImGui::TextDisabled("%s", tr("Nothing is moved or renamed. Missing fields are added to each photo where it is now (its organized copy if it has one)."));
    else if (a.plan_op != OP_COPY)
        ImGui::TextColored(g_warn, "%s", tr("Your original files will be renamed/moved (no copies are kept)."));
    else
        ImGui::TextDisabled("%s %s", tr("Organized copies go to"),
                            a.cfg.output.empty() && a.cfg.folders.size() > 1 ? tr("a jev-organized folder inside each photo folder") : output_dir(a.cfg).c_str());

    ImGui::BeginDisabled(busy);
    if (ImGui::Button(tr("Include all"))) a.pipe->set_include_where([](const PlanItem&) { return true; }, true);
    tip(tr("Tick every row again"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Exclude needs-review"))) a.pipe->set_include_where([](const PlanItem& it) { return it.needs_review; }, false);
    tip(tr("Untick the photos whose date is a guess"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Refresh preview"))) start_run(a, ST_ORGANIZE, true, want_op);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Rebuild the plan; excluded items are left out and the serial numbers close up."));
    ImGui::SameLine();
    std::string apply_label = util::fmt("%s (%d)", tr("Apply all"), n_inc);
    if (n_inc == 0 && n_ok > 0) {  // everything applied
        ImGui::BeginDisabled();
        ImGui::Button(tr("All applied"));
        ImGui::EndDisabled();
        tip(tr("Every change in this list has been made. Analyze again, or change the choices above, for a new list."));
    } else if (apply_button(a.cfg, apply_label.c_str(), ImVec2(0, 0), a.plan_op) && n_inc > 0) {
        apply_plan(a);
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 20);
    ImGui::Checkbox(tr("Needs review only"), &a.plan_review_only);
    tip(tr("Show only rows whose date is a guess"));
    if (n_decide) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, g_warn);
        ImGui::Checkbox(util::fmt("%d %s", n_decide, tr("descriptions to decide")).c_str(), &a.plan_decide_only);
        ImGui::PopStyleColor();
        tip(tr("These photos already have a description. The LLM and jev did not agree on keeping, appending or replacing it, so you choose (until then it is left as it is)."));
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220);
    ImGui::InputTextWithHint("##pf", tr("Search"), a.plan_filter, sizeof a.plan_filter);
    tip(tr("Show only rows whose file, date or place contains this text"));

    std::vector<int> vis;
    std::string f = util::lower(util::trim(a.plan_filter));
    for (int i = 0; i < int(a.plan.size()); i++) {
        auto& it = a.plan[size_t(i)];
        if (a.plan_review_only && !it.needs_review) continue;
        if (a.plan_decide_only && it.desc_action != "decide") continue;
        if (!f.empty() && util::lower(it.src + " " + it.dest + " " + it.date + " " + it.location).find(f) == std::string::npos) continue;
        vis.push_back(i);
    }
    float detail_h = std::clamp(a.plan_detail_h, 80.0f, std::max(80.0f, ImGui::GetContentRegionAvail().y - 120));
    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable |
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit;
    auto rel = [&](const std::string& p) {
        return rel_path(a.cfg, p);
    };
    if (ImGui::BeginTable("plan", 8, tf, ImVec2(0, ImGui::GetContentRegionAvail().y - detail_h - 8))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("", 0, 24);
        ImGui::TableSetupColumn(tr("Action"), 0, 70);
        ImGui::TableSetupColumn(tr("From"), 0, 330);
        ImGui::TableSetupColumn(tr("To"), 0, 330);
        ImGui::TableSetupColumn(tr("Date"), 0, 120);
        ImGui::TableSetupColumn(tr("Size"), 0, 64);
        ImGui::TableSetupColumn(tr("Metadata"), 0, 70);
        ImGui::TableSetupColumn(tr("Status"), 0, 220);
        ImGui::TableHeadersRow();
        int vpos = -1;
        for (int r = 0; r < int(vis.size()); r++)
            if (vis[size_t(r)] == a.plan_sel) { vpos = r; break; }
        auto plan_ids = [&] {
            std::vector<int64_t> ids;
            for (int k : vis) ids.push_back(a.plan[size_t(k)].id);
            return ids;
        };
        NavKey nk = nav_key(!a.viewer.open);
        if (nk != NK_NONE && !vis.empty()) {
            if (nk == NK_DOWN || nk == NK_UP || nk == NK_LEFT || nk == NK_RIGHT) {
                int step = nk == NK_DOWN ? 1 : nk == NK_UP ? -1 : nk == NK_RIGHT ? 10 : -10;  // h/l page by 10
                vpos = std::clamp(vpos < 0 ? 0 : vpos + step, 0, int(vis.size()) - 1);
                a.plan_sel = vis[size_t(vpos)];
                a.scroll_sel = true;
            } else if (nk == NK_ENTER && vpos >= 0) {
                open_viewer(a, plan_ids(), vpos);
            } else if (nk == NK_SPACE && vpos >= 0 && !busy) {
                PlanItem& it = a.plan[size_t(a.plan_sel)];
                if (it.action != "duplicate") { it.include = !it.include; a.pipe->set_include(it.id, it.include); }
            }
        }
        ImGuiListClipper clip;
        clip.Begin(int(vis.size()));
        if (a.scroll_sel && vpos >= 0) clip.IncludeItemByIndex(vpos);
        while (clip.Step())
            for (int r = clip.DisplayStart; r < clip.DisplayEnd; r++) {
                int i = vis[size_t(r)];
                PlanItem& it = a.plan[size_t(i)];
                bool dup = it.action == "duplicate";
                ImGui::TableNextRow();
                ImGui::PushID(i);
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(dup || busy);
                bool inc = it.include;
                if (ImGui::Checkbox("##inc", &inc)) {
                    it.include = inc;
                    a.pipe->set_include(it.id, inc);
                }
                tip(tr("Include this row when applying (Space)"));
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImVec4 ac = dup ? g_text_dim : it.action == "refile" ? g_warn
                          : it.action == "move" ? g_warn_red : it.action == "metadata" ? g_info
                                                                                                             : g_ok;
                if (ImGui::Selectable("##row", a.plan_sel == i,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap | ImGuiSelectableFlags_AllowDoubleClick)) {
                    a.plan_sel = i;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, plan_ids(), r);
                }
                if (a.scroll_sel && r == vpos) { ImGui::SetScrollHereY(0.5f); a.scroll_sel = false; }
                ImGui::SameLine(0, 0);
                ImGui::TextColored(ac, "%s", tr(it.action.c_str()));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(rel(it.action == "refile" ? it.old_dest : it.src).c_str());
                ImGui::TableNextColumn();
                if (dup) ImGui::TextDisabled("%s", it.note.c_str());
                else if (!it.include) ImGui::TextDisabled("%s", tr("excluded"));
                else ImGui::TextUnformatted(rel(it.dest).c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(it.needs_review ? g_warn : ImGui::GetStyleColorVec4(ImGuiCol_Text), "%s", it.date.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(util::human_size(it.size).c_str());
                ImGui::TableNextColumn();
                if (!it.meta_cmds.empty()) ImGui::Text("+%d%s", it.meta_added, it.meta_removed.empty() ? "" : util::fmt(" -%zu", it.meta_removed.size()).c_str());
                if (it.desc_action == "decide") { ImGui::SameLine(); ImGui::TextColored(g_warn, "%s", tr("desc?")); }
                else if (it.desc_action == "append" || it.desc_action == "replace") { ImGui::SameLine(); ImGui::TextDisabled("%s", tr(("desc " + it.desc_action).c_str())); }
                ImGui::TableNextColumn();
                if (it.result == "ok") ImGui::TextColored(g_ok, "%s", tr("applied"));
                else if (!it.result.empty()) ImGui::TextColored(g_err, "%s", it.result.c_str());
                else if (!dup) ImGui::TextDisabled("%s", it.note.c_str());
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    hsplit("##split_plan", a.plan_detail_h, 80, 2000, true);
    ImGui::BeginChild("plan_detail", ImVec2(0, 0), ImGuiChildFlags_Borders);
    if (a.plan_sel >= 0 && a.plan_sel < int(a.plan.size())) {
        const PlanItem& it = a.plan[size_t(a.plan_sel)];
        ImGui::Text("%s  %s", tr(it.action.c_str()), it.src.c_str());
        if (!it.old_dest.empty()) ImGui::TextColored(g_warn, "%s -> %s", it.old_dest.c_str(), it.dest.c_str());
        else if (it.action != "duplicate") ImGui::TextColored(g_ok, "-> %s", it.dest.c_str());
        if (!it.desc_new.empty() && it.desc_action != "add" && !it.desc_action.empty()) {
            // An existing description: what it says, what ours says, who decided, and your override.
            ImGui::SeparatorText(tr("Description"));
            ImGui::TextDisabled("%s", tr("In the file:"));
            ImGui::SameLine();
            ImGui::TextWrapped("%s", it.desc_old.c_str());
            ImGui::TextDisabled("%s", tr("Ours:"));
            ImGui::SameLine();
            ImGui::TextWrapped("%s", it.desc_new.c_str());
            ImGui::TextColored(it.desc_action == "decide" ? g_warn : g_info, "%s: %s  (%s)  %s",
                               tr("Decision"), tr(it.desc_action == "decide" ? "you decide" : it.desc_action.c_str()), it.desc_by.c_str(), it.desc_why.c_str());
            ImGui::BeginDisabled(busy || it.desc_action == "update" || it.desc_by == "same");
            for (const char* act : {"keep", "append", "replace"}) {
                ImGui::SameLine();
                bool on = it.desc_action == act;
                bool repl = std::string(act) == "replace";
                if (on) ImGui::PushStyleColor(ImGuiCol_Button, repl ? ImVec4(kDanger.x, kDanger.y, kDanger.z, 0.75f) : ImVec4(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 0.6f));
                if (repl) ImGui::PushStyleColor(ImGuiCol_Text, g_warn_red);
                bool clicked = ImGui::SmallButton(tr(act));
                if (repl) ImGui::PopStyleColor();
                tip(std::string(act) == "keep" ? tr("Leave the file's description as it is")
                    : std::string(act) == "append" ? tr("Add ours after the file's description (separated by a dash)")
                                                   : tr("Replace the file's description with ours; the old text is kept in Xmp.jev.PreviousDescription"));
                if (clicked) {
                    a.pipe->set_desc_action(it.id, act);
                    a.plan[size_t(a.plan_sel)].desc_action = act;
                    a.plan[size_t(a.plan_sel)].desc_by = "you";
                }
                if (on) ImGui::PopStyleColor();
            }
            ImGui::EndDisabled();
            tip(tr("Replace keeps the old text in Xmp.jev.PreviousDescription."));
        }
        if (!it.meta_target.empty()) {
            ImGui::TextDisabled("%s %s  (%s)", tr("Metadata additions to"), it.meta_target.c_str(),
                                tr("existing values are never changed; Xmp.jev.* is provenance"));
            for (auto& c : it.meta_cmds) {
                bool ours = util::starts_with(c, "set Xmp.jev.");
                ImGui::TextColored(ours ? g_text_dim : g_info, "%s", c.c_str());
            }
            for (auto& k : it.meta_removed)
                ImGui::TextColored(g_warn_red, "- %s  (%s)", k.c_str(), tr("keyword jev-photos wrote earlier, replaced by the current tags"));
            if (it.meta_cmds.empty()) ImGui::TextDisabled("%s", tr("nothing to add"));
        }
    } else {
        ImGui::TextDisabled("%s", tr("Select a row to see exactly which metadata fields would be added."));
    }
    ImGui::EndChild();
}

// Tags & EXIF: what each photo is tagged with, what its file already carries, and what would be filled in.
static void draw_tags(App& a) {
    a.on_tags = true;
    if (!a.tags_built) {  // first visit: the worker is building the rows
        ImGui::TextDisabled("%s", tr("Loading..."));
        return;
    }
    bool busy = a.pipe->running();
    Config& c = a.cfg;
    const auto& rows = a.tag_rows;
    int n = int(rows.size()), untagged = 0, weak = 0, stale = 0, edited = 0, no_date = 0, no_kw = 0, no_desc = 0, no_place = 0, ai = 0;
    for (auto& r : rows) {
        ai += !r.gen_tool.empty();
        untagged += !r.tagged && !r.user_edited;
        weak += (r.tagged || r.user_edited) && r.best < kTagWordMin;
        stale += r.stale && !r.user_edited;
        edited += r.user_edited;
        no_date += !r.exif_date;
        no_kw += !r.keywords;
        no_desc += !r.description;
        no_place += !r.place;
    }
    // Summary + actions
    ImGui::Text("%d %s  ·  %d %s  ·  %d %s  ·  %d %s  ·  %d %s", n, tr("photos"), untagged, tr("without tags"), weak, tr("with only weak tags"),
                stale, tr("tagged with an older tag list"), edited, tr("edited by you"));
    ImGui::TextDisabled("%s  %d %s · %d %s · %d %s · %d %s  ·  %d %s", tr("In the files:"), no_date, tr("no date"), no_kw, tr("no keywords"), no_desc,
                        tr("no description"), no_place, tr("no place"), ai, tr("AI-generated with a prompt"));
    ImGui::BeginDisabled(busy);
    if (ImGui::Button(tr("Fill in missing tags"))) {
        RunOptions o = a.opts;
        o.stages = ST_TAG;
        o.scope = all_scope(c);
        o.clip_mode = o.tag_mode = 0;
        submit_task(a, "Fill in missing tags", effective(c), o, ST_TAG, false, -1, false);
    }
    tip(tr("Tag photos that have no tags yet, and refresh tags made with an older tag list"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Edit tag list..."))) a.tag_editor_open = true;
    tip(tr("The words CLIP chooses tags from: search, add, rename, move to another category or delete them. Saving re-tags the photos in seconds."));
    ImGui::SameLine(0, 30);
    if (primary_button(tr("Fill in missing metadata..."),
                       tr("Lists every date, keyword, description and place that would be added to the files (nothing is overwritten), under Write into files. Apply it there."))) {
        if (c.write_mode == WRITE_DB_ONLY) c.write_mode = WRITE_EMBED;
        save_config(c, a.config_file);
        start_run(a, ST_ORGANIZE, true, OP_METADATA);
        a.meta_sub = "write";
    }
    ImGui::EndDisabled();

    // Filter
    const char* filters[] = {tr("All photos"), tr("Without tags"), tr("Weak tags only"), tr("Older tag list"), tr("Edited by me"),
                             tr("No date in the file"), tr("No keywords in the file"), tr("No description in the file"), tr("No place in the file"), tr("AI-generated (prompt found)")};
    ImGui::SetNextItemWidth(230);
    ImGui::Combo("##tagfilter", &a.tags_filter, filters, IM_ARRAYSIZE(filters));
    tip(tr("Show only photos in this group"));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(260);
    ImGui::InputTextWithHint("##tagsearch", tr("Filter by file or tag"), a.tags_search, sizeof a.tags_search);
    tip(tr("Show only photos whose file name or tags contain this text"));
    std::string f = util::lower(util::trim(a.tags_search));
    std::vector<int> vis;
    for (int i = 0; i < n; i++) {
        const TagRow& r = rows[size_t(i)];
        bool ok = true;
        switch (a.tags_filter) {
            case 1: ok = !r.tagged && !r.user_edited; break;
            case 2: ok = (r.tagged || r.user_edited) && r.best < kTagWordMin; break;
            case 3: ok = r.stale && !r.user_edited; break;
            case 4: ok = r.user_edited; break;
            case 5: ok = !r.exif_date; break;
            case 6: ok = !r.keywords; break;
            case 7: ok = !r.description; break;
            case 8: ok = !r.place; break;
            case 9: ok = !r.gen_tool.empty(); break;
        }
        if (ok && !f.empty()) {
            std::string hay = util::lower(r.src_path);
            for (auto& t : r.tags) hay += " " + util::lower(t.first);
            for (auto& t : r.prompt_tags) hay += " " + t;
            ok = hay.find(f) != std::string::npos;
        }
        if (ok) vis.push_back(i);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%zu / %d", vis.size(), n);

    if (a.tags_detail_w <= 0) a.tags_detail_w = std::max(380.0f, ImGui::GetContentRegionAvail().x * 0.34f);
    float tavail = ImGui::GetContentRegionAvail().x;
    a.tags_detail_w = std::clamp(a.tags_detail_w, 280.0f, std::max(280.0f, tavail - 300));
    ImGui::BeginChild("tagtable", ImVec2(tavail - a.tags_detail_w - 8, 0));
    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable |
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Sortable;
    static int sort_col = 0;
    static bool sort_desc = false;
    if (ImGui::BeginTable("tags", 6, tf)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(tr("File"), ImGuiTableColumnFlags_DefaultSort, 240);
        ImGui::TableSetupColumn(tr("Tags"), 0, 300);
        ImGui::TableSetupColumn(tr("Best"), ImGuiTableColumnFlags_PreferSortDescending, 50);
        ImGui::TableSetupColumn(tr("Date"), 0, 100);
        ImGui::TableSetupColumn(tr("In the file"), 0, 260);
        ImGui::TableSetupColumn(tr("Place"), 0, 90);
        ImGui::TableHeadersRow();
        if (ImGuiTableSortSpecs* ss = ImGui::TableGetSortSpecs())
            if (ss->SpecsCount > 0) {
                sort_col = ss->Specs[0].ColumnIndex;
                sort_desc = ss->Specs[0].SortDirection == ImGuiSortDirection_Descending;
                ss->SpecsDirty = false;
            }
        std::stable_sort(vis.begin(), vis.end(), [&](int x, int y) {
            const TagRow &p = rows[size_t(x)], &q = rows[size_t(y)];
            int cmp = 0;
            auto tagstr = [](const TagRow& r) { return r.tags.empty() ? std::string("~") : r.tags[0].first; };
            auto file_flags = [](const TagRow& r) { return int(r.exif_date) + r.keywords + r.description + r.place; };
            switch (sort_col) {
                case 1: cmp = tagstr(p).compare(tagstr(q)); break;
                case 2: cmp = p.best < q.best ? -1 : p.best > q.best; break;
                case 3: cmp = p.date_value.compare(q.date_value); break;
                case 4: cmp = file_flags(p) - file_flags(q); break;
                case 5: cmp = p.location.compare(q.location); break;
                default: cmp = p.src_path.compare(q.src_path); break;
            }
            return sort_desc ? cmp > 0 : cmp < 0;
        });
        int sel_pos = -1;
        for (int k = 0; k < int(vis.size()); k++)
            if (rows[size_t(vis[size_t(k)])].id == a.tags_sel) { sel_pos = k; break; }
        NavKey nk = nav_key(!a.viewer.open && !ImGui::GetIO().WantTextInput);
        if ((nk == NK_DOWN || nk == NK_UP || nk == NK_LEFT || nk == NK_RIGHT) && !vis.empty()) {
            int step = nk == NK_DOWN ? 1 : nk == NK_UP ? -1 : nk == NK_RIGHT ? 10 : -10;
            sel_pos = std::clamp(sel_pos < 0 ? 0 : sel_pos + step, 0, int(vis.size()) - 1);
            a.tags_sel = rows[size_t(vis[size_t(sel_pos)])].id;
            a.scroll_sel = true;
        }
        auto ids = [&] {
            std::vector<int64_t> v;
            for (int k : vis) v.push_back(rows[size_t(k)].id);
            return v;
        };
        if ((nk == NK_ENTER || nk == NK_SPACE) && sel_pos >= 0) open_viewer(a, ids(), sel_pos);
        ImGuiListClipper clip;
        clip.Begin(int(vis.size()));
        if (a.scroll_sel && sel_pos >= 0) clip.IncludeItemByIndex(sel_pos);
        while (clip.Step())
            for (int k = clip.DisplayStart; k < clip.DisplayEnd; k++) {
                const TagRow& r = rows[size_t(vis[size_t(k)])];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID(int(r.id));
                std::string shown = rel_path(c, r.src_path);
                if (ImGui::Selectable(shown.c_str(), a.tags_sel == r.id, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
                    a.tags_sel = r.id;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, ids(), k);
                }
                if (a.scroll_sel && k == sel_pos) { ImGui::SetScrollHereY(0.5f); a.scroll_sel = false; }
                ImGui::PopID();
                ImGui::TableNextColumn();
                // Confident tags bright, guesses dimmed; the user's own tags in the accent colour.
                bool first = true;
                for (auto& [t, conf] : r.tags) {
                    if (!first) { ImGui::SameLine(0, 0); ImGui::TextDisabled(", "); ImGui::SameLine(0, 0); }
                    first = false;
                    if (r.user_edited) ImGui::TextColored(ImVec4(c.accent[0], c.accent[1], c.accent[2], 1), "%s", t.c_str());
                    else if (conf >= kTagWordMin) ImGui::TextUnformatted(t.c_str());
                    else ImGui::TextDisabled("%s?", t.c_str());
                }
                // Keywords from the generation prompt, after the picture's own tags.
                for (size_t q = 0; q < r.prompt_tags.size() && q < 8; q++) {
                    if (!first) { ImGui::SameLine(0, 0); ImGui::TextDisabled(", "); ImGui::SameLine(0, 0); }
                    first = false;
                    ImGui::TextColored(g_info, "%s", r.prompt_tags[q].c_str());
                }
                if (r.tags.empty() && r.prompt_tags.empty()) ImGui::TextDisabled("%s", r.tagged ? tr("(nothing clear)") : tr("(not tagged)"));
                if (r.stale && !r.user_edited) { ImGui::SameLine(); ImGui::TextColored(g_warn, "*"); }
                ImGui::TableNextColumn();
                if (r.user_edited) ImGui::TextDisabled("%s", tr("mine"));
                else if (!r.tags.empty()) ImGui::Text("%.0f%%", r.best * 100);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.date_value.substr(0, 10).c_str());
                ImGui::TableNextColumn();
                auto flag = [](bool on, const char* label) {
                    ImGui::SameLine(0, 6);
                    if (on) ImGui::TextColored(g_ok, "%s", label);
                    else ImGui::TextDisabled("%s", label);
                };
                ImGui::Dummy(ImVec2(0, 0));
                flag(r.exif_date, tr("date"));
                flag(r.keywords, tr("keywords"));
                flag(r.description, tr("desc"));
                flag(r.place, tr("place"));
                if (!r.gen_tool.empty()) { ImGui::SameLine(0, 6); ImGui::TextColored(g_info, "%s", r.gen_tool.c_str()); }
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.location.c_str());
            }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    vsplit("##split_tags", a.tags_detail_w, 280, 2000, true);
    ImGui::BeginChild("tagdetail", ImVec2(0, 0), ImGuiChildFlags_Borders);
    const TagRow* sel = nullptr;
    for (auto& r : rows)
        if (r.id == a.tags_sel) sel = &r;
    if (!sel) {
        ImGui::TextWrapped("%s", tr("Select a photo to see its tags with their confidence, edit them, and see exactly what would be added to its file."));
        ImGui::Spacing();
        ImGui::TextDisabled("%s", tr("Bright tags are confident and are searchable and written to files; dimmed tags with ? are guesses and are not. Blue: keywords from the AI generation prompt. * = tagged with an older tag list."));
    } else {
        // Load the photo (thumbnail + metadata preview) the same way the Photos detail does.
        if (a.selected != sel->id) { a.selected = sel->id; a.sel_loaded = false; }
        if (!a.sel_loaded || a.sel.id != a.selected) a.sel_loaded = a.ui_db.load(a.selected, a.sel);
        if (a.sel_loaded) {
            Photo& p = a.sel;
            request_thumb(a, p);
            upload_thumb(a);
            float avail = ImGui::GetContentRegionAvail().x;
            if (a.thumb.tex && a.thumb_id == p.id) {
                float sc = std::min(avail / a.thumb.w, 240.0f / a.thumb.h);
                ImGui::Image(ImTextureRef((ImTextureID)(intptr_t)a.thumb.tex), ImVec2(a.thumb.w * sc, a.thumb.h * sc));
            }
            copy_text(a, p.src_path, util::basename(p.src_path));
            if (c.name_style == NAME_KEEP_PREFIX)
                ImGui::TextDisabled("%s %s", tr("would be named"),
                                    example_name(c, util::stem(p.src_path), "." + util::lower(p.ext),
                                                 p.date_value.size() >= 10 ? p.date_value.substr(0, 4) + p.date_value.substr(5, 2) + p.date_value.substr(8, 2) : "00000000")
                                        .c_str());
            ImGui::SeparatorText(tr("Tags"));
            for (auto& [t, conf] : sel->tags) {
                tag_star(a, t);
                ImGui::SameLine();
                ImGui::ProgressBar(sel->user_edited ? 1.0f : conf, ImVec2(120, 0), sel->user_edited ? tr("mine") : util::fmt("%.0f%%", conf * 100).c_str());
                ImGui::SameLine();
                if (sel->user_edited || conf >= kTagWordMin) ImGui::TextUnformatted(t.c_str());
                else ImGui::TextDisabled("%s  (%s)", t.c_str(), tr("guess: not searchable, not written"));
            }
            if (!p.clip_scene.empty()) ImGui::TextDisabled("%s: %s", tr("Scene"), p.clip_scene.c_str());
            draw_gen_info(a, p);
            // Edit: comma-separated; saving replaces the automatic tags for this photo.
            if (a.tags_edit_id != p.id) {
                a.tags_edit_id = p.id;
                std::string cur;
                for (auto& t : effective_tags(p)) cur += (cur.empty() ? "" : ", ") + t;
                snprintf(a.tags_edit, sizeof a.tags_edit, "%s", cur.c_str());
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            bool enter = ImGui::InputTextWithHint("##tagedit", tr("tags, separated by commas"), a.tags_edit, sizeof a.tags_edit, ImGuiInputTextFlags_EnterReturnsTrue);
            tip(tr("Your own tags for this photo, separated by commas; Enter saves"));
            if (ImGui::Button(tr("Save my tags")) || enter) {
                json list = json::array();
                for (auto& t : util::split(a.tags_edit, ','))
                    if (!util::trim(t).empty()) list.push_back(util::trim(t));
                std::string js = list.dump();
                int64_t id = p.id;
                a.view.write([id, js](Db& db) { db.set_user_tags(id, js); });
                a.sel_loaded = false;
                a.rows_dirty = true;
                a.catalog_gen++;
                a.meta_preview_id = 0;
            }
            tip(tr("Use these tags for this photo instead of the automatic ones"));
            if (!p.user_tags.empty()) {
                ImGui::SameLine();
                if (ImGui::Button(tr("Back to automatic tags"))) {
                    int64_t id = p.id;
                    a.view.write([id](Db& db) { db.set_user_tags(id, ""); });
                    a.sel_loaded = false;
                    a.tags_edit_id = 0;
                    a.rows_dirty = true;
                    a.catalog_gen++;
                    a.meta_preview_id = 0;
                }
                tip(tr("Forget your own tags for this photo and use CLIP's again"));
            }
            // What the file has, and what filling in would add (computed off the UI thread).
            ImGui::SeparatorText(util::fmt("%s %s", tr("Metadata in the original"), util::basename(p.src_path).c_str()).c_str());
            if (a.meta_preview_id != p.id) {
                a.meta_preview_id = p.id;
                auto slot = a.meta_preview;
                Config cc = c;
                Photo pc = p;
                {
                    std::lock_guard<std::mutex> l(slot->mu);
                    slot->ready = false;
                    slot->id = p.id;
                }
                std::thread([slot, cc, pc] {
                    std::string file;
                    WriteResult r = preview_metadata(cc, pc, file);
                    std::lock_guard<std::mutex> l(slot->mu);
                    if (slot->id != pc.id) return;
                    slot->r = r;
                    slot->file = file;
                    slot->ready = true;
                    glfwPostEmptyEvent();
                }).detach();
            }
            MetaMap m = MetaMap::from_json(p.meta_json);
            auto show = [&](const char* label, std::initializer_list<const char*> keys) {
                std::string v;
                for (auto* k : keys)
                    if (m.has(k)) { v = m.get(k); break; }
                if (v.size() > 60) v = v.substr(0, 60) + "…";
                if (v.empty()) ImGui::TextDisabled("%-12s -", label);
                else copy_text(a, v, util::fmt("%-12s %s", label, v.c_str()));
            };
            show(tr("Date"), {"Exif.Photo.DateTimeOriginal", "Xmp.exif.DateTimeOriginal", "Xmp.photoshop.DateCreated"});
            show(tr("Keywords"), {"Xmp.dc.subject", "Iptc.Application2.Keywords"});
            show(tr("Description"), {"Xmp.dc.description", "Exif.Image.ImageDescription"});
            show(tr("Place"), {"Xmp.iptc.Location", "Xmp.photoshop.City", "Iptc.Application2.City"});
            show(tr("Camera"), {"Exif.Image.Model"});
            {
                std::lock_guard<std::mutex> l(a.meta_preview->mu);
                if (!a.meta_preview->ready || a.meta_preview->id != p.id) {
                    ImGui::TextDisabled("%s", tr("checking..."));
                } else {
                    const WriteResult& w = a.meta_preview->r;
                    ImGui::TextDisabled("%s %s", tr("Filling in would change"), util::basename(a.meta_preview->file).c_str());
                    int shown_n = 0;
                    std::set<std::string> kept;
                    for (auto& k : w.kept) kept.insert(k);
                    for (auto& cmd : w.commands) {
                        if (util::starts_with(cmd, "reg ") || util::starts_with(cmd, "set Xmp.jev.") || util::starts_with(cmd, "del ")) continue;
                        if (util::starts_with(cmd, "set Xmp.dc.subject ")) {  // a kept keyword put back is not a change
                            std::string k = cmd.substr(19);
                            if (util::starts_with(k, "XmpBag ")) k = k.substr(7);
                            if (kept.count(k)) continue;
                        }
                        ImGui::TextColored(ImVec4(0.6f, 0.85f, 1, 1), "+ %s", cmd.substr(4).c_str());
                        shown_n++;
                    }
                    for (auto& k : w.removed) ImGui::TextColored(g_warn_red, "- Xmp.dc.subject %s", k.c_str());
                    if (!shown_n && w.removed.empty()) ImGui::TextDisabled("%s", tr("nothing: the file already has it all"));
                }
            }
        }
    }
    ImGui::EndChild();
}

// One thumbnail for a grid cell: the texture when ready, else start decoding it (at most 4 at a time).
static const Texture* grid_thumb(App& a, const PhotoRow& r) {
    auto g = a.grid;
    g->used[r.id] = g->frame;
    auto it = g->tex.find(r.id);
    if (it != g->tex.end()) return &it->second;
    std::lock_guard<std::mutex> l(g->mu);
    if (g->pending.count(r.id) || g->failed.count(r.id) || g->pending.size() >= 4) return nullptr;
    g->pending.insert(r.id);
    ThumbKey key{r.src_path, r.size, r.mtime};  // the scanned mtime: a cached thumbnail still shows when the drive is away
    struct stat st{};
    if (stat(r.src_path.c_str(), &st) == 0) key.mtime = ST_MTIME(st);
    int side = thumb_side_for(r.src_path, a.cfg.thumb_side, a.cfg.vl_max_side);
    int64_t id = r.id;
    std::thread([g, key, side, id] {
        Thumb t;
        t.id = id;
        std::string jpeg, err, tp = ensure_thumb(key, 1, side, err);
        if (!tp.empty() && util::read_file(tp, jpeg)) {
            int n;
            unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(jpeg.data()), int(jpeg.size()), &t.w, &t.h, &n, 4);
            if (px) {
                t.rgba.assign(px, px + size_t(t.w) * t.h * 4);
                stbi_image_free(px);
            }
        }
        std::lock_guard<std::mutex> l(g->mu);
        g->done.push_back({id, std::move(t)});
        glfwPostEmptyEvent();
    }).detach();
    return nullptr;
}

static void grid_upload(App& a) {
    auto g = a.grid;
    g->frame++;
    std::lock_guard<std::mutex> l(g->mu);
    for (int k = 0; k < 6 && !g->done.empty(); k++) {  // a few per frame keeps scrolling smooth
        auto d = std::move(g->done.front());
        g->done.pop_front();
        g->pending.erase(d.id);
        if (!d.t.rgba.empty()) {
            g->tex[d.id].upload(d.t.rgba.data(), d.t.w, d.t.h);
            g->shown_at[d.id] = glfwGetTime();
        }
        else g->failed.insert(d.id);
    }
    if (g->tex.size() > 400) {  // drop what has not been on screen for a while
        for (auto it = g->tex.begin(); it != g->tex.end();) {
            if (g->frame - g->used[it->first] > 600) { it->second.unload(); it = g->tex.erase(it); }
            else ++it;
        }
    }
}

// Click on a photo row or cell: plain = only this one, Ctrl = add/remove it, Shift = everything from the anchor.
static void select_photo(App& a, int idx) {
    const auto& rows = a.rows;
    if (idx < 0 || idx >= int(rows.size())) return;
    int64_t id = rows[size_t(idx)].id;
    ImGuiIO& io = ImGui::GetIO();
    if (io.KeyShift && a.anchor) {
        int ai = -1;
        for (int i = 0; i < int(rows.size()); i++)
            if (rows[size_t(i)].id == a.anchor) ai = i;
        if (ai >= 0) {
            if (!io.KeyCtrl) a.multi.clear();
            for (int i = std::min(ai, idx); i <= std::max(ai, idx); i++) a.multi.insert(rows[size_t(i)].id);
        }
    } else if (io.KeyCtrl) {
        if (!a.multi.count(id)) a.multi.insert(id);
        else a.multi.erase(id);
        a.anchor = id;
    } else {
        a.multi = {id};
        a.anchor = id;
    }
    a.selected = id;
}

// Move the selected photos to the Trash: a confirmation with what moves, how much, and (optionally) their organized copies.
static void draw_photo_trash(App& a) {
    if (a.photo_trash_confirm) {
        ImGui::OpenPopup(tr("Move to Trash###ptrash"));
        a.photo_trash_confirm = false;
    }
    if (!ImGui::BeginPopupModal(tr("Move to Trash###ptrash"), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    std::vector<const PhotoRow*> sel;
    for (auto& r : a.rows)
        if (a.multi.count(r.id)) sel.push_back(&r);
    static PhotoRow viewer_row;  // a photo opened in the viewer from outside the current list
    if (sel.empty() && a.viewer.open && a.multi.count(a.viewer.p.id)) {
        viewer_row = PhotoRow();
        viewer_row.id = a.viewer.p.id;
        viewer_row.src_path = a.viewer.p.src_path;
        viewer_row.dest_path = a.viewer.p.dest_path;
        viewer_row.size = a.viewer.p.size;
        sel.push_back(&viewer_row);
    }
    int64_t bytes = 0;
    int copies = 0;
    for (auto* r : sel) { bytes += r->size; copies += !r->dest_path.empty() && r->dest_path != r->src_path; }
    ImGui::Text("%zu %s · %s", sel.size(), tr("photos"), util::human_size(bytes).c_str());
    for (size_t i = 0; i < sel.size() && i < 8; i++) ImGui::BulletText("%s", rel_path(a.cfg, sel[i]->src_path).c_str());
    if (sel.size() > 8) ImGui::TextDisabled("... %s %zu %s", tr("and"), sel.size() - 8, tr("more"));
    bool gio = util::have_trash();
    ImGui::TextWrapped("%s", gio ? tr("They go to the desktop Trash and can be restored from there. They also leave the catalog.")
                                 : tr("No desktop Trash here: they are moved into a jev-duplicates folder inside their photo folder."));
    if (copies) {
        ImGui::Checkbox(util::fmt("%s (%d)", tr("Also move their organized copies"), copies).c_str(), &a.trash_with_copies);
        tip(tr("The copies jev-photos made in jev-organized go to the Trash too"));
    }
    ImGui::BeginDisabled(sel.empty() || a.trash_busy);
    bool enter = ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false);
    if (danger_button(util::fmt("%s (%zu)  \xE2\x8F\x8E", tr("Move to Trash"), sel.size()).c_str(), tr("Moves these files to the Trash (restorable). Enter confirms, Esc keeps them."), ImVec2(220, 0)) ||
        (enter && !sel.empty() && !a.trash_busy)) {
        std::vector<std::pair<int64_t, std::vector<std::string>>> items;
        for (auto* r : sel) {
            std::vector<std::string> paths{r->src_path};
            if (a.trash_with_copies && !r->dest_path.empty() && r->dest_path != r->src_path) paths.push_back(r->dest_path);
            items.push_back({r->id, paths});
        }
        if (a.trash_with_copies || !util::have_trash()) {
            trash_files(a, items, "photos");
            a.multi.clear();
            a.selected = 0;
        } else {
            trash_selected_now(a);  // also selects the next photo
        }
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    tip(tr("Keep the files"));
    ImGui::EndPopup();
}

// Photos as a thumbnail grid (the Grid view of the Photos tab): same rows, same selection and keys as the table.
static void draw_photo_grid(App& a) {
    grid_upload(a);
    const auto& rows = a.rows;
    if (rows.empty()) {
        ImGui::Dummy(ImVec2(0, 20));
        ImGui::TextWrapped("%s", a.fav_only ? tr("No favorites here yet. Star a photo with f, or with the star in the list view.") : tr("No photos."));
        return;
    }
    // Ctrl+wheel over the grid changes the thumbnail size (the list keeps scrolling with the plain wheel)
    ImGuiIO& gio = ImGui::GetIO();
    if (gio.KeyCtrl && gio.MouseWheel != 0 && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
        a.grid_cell = std::clamp(a.grid_cell * std::pow(1.12f, gio.MouseWheel), 96.0f, 420.0f);
        a.scroll_sel = true;  // keep the selected photo in view
    }
    const float cell = std::round(a.grid_cell), img = cell - 18;
    int cols = std::max(1, int(ImGui::GetContentRegionAvail().x / cell));
    int sel = -1;
    for (int i = 0; i < int(rows.size()); i++)
        if (rows[size_t(i)].id == a.selected) sel = i;
    auto ids = [&] {
        std::vector<int64_t> v;
        for (auto& r : rows) v.push_back(r.id);
        return v;
    };
    NavKey nk = nav_key(!a.viewer.open);
    if (nk != NK_NONE) {
        int step = nk == NK_RIGHT ? 1 : nk == NK_LEFT ? -1 : nk == NK_DOWN ? cols : nk == NK_UP ? -cols : 0;
        if (step) { sel = std::clamp(sel < 0 ? 0 : sel + step, 0, int(rows.size()) - 1); a.selected = rows[size_t(sel)].id; a.scroll_sel = true; }
        else if ((nk == NK_ENTER || nk == NK_SPACE) && sel >= 0) open_viewer(a, ids(), sel);
        else if (nk == NK_FAV && sel >= 0) set_favorite(a, rows[size_t(sel)].id, !rows[size_t(sel)].favorite);
    }
    // Lines: a month heading whenever the month changes (only while the photos are in date order), then rows of cells.
    struct GLine { int start = 0, n = 0; std::string head; float y = 0, h = 0; };
    std::vector<GLine> lines;
    auto month_of = [](const PhotoRow& r) { return r.date_value.size() >= 7 ? r.date_value.substr(0, 7) : std::string(); };
    bool by_date = !a.searching && a.section != SEC_MONTH;
    if (by_date) {  // monotonic months (either direction), undated anywhere
        int dir = 0;
        std::string prev;
        for (auto& r : rows) {
            std::string m = month_of(r);
            if (m.empty()) continue;
            if (!prev.empty() && m != prev) {
                int d = m > prev ? 1 : -1;
                if (dir && d != dir) { by_date = false; break; }
                dir = d;
            }
            prev = m;
        }
    }
    const float row_h = img + ImGui::GetStyle().ItemSpacing.y * 2 + ImGui::GetTextLineHeight() + 4;
    const float head_h = ImGui::GetTextLineHeight() * 1.5f + 14;
    float y = 0;
    int sel_line = -1;
    for (int i = 0; i < int(rows.size());) {
        std::string m = by_date ? month_of(rows[size_t(i)]) : "";
        if (by_date && (i == 0 || m != month_of(rows[size_t(i - 1)]))) {
            GLine h;
            static const char* mnames[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
            int mo = m.size() == 7 ? atoi(m.substr(5).c_str()) : 0;
            h.head = m.empty() ? std::string(tr("Undated")) : mo >= 1 && mo <= 12 ? util::fmt("%s %s", tr(mnames[mo - 1]), m.substr(0, 4).c_str()) : m;
            int cnt = 0;
            for (int k = i; k < int(rows.size()) && month_of(rows[size_t(k)]) == m; k++) cnt++;
            h.head += util::fmt("  ·  %d", cnt);
            h.y = y;
            h.h = head_h;
            y += head_h;
            lines.push_back(h);
        }
        int n = 0;
        while (n < cols && i + n < int(rows.size()) && (!by_date || month_of(rows[size_t(i + n)]) == m)) n++;
        GLine l;
        l.start = i;
        l.n = n;
        l.y = y;
        l.h = row_h;
        if (sel >= i && sel < i + n) sel_line = int(lines.size());
        lines.push_back(l);
        y += row_h;
        i += n;
    }
    float y0 = ImGui::GetCursorPosY(), x0 = ImGui::GetCursorPosX();
    float top = ImGui::GetScrollY() - y0, bottom = top + ImGui::GetWindowHeight();
    if (a.scroll_sel && sel_line >= 0) {
        const GLine& l = lines[size_t(sel_line)];
        if (l.y < top || l.y + l.h > bottom) ImGui::SetScrollY(y0 + l.y - (bottom - top - l.h) / 2);
        a.scroll_sel = false;
    }
    // Only the lines in view are drawn (binary search for the first one).
    int first = int(std::lower_bound(lines.begin(), lines.end(), top, [](const GLine& l, float t) { return l.y + l.h < t; }) - lines.begin());
    for (int li = first; li < int(lines.size()) && lines[size_t(li)].y < bottom; li++) {
        const GLine& l = lines[size_t(li)];
        if (!l.head.empty()) {
            ImGui::SetCursorPos(ImVec2(x0 + 2, y0 + l.y + 8));
            ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.15f);
            ImGui::TextUnformatted(l.head.c_str());
            ImGui::PopFont();
            continue;
        }
        for (int c = 0; c < l.n; c++) {
            int i = l.start + c;
            const PhotoRow& r = rows[size_t(i)];
            ImGui::SetCursorPos(ImVec2(x0 + c * cell, y0 + l.y));
            ImGui::BeginGroup();
            ImGui::PushID(int(r.id));
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            if (ImGui::Selectable("##cell", a.selected == r.id || a.multi.count(r.id), ImGuiSelectableFlags_AllowDoubleClick, ImVec2(cell - 8, img))) {
                select_photo(a, i);
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, ids(), i);
            }
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem(r.favorite ? tr("Remove from favorites") : tr("Add to favorites"))) set_favorite(a, r.id, !r.favorite);
                tip(tr("Star or unstar this photo (f)"));
                if (ImGui::MenuItem(tr("Find similar"))) {
                    a.tokens.clear();
                    snprintf(a.search, sizeof a.search, "like:#%lld", (long long)r.id);
                    a.rows_dirty = true;
                }
                tip(tr("Photos that look like this one (CLIP); also like:#id in the search box"));
                if (ImGui::MenuItem(tr("Open folder"))) util::open_path(util::dirname(r.dest_path.empty() ? r.src_path : r.dest_path));
                tip(tr("Open the folder that holds this photo"));
                if (ImGui::MenuItem(tr("Rename..."), "F2")) begin_rename(a, r.id);
                tip(tr("Rename this file on disk (undoable from Activity)"));
                if (ImGui::MenuItem(tr("Move to Trash"), "Del")) {
                    if (!a.multi.count(r.id)) a.multi = {r.id};
                    trash_selected(a);
                }
                tip(tr("Move this photo (or the selected ones) to the Trash; restorable"));
                ImGui::EndPopup();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s\n%s", rel_path(a.cfg, r.src_path).c_str(), r.tags.c_str());
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (const Texture* t = grid_thumb(a, r); t && t->tex) {
                float s2 = std::min(img / t->w, img / t->h);
                ImVec2 sz(t->w * s2, t->h * s2);
                ImVec2 q0(p0.x + (cell - 8 - sz.x) / 2, p0.y + (img - sz.y) / 2);
                auto sa = a.grid->shown_at.find(r.id);
                float fade = sa == a.grid->shown_at.end() ? 1.0f : anim::progress(sa->second, 0.15f);
                dl->AddImageRounded(ImTextureRef((ImTextureID)(intptr_t)t->tex), q0, ImVec2(q0.x + sz.x, q0.y + sz.y), ImVec2(0, 0), ImVec2(1, 1),
                                    IM_COL32(255, 255, 255, int(255 * fade)), 4.0f);
            }
            else {  // loading, or no preview possible
                bool failed = false;
                {
                    std::lock_guard<std::mutex> l(a.grid->mu);
                    failed = a.grid->failed.count(r.id) > 0;
                }
                ImVec2 q0(p0.x + 4, p0.y + 4), q1(p0.x + cell - 12, p0.y + img - 4);
                dl->AddRectFilled(q0, q1, ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);
                const char* ic = failed ? ICON_ALERT_TRIANGLE : ICON_PHOTO;
                ImVec2 ts = ImGui::CalcTextSize(ic);
                dl->AddText(ImVec2((q0.x + q1.x - ts.x) / 2, (q0.y + q1.y - ts.y) / 2), ImGui::GetColorU32(failed ? g_warn : g_text_dim), ic);
            }
            if (r.favorite) dl->AddText(ImVec2(p0.x + 6, p0.y + 4), ImGui::GetColorU32(g_star), ICON_STAR);
            if (r.needs_review) dl->AddText(ImVec2(p0.x + cell - 30, p0.y + 4), ImGui::GetColorU32(g_warn), ICON_CALENDAR_QUESTION);
            std::string name = util::basename(r.dest_path.empty() ? r.src_path : r.dest_path);
            if (ImGui::CalcTextSize(name.c_str()).x > cell - 14) {  // cut to the tile width (UTF-8 safe), with an ellipsis
                while (name.size() > 1 && ImGui::CalcTextSize((name + "…").c_str()).x > cell - 14) {
                    size_t k = name.size() - 1;
                    while (k > 0 && (name[k] & 0xC0) == 0x80) k--;
                    name.resize(k);
                }
                name += "…";
            }
            ImGui::TextDisabled("%s", name.c_str());
            ImGui::PopID();
            ImGui::EndGroup();
        }
    }
    ImGui::SetCursorPos(ImVec2(x0, y0 + y));
    ImGui::Dummy(ImVec2(1, 1));
}

// Corrections: specific fixes of what jev-photos generated (tags, names it gave), with how sure we are. Tick and apply;
// the file's own metadata is never a target.
static void draw_corrections(App& a) {
    bool busy = a.pipe->running();
    auto& cs = a.corrections;
    for (auto& c : cs)
        if (!a.corr_sel.count(c.id)) a.corr_sel[c.id] = c.confidence >= 0.7;  // sure enough: ticked by default
    int ticked = 0;
    for (auto& c : cs) ticked += a.corr_sel[c.id];
    ImGui::TextWrapped("%s", tr("Re-checks what jev-photos generated itself (CLIP tags, names it gave) against the prompt, the camera EXIF, the file name and CLIP. The file's own metadata (prompt, camera, lens, dates, descriptions others wrote) is only evidence and is never changed. Proposals at 70% or more are ticked."));
    ImGui::BeginDisabled(busy);
    if (ImGui::Button(tr("Check again..."))) a.corr_confirm = true;
    tip(a.llm_ok == 1 ? tr("Re-check every photo; asks first, because AI images are read by the LLM on the GPU server")
                      : tr("The LLM is down: the check would use the rules only"));    tip(tr("Re-check every photo, not only new or changed ones"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Tick >= 70%"))) for (auto& c : cs) a.corr_sel[c.id] = c.confidence >= 0.7;
    tip(tr("Tick only the proposals jev-photos is fairly sure about"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Tick all"))) for (auto& c : cs) a.corr_sel[c.id] = true;
    tip(tr("Tick every proposal"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Untick all"))) for (auto& c : cs) a.corr_sel[c.id] = false;
    tip(tr("Untick every proposal"));
    ImGui::SameLine(0, 20);
    auto act = [&](const char* status) {
        std::vector<Correction> todo;
        for (auto& c : cs)
            if (a.corr_sel[c.id]) todo.push_back(c);
        std::string st = status;
        a.view.write([todo, st](Db& db) {
            for (auto& c : todo) {
                std::string err;
                if (st == "applied" && !apply_correction(db, c, err)) { db.set_correction_status(c.id, "failed: " + err); continue; }
                db.set_correction_status(c.id, st);
            }
        });
        for (auto& c : todo) a.corr_sel.erase(c.id);
        cs.erase(std::remove_if(cs.begin(), cs.end(), [&](const Correction& c) {
                     for (auto& t : todo) if (t.id == c.id) return true;
                     return false;
                 }), cs.end());
        a.catalog_gen++;
        a.rows_dirty = true;
        a.sel_loaded = false;
        if (st == "applied") {
            a.toast = util::fmt(tr("%zu corrections applied. Files keep their old keywords until you apply \"Add metadata only\"."), todo.size());
            a.toast_until = glfwGetTime() + 5;
        }
    };
    if (primary_button(util::fmt("%s (%d)", tr("Apply ticked"), ticked).c_str(),
                       tr("Changes the catalog's tags (and renames files jev-photos named, for name proposals). Files get the new keywords when you next write metadata.")) &&
        ticked)
        act("applied");
    ImGui::SameLine();
    if (ImGui::Button(util::fmt("%s (%d)", tr("Reject ticked"), ticked).c_str()) && ticked) act("rejected");
    tip(tr("Rejected proposals are not made again for the same photo."));
    ImGui::EndDisabled();
    if (cs.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("%s", tr("Nothing to correct. Analyze checks new and changed photos; \"Check again\" re-checks all of them."));
        return;
    }
    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV |
                         ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Sortable;
    if (ImGui::BeginTable("corr", 6, tf)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_NoSort, 24);
        ImGui::TableSetupColumn(tr("Sure"), ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending, 80);
        ImGui::TableSetupColumn(tr("Photo"), 0, 260);
        ImGui::TableSetupColumn(tr("Change"), 0, 230);
        ImGui::TableSetupColumn(tr("Why"), 0, 520);
        ImGui::TableSetupColumn(tr("Evidence"), 0, 150);
        ImGui::TableHeadersRow();
        if (ImGuiTableSortSpecs* ss = ImGui::TableGetSortSpecs())
            if (ss->SpecsDirty && ss->SpecsCount > 0) {
                int col = ss->Specs[0].ColumnIndex;
                bool desc = ss->Specs[0].SortDirection == ImGuiSortDirection_Descending;
                std::stable_sort(cs.begin(), cs.end(), [&](const Correction& x, const Correction& y) {
                    int c = col == 1 ? (x.confidence < y.confidence ? -1 : x.confidence > y.confidence) : col == 2 ? x.path.compare(y.path)
                          : col == 3 ? (x.current + x.proposed).compare(y.current + y.proposed) : x.source.compare(y.source);
                    return desc ? c > 0 : c < 0;
                });
                ss->SpecsDirty = false;
            }
        std::vector<int64_t> photo_ids;
        for (auto& c : cs) photo_ids.push_back(c.photo_id);
        ImGuiListClipper clip;
        clip.Begin(int(cs.size()));
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                const Correction& c = cs[size_t(i)];
                ImGui::TableNextRow();
                ImGui::PushID(int(c.id));
                ImGui::TableNextColumn();
                bool on = a.corr_sel[c.id];
                if (ImGui::Checkbox("##t", &on)) a.corr_sel[c.id] = on;
                tip(tr("Apply or reject this proposal with the buttons above"));
                ImGui::TableNextColumn();
                ImGui::ProgressBar(float(c.confidence), ImVec2(-1, 0), util::fmt("%.0f%%", c.confidence * 100).c_str());
                ImGui::TableNextColumn();
                std::string shown = util::basename(c.path);
                if (ImGui::Selectable(shown.c_str(), a.corr_focus == c.id, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap |
                                                                                  ImGuiSelectableFlags_AllowDoubleClick)) {
                    a.corr_focus = c.id;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, photo_ids, i);
                }
                tip(c.path.c_str());
                ImGui::TableNextColumn();
                if (c.field == "name") ImGui::TextColored(g_warn, "%s -> %s", c.current.c_str(), c.proposed.c_str());
                else if (c.action == "remove") ImGui::TextColored(g_warn_red, "- %s %s", tr("tag"), c.current.c_str());
                else ImGui::TextColored(g_ok, "+ %s %s", tr("tag"), c.proposed.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(c.reason.c_str());
                tip(c.reason.c_str());
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", c.source.c_str());
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
}

static void draw_dupes(App& a) {
    if (a.cfg.folders.empty()) {
        ImGui::TextWrapped("%s", tr("Choose a folder of photos to begin."));
        return;
    }
    bool busy = a.pipe->running();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button(tr("Check duplicates"))) {
        start_run(a, ST_SCAN | ST_DUPES);
        a.dups_dirty = true;
    }
    tip(tr("Look for identical files (and, if ticked, similar photos) in the folders"));
    ImGui::SameLine();
    ImGui::Checkbox(tr("Byte-verify"), &a.cfg.verify_dupes);
    tip(tr("Compare files byte by byte before marking them identical."));
    ImGui::SameLine();
    ImGui::Checkbox(tr("Also find similar photos"), &a.cfg.similar_check);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Resized, re-saved or re-compressed copies (perceptual hash). Decodes each image once; results are cached."));
    if (a.cfg.similar_check) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderInt(tr("max distance"), &a.cfg.similar_threshold, 1, 16);
        tip(tr("How different two photos may be and still count as similar (bits of a 64-bit picture fingerprint)"));
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 20);
    {
        int ndup = 0;
        for (auto& r : a.dup_rows) ndup += r.dup_of != 0;
        ImGui::BeginDisabled(busy || a.trash_busy || ndup == 0);
        if (danger_button(tr("Move extra copies to Trash..."), tr("Shows exactly which files would move to the Trash (and which copy stays) before anything moves."))) a.trash_confirm = true;
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Copy report"))) {
        std::string csv = "group,role,size,width,height,date,path\n";
        for (auto& l : a.dup_lines) {
            if (l.group < 0) continue;
            auto it = std::find_if(a.dup_rows.begin(), a.dup_rows.end(), [&](const DupRow& r) { return r.id == l.id; });
            if (it == a.dup_rows.end()) continue;
            csv += util::fmt("%d,%s,%lld,%d,%d,%s,\"%s\"\n", l.group + 1, l.keeper ? (l.similar ? "largest" : "keep") : (l.similar ? "similar" : "duplicate"),
                             (long long)it->size, it->width, it->height, it->date_value.c_str(), util::replace_all(it->src_path, "\"", "\"\"").c_str());
        }
        glfwSetClipboardString(a.win, csv.c_str());
        a.toast = tr("Report copied as CSV");
        a.toast_until = glfwGetTime() + 2;
    }
    tip(tr("Copy the duplicate groups as CSV"));
    ImGui::TextUnformatted(a.dup_summary.c_str());
    if (ImGui::RadioButton(tr("Exact duplicates"), !a.dup_show_similar)) { a.dup_show_similar = false; a.dups_dirty = true; }
    tip(tr("Byte-identical files"));
    ImGui::SameLine();
    if (ImGui::RadioButton(tr("Similar photos"), a.dup_show_similar)) { a.dup_show_similar = true; a.dups_dirty = true; }
    tip(tr("Resized, re-saved or burst photos that look alike (report only)"));
    ImGui::SameLine();
    ImGui::TextDisabled("%s", a.dup_show_similar ? tr("(report only: similar photos are never skipped automatically)")
                                                 : tr("(duplicates are not copied into the library; nothing is deleted)"));

    std::map<int64_t, const DupRow*> by_id;
    for (auto& r : a.dup_rows) by_id[r.id] = &r;
    float davail = ImGui::GetContentRegionAvail().x;
    a.dup_pane_w = std::clamp(a.dup_pane_w, 240.0f, std::max(240.0f, davail - 300));
    float pane_w = a.dup_pane_w;
    ImGui::BeginChild("duptable", ImVec2(davail - pane_w - 8, 0));
    if (a.dup_lines.empty()) ImGui::TextDisabled("%s", tr("No groups. Run \"Check duplicates\"."));
    else if (ImGui::BeginTable("dups", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable |
                                              ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(a.dup_show_similar ? "" : tr("Keep"), 0, 44);
        ImGui::TableSetupColumn(tr("Status"), 0, 110);
        ImGui::TableSetupColumn(tr("Size"), 0, 70);
        ImGui::TableSetupColumn(tr("Pixels"), 0, 100);
        ImGui::TableSetupColumn(tr("Date"), 0, 150);
        ImGui::TableSetupColumn(tr("Original"), 0, 700);
        ImGui::TableHeadersRow();
        std::vector<int> photo_lines;  // lines that are photos (not group headers)
        for (int i = 0; i < int(a.dup_lines.size()); i++)
            if (a.dup_lines[size_t(i)].group >= 0) photo_lines.push_back(i);
        int dpos = -1;
        for (int k = 0; k < int(photo_lines.size()); k++)
            if (a.dup_lines[size_t(photo_lines[size_t(k)])].id == a.dup_sel) { dpos = k; break; }
        auto dup_ids = [&] {
            std::vector<int64_t> ids;
            for (int i : photo_lines) ids.push_back(a.dup_lines[size_t(i)].id);
            return ids;
        };
        NavKey nk = nav_key(!a.viewer.open);
        if (nk != NK_NONE && !photo_lines.empty()) {
            if (nk == NK_DOWN || nk == NK_UP || nk == NK_LEFT || nk == NK_RIGHT) {
                int step = nk == NK_DOWN ? 1 : nk == NK_UP ? -1 : nk == NK_RIGHT ? 10 : -10;
                dpos = std::clamp(dpos < 0 ? 0 : dpos + step, 0, int(photo_lines.size()) - 1);
                a.dup_sel = a.dup_lines[size_t(photo_lines[size_t(dpos)])].id;
                a.scroll_sel = true;
            } else if ((nk == NK_ENTER || nk == NK_SPACE) && dpos >= 0) {
                open_viewer(a, dup_ids(), dpos);
            }
        }
        int sel_line = dpos >= 0 ? photo_lines[size_t(dpos)] : -1;
        ImGuiListClipper clip;
        clip.Begin(int(a.dup_lines.size()));
        if (a.scroll_sel && sel_line >= 0) clip.IncludeItemByIndex(sel_line);
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                const DupLine& l = a.dup_lines[size_t(i)];
                ImGui::TableNextRow();
                if (l.group < 0) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.10f)));
                    ImVec4 acc(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 1);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextColored(acc, "%s %lld", tr("Group"), (long long)l.id);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::TextColored(acc, "%s", l.text.c_str());
                    continue;
                }
                auto f = by_id.find(l.id);
                if (f == by_id.end()) continue;
                const DupRow& r = *f->second;
                ImGui::PushID(i);
                ImGui::TableNextColumn();
                if (!l.similar) {
                    if (ImGui::RadioButton("##keep", l.keeper) && !l.keeper && !busy) {
                        // Pin this copy as the keeper, then regroup (hashes are cached, so this is quick).
                        std::vector<int64_t> grp;
                        for (auto& o : a.dup_lines)
                            if (o.group == l.group) grp.push_back(o.id);
                        int64_t keep = l.id;
                        // Store the pin first (on the worker), then regroup once it is committed.
                        a.view.write([grp, keep, &a](Db& db) {
                            db.set_keeper(grp, keep);
                            a.want_dupes_run = true;
                            glfwPostEmptyEvent();
                        });
                    }
                    tip(tr("Keep this copy of the group; the others are the extra copies"));
                }
                ImGui::TableNextColumn();
                bool in_lib = !r.dest_path.empty();
                if (l.keeper) ImGui::TextColored(g_ok, "%s%s", l.similar ? tr("largest") : tr("kept"), in_lib ? " · lib" : "");
                else if (l.similar) ImGui::TextColored(g_info, "%s %d", tr("distance"), r.similar_dist);
                else ImGui::TextColored(g_warn, "%s", tr("duplicate"));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(util::human_size(r.size).c_str());
                ImGui::TableNextColumn();
                if (r.width) ImGui::Text("%d x %d", r.width, r.height);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.date_value.c_str());
                ImGui::TableNextColumn();
                if (ImGui::Selectable(r.src_path.c_str(), a.dup_sel == r.id,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap | ImGuiSelectableFlags_AllowDoubleClick)) {
                    a.dup_sel = r.id;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        auto ids = dup_ids();
                        open_viewer(a, ids, int(std::find(ids.begin(), ids.end(), r.id) - ids.begin()));
                    }
                }
                if (a.scroll_sel && i == sel_line) { ImGui::SetScrollHereY(0.5f); a.scroll_sel = false; }
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    vsplit("##split_dups", a.dup_pane_w, 240, 2000, true);
    ImGui::BeginChild("duppane", ImVec2(0, 0), ImGuiChildFlags_Borders);
    static Photo p;
    static int p_gen = -1;
    if (p_gen != a.view_gen) { p.id = 0; p_gen = a.view_gen; }
    if (a.dup_sel && (p.id == a.dup_sel || a.ui_db.load(a.dup_sel, p))) {
        request_thumb(a, p);
        upload_thumb(a);
        if (a.thumb.tex) {
            float s = std::min(ImGui::GetContentRegionAvail().x / a.thumb.w, 300.0f / a.thumb.h);
            ImGui::Image(ImTextureRef((ImTextureID)(intptr_t)a.thumb.tex), ImVec2(a.thumb.w * s, a.thumb.h * s));
        }
        ImGui::TextWrapped("%s", p.src_path.c_str());
        ImGui::TextDisabled("%s · %d x %d · %s", util::human_size(p.size).c_str(), p.width, p.height, p.date_value.c_str());
        if (!p.content_hash.empty()) ImGui::TextDisabled("XXH3-128 %s", p.content_hash.c_str());
        else if (!p.quick_hash.empty()) ImGui::TextDisabled("quick %s", p.quick_hash.c_str());
        if (ImGui::SmallButton(tr("Open folder"))) util::open_path(util::dirname(p.src_path));
        tip(tr("Open the folder that holds this photo"));
    } else {
        ImGui::TextDisabled("%s", tr("Select a file to preview it."));
    }
    ImGui::EndChild();
}

static void draw_log(App& a) {
    ImGui::BeginChild("log");
    static std::vector<LogLine> lines;
    static size_t seen = size_t(-1);
    if (a.log.size() != seen) {  // copy only when something new was logged
        lines = a.log.tail(5000);
        seen = a.log.size();
    }
    ImGuiListClipper clip;
    clip.Begin(int(lines.size()));
    while (clip.Step())
        for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
            const LogLine& l = lines[size_t(i)];
            ImVec4 c = l.level == 2 ? g_err : l.level == 1 ? g_warn : ImGui::GetStyleColorVec4(ImGuiCol_Text);
            ImGui::TextColored(c, "%s  %s", l.time.c_str(), l.text.c_str());
        }
    static size_t last = 0;
    if (a.log.size() != last) {
        ImGui::SetScrollHereY(1.0f);
        last = a.log.size();
    }
    ImGui::EndChild();
}

// ---- Settings, laid out like a preferences page: small-caps section titles, rounded cards, one setting per row
// (title and a one-line explanation on the left, the control on the right). Every change is saved at once.

namespace prefs {
struct Card {
    ImVec2 p0;
    float x0 = 0, w = 0;
    bool first = true;
};
static Card g_card;
static bool g_dirty = false;
// The index of every row (for "go to a setting"), and the row to scroll to / flash.
static std::vector<std::pair<std::string, std::string>> g_index;  // section, title
static std::set<std::string> g_index_keys, g_advanced_sections;
static std::string g_cur_section, g_find, g_flash_key;
static double g_flash_until = 0;
static bool g_advanced = false, g_indexing = false, g_in_advanced = false;
static const float kPad = 14;

static void section(const char* title) {
    g_cur_section = title;
    if (g_in_advanced) g_advanced_sections.insert(title);
    ImGui::Dummy(ImVec2(0, 10));
    std::string up;
    for (const char* s = title; *s; s++) up += char(toupper((unsigned char)*s));  // non-ASCII stays as it is
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 0.82f);
    ImGui::SetCursorPosX(g_card.x0 + 2);
    ImGui::TextDisabled("%s", up.c_str());
    ImGui::PopFont();
    ImGui::Dummy(ImVec2(0, 2));
}

static void card_begin() {
    g_card.first = true;
    ImGui::SetCursorPosX(g_card.x0);
    g_card.p0 = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->ChannelsSplit(2);
    dl->ChannelsSetCurrent(1);
}

static void card_end() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(g_card.p0.x + g_card.w, ImGui::GetCursorScreenPos().y);
    dl->ChannelsSetCurrent(0);
    dl->AddRectFilled(g_card.p0, p1, ImGui::GetColorU32(g_panel), 10.0f);
    dl->AddRect(g_card.p0, p1, ImGui::GetColorU32(ImGuiCol_Border), 10.0f);
    dl->ChannelsMerge();
}

// One row. Returns the screen position where a control of width ctrl_w goes (vertically centred, right-aligned).
static ImVec2 row(const char* title, const char* desc, float ctrl_w, const char* more = nullptr, bool error = false) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::SetCursorPosX(g_card.x0);
    ImVec2 top = ImGui::GetCursorScreenPos();
    float lh = ImGui::GetTextLineHeight(), small = ImGui::GetStyle().FontSizeBase * 0.86f;
    float h = kPad * 2 + lh + (desc && *desc ? small + 4 : 0);
    if (!g_card.first) dl->AddLine(ImVec2(top.x + 1, top.y), ImVec2(top.x + g_card.w - 1, top.y), ImGui::GetColorU32(ImGuiCol_Border));
    g_card.first = false;
    std::string key = g_cur_section + "\x1f" + title;
    if (g_index_keys.insert(key).second) g_index.push_back({g_cur_section, title});
    if (!g_indexing && !g_find.empty() && g_find == key) {  // arrived from "go to": bring it into view and flash it
        ImGui::SetScrollHereY(0.25f);
        g_find.clear();
        g_flash_key = key;
        g_flash_until = glfwGetTime() + 1.6;
    }
    if (!g_indexing && g_flash_key == key && glfwGetTime() < g_flash_until) {
        float t = float((g_flash_until - glfwGetTime()) / 1.6);
        dl->AddRectFilled(ImVec2(top.x + 1, top.y + 1), ImVec2(top.x + g_card.w - 1, top.y + h - 1),
                          ImGui::GetColorU32(ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.22f * t)), 8.0f);
        g_wake_at = glfwGetTime() + 0.016;
    }
    float text_w = g_card.w - kPad * 3 - ctrl_w;
    ImGui::SetCursorScreenPos(ImVec2(top.x + kPad, top.y + kPad));
    ImGui::TextUnformatted(title);
    if (desc && *desc) {
        ImGui::SetCursorScreenPos(ImVec2(top.x + kPad, top.y + kPad + lh + 3));
        ImGui::PushFont(nullptr, small);
        std::string d = desc;  // one line, cut with an ellipsis; the whole text on hover
        bool cut = false;
        while (d.size() > 8 && ImGui::CalcTextSize(d.c_str()).x > text_w) {
            size_t k = d.size() - 1;
            while (k > 0 && (d[k] & 0xC0) == 0x80) k--;  // stay on UTF-8 boundaries
            d.resize(k);
            cut = true;
        }
        if (cut) d += "…";
        if (error) ImGui::TextColored(kDanger, "%s", d.c_str());  // errors replace the grey line, in the row itself
        else ImGui::TextDisabled("%s", d.c_str());
        ImGui::PopFont();
        if (cut || more) tip(more ? (std::string(desc) + "\n" + more) : std::string(desc));
    } else if (more) {
        tip(more);
    }
    ImGui::SetCursorScreenPos(ImVec2(top.x, top.y + h));
    ImGui::Dummy(ImVec2(g_card.w, 0));
    ImVec2 after = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(top.x + g_card.w - kPad - ctrl_w, top.y + (h - ImGui::GetFrameHeight()) / 2));
    (void)after;
    return top;
}
// Call after the control: continue below the row.
static void row_end(ImVec2 top, const char* desc) {
    float lh = ImGui::GetTextLineHeight(), small = ImGui::GetStyle().FontSizeBase * 0.86f;
    float h = kPad * 2 + lh + (desc && *desc ? small + 4 : 0);
    ImGui::SetCursorScreenPos(ImVec2(top.x, top.y + h));
    ImGui::Dummy(ImVec2(g_card.w, 0));
    ImGui::SetCursorScreenPos(ImVec2(top.x, top.y + h));
}

static float seg_width(const std::vector<std::string>& labels) {
    float w = 4;
    for (auto& l : labels) w += ImGui::CalcTextSize(l.c_str()).x + 22;
    return w;
}
// Segmented control: the chosen option is a raised pill.
static bool seg(const char* id, int* v, const std::vector<std::string>& labels) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetFrameHeight(), w = seg_width(labels);
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(ImGuiCol_FrameBg), 7.0f);
    bool changed = false;
    ImGui::PushID(id);
    // where the chosen pill should be (relative to the track), and where it is drawn now: it slides there in ~150 ms
    float sel_x = 2, sel_w = 0, x = 2;
    for (int i = 0; i < int(labels.size()); i++) {
        float iw = ImGui::CalcTextSize(labels[size_t(i)].c_str()).x + 22;
        if (i == *v) { sel_x = x; sel_w = iw; }
        x += iw;
    }
    ImGuiStorage* st = ImGui::GetStateStorage();
    ImGuiID kx = ImGui::GetID("##pill_x"), kw = ImGui::GetID("##pill_w");
    float cx = st->GetFloat(kx, -1), cw = st->GetFloat(kw, -1);
    if (cx < 0 || ImGui::IsWindowAppearing() || anim::g_reduce) { cx = sel_x; cw = sel_w; }
    float k = 1.0f - std::exp(-ImGui::GetIO().DeltaTime / 0.035f);  // ease-out, settles in ~150 ms
    cx += (sel_x - cx) * k;
    cw += (sel_w - cw) * k;
    if (std::fabs(sel_x - cx) < 0.5f && std::fabs(sel_w - cw) < 0.5f) { cx = sel_x; cw = sel_w; }
    else g_wake_at = glfwGetTime() + 0.016;  // keep drawing until it arrives
    st->SetFloat(kx, cx);
    st->SetFloat(kw, cw);
    if (sel_w > 0) {
        ImVec2 a(p.x + cx, p.y + 2), b(p.x + cx + cw, p.y + h - 2);
        if (g_seg_shadow) dl->AddRectFilled(ImVec2(a.x, a.y + 1), ImVec2(b.x, b.y + 1), IM_COL32(0, 0, 0, 22), 6.0f);  // soft 1 px shadow
        dl->AddRectFilled(a, b, ImGui::GetColorU32(g_seg_on), 6.0f);
        dl->AddRect(a, b, ImGui::GetColorU32(ImGuiCol_Border), 6.0f);
    }
    x = p.x + 2;
    for (int i = 0; i < int(labels.size()); i++) {
        float iw = ImGui::CalcTextSize(labels[size_t(i)].c_str()).x + 22;
        ImGui::SetCursorScreenPos(ImVec2(x, p.y));
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##s", ImVec2(iw, h)) && *v != i) { *v = i; changed = true; }
        bool hov = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
        ImGui::PopID();
        if (held) dl->AddRectFilled(ImVec2(x, p.y + 2), ImVec2(x + iw, p.y + h - 2), IM_COL32(0, 0, 0, 26), 6.0f);  // pressed
        bool on = *v == i;
        ImVec2 ts = ImGui::CalcTextSize(labels[size_t(i)].c_str());
        dl->AddText(ImVec2(x + (iw - ts.x) / 2, p.y + (h - ts.y) / 2), ImGui::GetColorU32(on || hov ? ImGuiCol_Text : ImGuiCol_TextDisabled),
                    labels[size_t(i)].c_str());
        x += iw;
    }
    ImGui::PopID();
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(w, h));  // one item for the layout: SameLine and the next line work as for any widget
    g_dirty |= changed;
    return changed;
}

// The row helpers: a title, an explanation, and one control.
static bool choice(const char* title, const char* desc, int* v, const std::vector<std::string>& labels, const char* more = nullptr) {
    ImVec2 top = row(title, desc, seg_width(labels), more);
    bool ch = seg(title, v, labels);
    row_end(top, desc);
    return ch;
}
static bool toggle(const char* title, const char* desc, bool* b, const char* more = nullptr) {
    int v = *b ? 1 : 0;
    bool ch = choice(title, desc, &v, {tr("Off"), tr("On")}, more);
    *b = v == 1;
    return ch;
}
static bool combo(const char* title, const char* desc, int* v, const char* const* items, int n, float w, const char* more = nullptr) {
    ImVec2 top = row(title, desc, w, more);
    ImGui::SetNextItemWidth(w);
    bool ch = ImGui::Combo((std::string("##") + title).c_str(), v, items, n);
    row_end(top, desc);
    g_dirty |= ch;
    return ch;
}
static bool slider_i(const char* title, const char* desc, int* v, int lo, int hi, const char* fmt = "%d") {
    ImVec2 top = row(title, desc, 200);
    ImGui::SetNextItemWidth(200);
    bool ch = ImGui::SliderInt((std::string("##") + title).c_str(), v, lo, hi, fmt);
    row_end(top, desc);
    g_dirty |= ch;
    return ch;
}
static bool slider_f(const char* title, const char* desc, float* v, float lo, float hi, const char* fmt = "%.2f") {
    ImVec2 top = row(title, desc, 200);
    ImGui::SetNextItemWidth(200);
    bool ch = ImGui::SliderFloat((std::string("##") + title).c_str(), v, lo, hi, fmt);
    row_end(top, desc);
    g_dirty |= ch;
    return ch;
}
// Text: saved when you leave the field.
static void text(const char* title, const char* desc, char* buf, size_t n, float w, const char* hint = "", bool password = false) {
    ImVec2 top = row(title, desc, w);
    ImGui::SetNextItemWidth(w);
    ImGui::InputTextWithHint((std::string("##") + title + desc).c_str(), hint, buf, n, password ? ImGuiInputTextFlags_Password : 0);
    if (ImGui::IsItemDeactivatedAfterEdit()) g_dirty = true;
    row_end(top, desc);
}
static void number(const char* title, const char* desc, int* v, int step = 1) {
    ImVec2 top = row(title, desc, 130);
    ImGui::SetNextItemWidth(130);
    ImGui::InputInt((std::string("##") + title + desc).c_str(), v, step);
    if (ImGui::IsItemDeactivatedAfterEdit()) g_dirty = true;
    row_end(top, desc);
}
}  // namespace prefs

static void draw_settings(App& a) {
    using namespace prefs;
    if (!a.settings_loaded) load_settings_buffers(a);
    Config& c = a.cfg;
    ImGui::BeginChild("settings", ImVec2(0, 0), 0, ImGuiWindowFlags_None);
    float avail = ImGui::GetContentRegionAvail().x;
    g_card.w = std::min(avail - 8, 760.0f);
    g_card.x0 = std::max(4.0f, (avail - g_card.w) / 2);
    g_dirty = false;

    section(tr("Preferences"));
    card_begin();
    {
        static const int order[] = {THEME_SYSTEM, THEME_DARK, THEME_TOKYO, THEME_LIGHT};
        int sel = c.theme == THEME_SYSTEM ? 0 : c.theme == THEME_DARK ? 1 : c.theme == THEME_TOKYO ? 2 : 3;
        if (choice(tr("Appearance"), tr("System (as the desktop is), Dark, Tokyo Night (deep blue with soft neon colours) or Light; Ctrl+Shift+T switches"),
                   &sel, {tr("System"), tr("Dark"), tr("Tokyo Night"), tr("Light")})) {
            c.theme = order[sel];
            if (c.theme == THEME_SYSTEM) g_system_dark = -1;
            theme_default_accent(theme_kind(c), c.accent);  // each theme brings its own accent
            apply_style(c);
        }
    }
    {
        ImVec2 top = row(tr("Accent colour"), tr("The main buttons and highlights; each theme sets its own first"), ImGui::GetFrameHeight());
        if (ImGui::ColorEdit3("##accent", c.accent, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) { apply_style(c); g_dirty = true; }
        row_end(top, "x");
    }
    {
        int lang = get_lang();
        std::vector<std::string> names;
        for (int i = 0; i < L_COUNT; i++) names.push_back(kLangNames[i]);
        if (choice(tr("Language"), tr("Applies at once, no restart"), &lang, names)) {
            set_lang(Lang(lang));
            c.lang = kLangCodes[lang];
            a.want_font_rebuild = true;  // the new language's CJK face goes first
        }
    }
    {
        const float sizes[] = {16, 17.6f, 20, 24};
        int si = 0;
        for (int i = 0; i < 4; i++)
            if (std::fabs(c.font_size - sizes[i]) < std::fabs(c.font_size - sizes[si])) si = i;
        if (choice(tr("Text size"), tr("Everything in the window, larger; Ctrl+ Ctrl– Ctrl+0 too"), &si, {"100%", "110%", "125%", "150%"})) c.font_size = sizes[si];
    }
    if (choice(tr("Motion"), tr("Short fades and slides; Reduced keeps only the fades (the system setting is followed by default)"), &c.motion,
               {tr("System"), tr("Full"), tr("Reduced")}))
        apply_motion(c);
    {
        int tb = c.system_titlebar ? 1 : 0;
        if (choice(tr("Title bar"), tr("The app's own top bar with window buttons, or the desktop's (after restart)"), &tb, {tr("App"), tr("System")}))
            c.system_titlebar = tb == 1;
    }
    choice(tr("Renderer"), (std::string(tr("Software when the GPU driver cannot open a window; now: ")) + a.gl_renderer).c_str(), &c.renderer,
           {tr("Auto"), tr("GPU"), tr("Software")}, tr("Applies after restart"));
    card_end();

    section(tr("Folders and organizing"));
    card_begin();
    toggle(tr("Watch the folders"), tr("Every 10 minutes while idle, new or changed photos are analysed quietly; files are not changed"), &c.watch_folders);
    toggle(tr("Preview before changes"), tr("Analyze ends with a list to review; nothing is copied or written until you press Apply"), &c.preview_first);
    {
        const char* ops[] = {tr("Copy and rename"), tr("Move and rename"), tr("Rename in place"), tr("Only add metadata")};
        const char* what = c.file_op == OP_COPY ? tr("Your photos stay as they are; renamed copies go into month folders")
                         : c.file_op == OP_MOVE ? tr("Your photos are moved into month folders and renamed (no copies)")
                         : c.file_op == OP_RENAME ? tr("Your photos keep their folders and get date names")
                                                  : tr("Missing dates, tags, descriptions and places are added where the files are");
        combo(tr("Organize does"), what, &c.file_op, ops, OP_COUNT, 200);
    }
    choice(tr("File names"), tr("20190512_00001.jpg, or the original name in front: IMG_20190512_00001.jpg"), &c.name_style,
           {tr("Date"), tr("Original + date")});
    {
        const char* seps[] = {"_", "-", " ", ""};
        int si = c.name_sep == "-" ? 1 : c.name_sep == " " ? 2 : c.name_sep.empty() ? 3 : 0;
        if (choice(tr("Name separator"), tr("Between the date and the serial number"), &si, {"_", "-", tr("space"), tr("none")})) c.name_sep = seps[si];
    }
    {
        int d = std::clamp(c.sn_digits, 3, 6) - 3;
        if (choice(tr("Serial digits"), tr("How many digits the serial number has"), &d, {"3", "4", "5", "6"})) c.sn_digits = d + 3;
    }
    choice(tr("Month folders"), tr("How month folders are named inside the organized folder"), &c.layout, {"2019/2019-05", "2019-05", "2019/05"});
    {
        std::string hint = c.folder.empty() ? std::string(tr("next to your photos")) : util::basename(c.folder) + "/jev-organized";
        ImVec2 top = row(tr("Organized copies folder"), tr("Empty: a jev-organized folder inside each photo folder"), 330);
        ImGui::SetNextItemWidth(250);
        ImGui::InputTextWithHint("##outdir", hint.c_str(), a.lib_buf, sizeof a.lib_buf);
        if (ImGui::IsItemDeactivatedAfterEdit()) g_dirty = true;
        ImGui::SameLine(0, 6);
        if (ImGui::Button((std::string(tr("Browse...")) + "##out").c_str(), ImVec2(74, 0))) start_dir_dialog(a, 1);
        tip(tr("Choose a folder with the file dialog"));
        row_end(top, "x");
    }
    card_end();

    section(tr("Info written into files"));
    card_begin();
    choice(tr("Where it goes"), tr("Into the file, a .xmp file next to it, or only the catalog"), &c.write_mode,
           {tr("In the file"), tr("Sidecar"), tr("Catalog only")});
    toggle(tr("Replace keywords written earlier"), tr("When tags change, keywords jev-photos wrote are updated; the file's own stay"), &c.rewrite_tags);
    toggle(tr("Write a description"), tr("The AI prompt plus \"Shows: <tags>\"; existing text only changes as you decide"), &c.write_description);
    toggle(tr("Keywords from AI prompts"), tr("Tag-style prompts give their words; prose prompts are summarised by the LLM, once each"), &c.gen_keywords);
    {
        const char* dp[] = {tr("Only fill empty ones"), tr("Rules + LLM + jev (recommended)"), tr("Rules + LLM"), tr("Always ask me")};
        combo(tr("Existing descriptions"), tr("Placeholders are replaced; for the rest, keep / append / replace is decided like this"), &c.desc_policy, dp,
              DESC_COUNT, 240, tr("Replaced text is kept in Xmp.jev.PreviousDescription."));
    }
    card_end();

    section(tr("Image recognition"));
    card_begin();
    toggle(tr("Recognise pictures"), tr("CLIP reads each picture once, for tags and search by meaning"), &c.clip_enabled);
    toggle(tr("Ask before analysing"), tr("Analyze first asks how much image analysis to do (the slow part)"), &c.analyze_ask);
    {
        int ci = c.clip_model == "b32" ? 1 : c.clip_model == "l14" ? 2 : 0;
        if (choice(tr("Model"), tr("Best: ViT-L/14, ~0.35 s per photo; Fast: ViT-B/32, ~60 ms"), &ci, {tr("Auto"), tr("Fast"), tr("Best")}))
            c.clip_model = ci == 1 ? "b32" : ci == 2 ? "l14" : "auto";
    }
    choice(tr("Run on"), tr("The GPU needs an ONNX Runtime build with CUDA"), &c.clip_device, {tr("Auto"), tr("CPU"), tr("GPU")});
    {
        // The models are downloaded once from Hugging Face (curl; on Windows 10+ it is built in).
        struct Dl {
            std::mutex mu;
            std::string status, err;
            std::atomic<bool> busy{false};
        };
        static auto dl = std::make_shared<Dl>();
        bool have_b32 = clip_files_present(models_dir() + "/clip-vit-base-patch32");
        bool have_l14 = clip_files_present(models_dir() + "/clip-vit-large-patch14");
        std::string st, er;
        {
            std::lock_guard<std::mutex> l(dl->mu);
            st = dl->status;
            er = dl->err;
        }
        std::string desc = dl->busy ? std::string(tr("Downloading")) + " " + st
                         : !er.empty() ? std::string(tr("Download failed: ")) + er
                                       : util::fmt("%s: %s  ·  %s: %s", tr("Fast"), have_b32 ? tr("ready") : tr("not downloaded"), tr("Best"),
                                                   have_l14 ? tr("ready") : tr("not downloaded"));
        float bw = 0;
        std::vector<std::pair<const char*, const char*>> btns;
        if (!have_b32) btns.push_back({"b32", tr("Get fast (600 MB)")});
        if (!have_l14) btns.push_back({"l14", tr("Get best (1.7 GB)")});
        for (auto& b : btns) bw += ImGui::CalcTextSize(b.second).x + 20;
        ImVec2 top = row(tr("Models"), desc.c_str(), std::max(bw, 1.0f), dl->busy ? nullptr : tr("Saved in your app data folder; downloaded once"),
                         !dl->busy && !er.empty());
        ImGui::BeginDisabled(dl->busy);
        for (size_t i = 0; i < btns.size(); i++) {
            if (i) ImGui::SameLine(0, 6);
            if (ImGui::Button(btns[i].second)) {
                std::string which = btns[i].first;
                dl->busy = true;
                {
                    std::lock_guard<std::mutex> l(dl->mu);
                    dl->err.clear();
                    dl->status.clear();
                }
                std::thread([which, d = dl, &a] {
                    std::string err;
                    bool ok = clip_download(which, err, [d](const std::string& s2) {
                        std::lock_guard<std::mutex> l(d->mu);
                        d->status = s2;
                        glfwPostEmptyEvent();
                    });
                    {
                        std::lock_guard<std::mutex> l(d->mu);
                        d->err = ok ? "" : err;
                    }
                    d->busy = false;
                    a.health_at = -100;  // check again: recognition may be ready now
                    glfwPostEmptyEvent();
                }).detach();
            }
        }
        ImGui::EndDisabled();
        row_end(top, "x");
    }
    slider_i(tr("CPU threads"), tr("More is faster but leaves less of the machine for other work"), &c.clip_threads, 1, 16);
    slider_i(tr("Tags per photo"), tr("At most this many tags per photo"), &c.clip_max_tags, 1, 15);
    slider_i(tr("Thumbnail size"), tr("Side of the cached thumbnails the analysis and the lists use (px)"), &c.thumb_side, 256, 1024);
    {
        ImVec2 top = row(tr("Tag list"), tag_vocabulary_path().c_str(), 90);
        if (ImGui::Button((std::string(tr("Edit...")) + "##tags").c_str(), ImVec2(90, 0))) a.tag_editor_open = true;
        tip(tr("The words recognition chooses from: add, rename, move or delete tags"));
        row_end(top, "x");
    }
    card_end();

    section(tr("Assistant (language model)"));
    card_begin();
    toggle(tr("Use the language model"), tr("Ask search, keywords from prompts, description and tag checks"), &c.llm_enabled);
    text(tr("Address"), tr("An OpenAI-compatible server"), a.llm_url, sizeof a.llm_url, 300, "http://127.0.0.1:8888/v1");
    text(tr("Model"), tr("The model name on that server"), a.llm_model, sizeof a.llm_model, 300);
    text(tr("API key"), tr("Only for servers that need one; kept in the settings file (mode 0600)"), a.llm_key, sizeof a.llm_key, 300, "", true);
    toggle(tr("Disable thinking"), tr("Much faster answers from reasoning models (Qwen3 style)"), &c.llm_no_think);
    {
        const char* dec[] = {tr("LLM"), tr("jev only"), tr("Local ranking only"), tr("LLM, jev for close calls")};
        combo(tr("Who picks Ask results"), tr("Who rates the candidates of a sentence search"), &c.ask_decider, dec, JUDGE_COUNT, 220);
    }
    slider_i(tr("Keep results rated at least"), tr("Ask search drops candidates rated below this"), &c.ask_keep, 0, 100, "%d%%");
    slider_i(tr("Candidates per request"), tr("How many candidates go into one LLM request"), &c.ask_batch, 5, 60);
    slider_i(tr("Most candidates to check"), tr("Ask search rates at most this many"), &c.ask_max_candidates, 20, 500);
    slider_f(tr("Meaning search strictness"), tr("How clearly a picture must show what you typed; higher = fewer, surer results"), &c.search_min_match, 0.1f, 0.8f);
    number(tr("Timeout (s)"), tr("Give up on a request after this many seconds"), &c.llm_timeout);
    card_end();

    section(tr("jev decisions"));
    card_begin();
    toggle(tr("Use jev"), tr("Settles close calls: dates, places, search results, checks of the LLM"), &c.jev_enabled);
    text(tr("Address"), tr("The jev decision API"), a.jev_url, sizeof a.jev_url, 300, "http://127.0.0.1:8011");
    text(tr("Model"), tr("The jev model name, e.g. julia-1"), a.jev_model, sizeof a.jev_model, 300);
    text(tr("API key"), tr("Only when the server needs one"), a.jev_key, sizeof a.jev_key, 300, "", true);
    number(tr("Timeout (s)"), tr("Give up on a request after this many seconds"), &c.jev_timeout);
    card_end();

    section(tr("Advanced"));
    card_begin();
    toggle(tr("Show advanced settings"), tr("The vision-language model and the numbers behind date and place decisions"), &g_advanced);
    {
        ImVec2 top = row(tr("Model tests"), tr("For development: send your own input to the language, vision, recognition and jev models"), 120);
        if (ImGui::Button((std::string(ICON_FLASK " ") + tr("Open...")).c_str(), ImVec2(120, 0))) a.show_model_tests = true;
        row_end(top, "x");
    }
    card_end();
    g_in_advanced = true;
    if (g_advanced) {
        section(tr("Vision-language model"));
        card_begin();
        toggle(tr("Use a vision model"), tr("Captions and text in screenshots; needs a model server and a lot of memory"), &c.vl_enabled);
        text(tr("Address"), tr("OpenAI-compatible, e.g. http://127.0.0.1:11434/v1"), a.vl_url, sizeof a.vl_url, 300);
        text(tr("Model"), tr("The vision model name"), a.vl_model, sizeof a.vl_model, 300);
        text(tr("API key"), tr("Only when the server needs one"), a.vl_key, sizeof a.vl_key, 300, "", true);
        text(tr("Tag language"), tr("The language of the tags it writes"), a.vl_lang, sizeof a.vl_lang, 160);
        slider_i(tr("Screenshot image side"), tr("Size of images sent for screenshots and text (px)"), &c.vl_max_side, 384, 1024);
        choice(tr("Run on (Ollama)"), tr("Where Ollama runs the vision model"), &c.vl_device, {tr("Auto"), tr("CPU")});
        number(tr("Context"), tr("Ollama num_ctx"), &c.vl_num_ctx, 1024);
        c.vl_num_ctx = std::clamp(c.vl_num_ctx, 2048, 65536);
        slider_i(tr("Concurrency"), tr("Photos handled at once"), &c.vl_concurrency, 1, 8);
        number(tr("Timeout (s)"), tr("Give up on a request after this many seconds"), &c.vl_timeout);
        toggle(tr("JSON mode"), tr("Ask for strict JSON answers"), &c.vl_json_mode);
        card_end();

        section(tr("Decision rules"));
        card_begin();
        slider_f(tr("Uncertain below"), tr("Dates less sure than this are marked uncertain"), &c.review_below, 0, 1);
        slider_f(tr("Ask jev when margin below"), tr("jev is asked when the best two date answers are this close"), &c.jev_margin, 0.5f, 1);
        slider_f(tr("jev weight for dates"), tr("How much jev's answer counts in a close date decision"), &c.jev_date_weight, 0, 1);
        slider_f(tr("Place vote: rules"), tr("Weight of the folder-name rules"), &c.loc_w_rules, 0, 1);
        slider_f(tr("Place vote: vision model"), tr("Weight of the vision model"), &c.loc_w_vl, 0, 1);
        slider_f(tr("Place vote: jev"), tr("Weight of jev"), &c.loc_w_jev, 0, 1);
        slider_f(tr("Accept a place at"), tr("A place needs at least this score"), &c.loc_accept, 0, 1);
        number(tr("Earliest year"), tr("Dates before this year are treated as wrong"), &c.min_year);
        slider_i(tr("Scan threads"), tr("Files read in parallel while scanning"), &c.scan_threads, 1, 16);
        card_end();
    }
    g_in_advanced = false;

    ImGui::Dummy(ImVec2(0, 10));
    ImGui::SetCursorPosX(g_card.x0);
    if (ImGui::Button(tr("Test connections"))) check_health(a);
    tip(tr("Check image recognition, the language model and jev now"));
    ImGui::SameLine();
    ImGui::TextDisabled("%s", tr("Changes are saved as you make them."));
    ImGui::Dummy(ImVec2(0, 10));

    if (g_dirty && !g_indexing) {  // keep it: the text fields, then the file (the command line reads it too)
        c.output = util::trim(a.lib_buf);
        c.jev_url = a.jev_url;
        c.jev_model = a.jev_model;
        c.jev_key = a.jev_key;
        c.vl_url = a.vl_url;
        c.vl_model = a.vl_model;
        c.vl_key = a.vl_key;
        c.vl_tag_lang = a.vl_lang;
        c.llm_url = a.llm_url;
        c.llm_model = a.llm_model;
        c.llm_key = a.llm_key;
        c.jev_timeout = std::clamp(c.jev_timeout, 1, 600);
        c.vl_timeout = std::clamp(c.vl_timeout, 5, 3600);
        c.llm_timeout = std::clamp(c.llm_timeout, 1, 3600);
        save_config(c, a.config_file);
        if (!a.db.is_open()) open_db(a);
        static double last_health = 0;
        if (glfwGetTime() - last_health > 2) { last_health = glfwGetTime(); check_health(a); }
    }
    ImGui::EndChild();
}

// Analyze: first ask how much image analysis and tagging to do (it is the slow part), unless turned off.
static void request_analyze(App& a) {
    if (a.cfg.folders.empty()) { start_dir_dialog(a, 3); return; }
    if (!a.cfg.clip_enabled || !a.cfg.analyze_ask) {
        a.opts.clip_mode = a.opts.tag_mode = 0;
        { start_run(a, analyze_stages(a.cfg)); a.auto_meta_step = a.cfg.auto_meta ? -1 : 0; }
        return;
    }
    ClipInfo info = clip_choose(a.cfg);
    Config q = a.cfg;
    q.clip_model = "b32";
    ClipInfo quick = clip_choose(q);
    a.ana_counts = a.ui_db.clip_counts(all_scope(a.cfg), info.id, tag_vocabulary_hash(info.id));
    a.ana_counts_quick = a.ui_db.clip_counts(all_scope(a.cfg), quick.id, tag_vocabulary_hash(quick.id));
    a.ana_clip = 0;
    a.ana_tags = 0;
    a.analyze_dlg = true;
}

static void draw_analyze_dialog(App& a) {
    if (a.analyze_dlg) {
        ImGui::OpenPopup(tr("Analyze###ana"));
        a.analyze_dlg = false;
    }
    ImGui::SetNextWindowSize(ImVec2(640, 0), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(tr("Analyze###ana"), nullptr, ImGuiWindowFlags_NoResize)) return;
    Config& c = a.cfg;
    const ClipCounts& k = a.ana_counts;
    bool large = clip_choose(c).id.find("large") != std::string::npos;
    // Seconds per photo on this CPU (measured: ViT-L/14 ~0.35 s and ViT-B/32 ~0.06 s with 8 threads).
    double scale = std::max(1.0, 8.0 / std::max(1, c.clip_threads));
    double per = (large ? 0.35 : 0.06) * scale, per_quick = 0.06 * scale;
    auto est = [](double secs) { return secs < 1 ? std::string(tr("seconds")) : fmt_duration(secs); };
    ImGui::TextWrapped("%s", tr("Scanning, duplicates, dates and places are quick. Image analysis (CLIP reads every picture) is the slow part; tags are then worked out from it in seconds."));
    ImGui::SeparatorText(tr("Image analysis (CLIP)"));
    ImGui::RadioButton(util::fmt("%s  (%d %s, ~%s)", tr("Only photos not analysed yet"), k.no_vec, tr("photos"), est(k.no_vec * per).c_str()).c_str(), &a.ana_clip, 0);
    tip(tr("Recommended. New photos, and photos analysed with another model, are read; the rest keep their analysis."));
    ImGui::RadioButton(util::fmt("%s  (%d %s, ~%s)", tr("All photos again"), k.total, tr("photos"), est(k.total * per).c_str()).c_str(), &a.ana_clip, 1);
    tip(tr("Reads every picture again. Only needed after changing the model settings or if analysis results look wrong."));
    ImGui::RadioButton(util::fmt("%s  (%d %s, ~%s)", tr("Quick: the small model for photos not analysed yet"), a.ana_counts_quick.no_vec, tr("photos"),
                                 est(a.ana_counts_quick.no_vec * per_quick).c_str()).c_str(), &a.ana_clip, 2);
    tip(tr("ViT-B/32: about 6 times faster, noticeably less accurate. A later normal run upgrades these photos to the better model."));
    ImGui::RadioButton(tr("Skip image analysis this time"), &a.ana_clip, 3);
    tip(tr("No picture is read. New photos stay untagged and are not found by meaning search until a later run."));
    ImGui::SeparatorText(tr("Tags"));
    ImGui::RadioButton(util::fmt("%s  (%d %s)", tr("Only missing or outdated tags"), k.stale, tr("outdated")).c_str(), &a.ana_tags, 0);
    tip(tr("Recommended. Photos analysed now get tags; photos tagged with an older tag list are re-tagged."));
    ImGui::RadioButton(util::fmt("%s  (%d)", tr("Re-tag all analysed photos"), k.total - k.no_vec).c_str(), &a.ana_tags, 1);
    tip(tr("Recomputes every photo's tags from its stored analysis (fast: no picture is read). Your own tags are kept."));
    ImGui::RadioButton(util::fmt("%s  (%d)", tr("Re-tag photos with few tags"), k.few).c_str(), &a.ana_tags, 2);
    tip(tr("Photos with fewer than 2 confident tags get another look with the current tag list."));
    ImGui::SeparatorText(tr("Corrections"));
    ImGui::BeginDisabled(a.llm_ok != 1);
    ImGui::Checkbox(tr("Check generated tags against AI prompts (uses the LLM on the GPU server)"), &a.ana_check);
    ImGui::EndDisabled();
    tip(a.llm_ok == 1 ? tr("After tagging, the LLM reads each new AI image's prompt and proposes fixes of wrong tags (about 1-3 s per image on the GPU server). Off by default.")
                      : tr("The LLM is down. Actions > Corrections can check with the rules alone."));
    ImGui::SeparatorText(tr("Speed"));
    ImGui::SetNextItemWidth(200);
    ImGui::SliderInt(tr("CPU threads for image analysis"), &c.clip_threads, 1, 16);
    tip(tr("More threads analyse faster but leave less of the machine for other work."));
    bool dont = !c.analyze_ask;
    if (ImGui::Checkbox(tr("Don't ask again (always use the recommended choices)"), &dont)) c.analyze_ask = !dont;
    tip(tr("Settings > Tagging and search turns this question back on."));
    ImGui::Spacing();
    if (primary_button(tr("Start"), tr("Analyze with these choices. Nothing in your files changes: the results are shown to review first."), ImVec2(140, 0))) {
        a.opts.clip_mode = a.ana_clip;
        a.opts.tag_mode = a.ana_tags;
        save_config(c, a.config_file);
        start_run(a, analyze_stages(c) | (a.ana_check && a.llm_ok == 1 ? ST_FIX : 0));
        a.auto_meta_step = a.cfg.auto_meta ? -1 : 0;  // -1: an Analyze run that should be followed by Auto set meta
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    tip(tr("Close without analyzing (Esc)"));
    ImGui::EndPopup();
}

// The window has no system title bar: the empty part of the top bar moves it (double-click maximises), and the
// grip in the bottom-right corner resizes it.
static void window_chrome(App& a, float bar_bottom) {
    static bool moving = false, sizing = false;
    static double start_gx = 0, start_gy = 0;
    static int start_x = 0, start_y = 0, start_w = 0, start_h = 0;
    int wx, wy, ww, wh;
    glfwGetWindowPos(a.win, &wx, &wy);
    glfwGetWindowSize(a.win, &ww, &wh);
    double cx, cy;
    glfwGetCursorPos(a.win, &cx, &cy);
    double gx = wx + cx, gy = wy + cy;  // cursor on the screen: stable while the window moves
    ImGuiIO& io = ImGui::GetIO();
    bool free_spot = !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    bool on_bar = io.MousePos.y >= 0 && io.MousePos.y < bar_bottom && free_spot;
    // resize grip
    ImVec2 vs = ImGui::GetMainViewport()->Size;
    ImVec2 g0(vs.x - 18, vs.y - 18);
    bool on_grip = io.MousePos.x >= g0.x && io.MousePos.y >= g0.y && free_spot;
    ImDrawList* fg = ImGui::GetForegroundDrawList();
    ImU32 gc = ImGui::GetColorU32(on_grip || sizing ? ImGuiCol_Text : ImGuiCol_TextDisabled);
    for (int k = 0; k < 3; k++) fg->AddLine(ImVec2(vs.x - 4 - 5 * k, vs.y - 2), ImVec2(vs.x - 2, vs.y - 4 - 5 * k), gc, 1.5f);
    if (on_grip || sizing) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
    if (ImGui::IsMouseClicked(0)) {
        if (on_grip) { sizing = true; start_w = ww; start_h = wh; }
        else if (on_bar) { moving = true; start_x = wx; start_y = wy; }
        start_gx = gx;
        start_gy = gy;
    }
    if (on_bar && ImGui::IsMouseDoubleClicked(0)) {
        moving = false;
        glfwGetWindowAttrib(a.win, GLFW_MAXIMIZED) ? glfwRestoreWindow(a.win) : glfwMaximizeWindow(a.win);
    }
    if (!ImGui::IsMouseDown(0)) moving = sizing = false;
    if (moving && (gx != start_gx || gy != start_gy)) {
        if (glfwGetWindowAttrib(a.win, GLFW_MAXIMIZED)) glfwRestoreWindow(a.win);
        glfwSetWindowPos(a.win, start_x + int(gx - start_gx), start_y + int(gy - start_gy));
    }
    if (sizing) glfwSetWindowSize(a.win, std::max(900, start_w + int(gx - start_gx)), std::max(560, start_h + int(gy - start_gy)));
    // The other edges and corners resize too (5 px bands, not while maximized or full screen).
    static int edge = 0;  // bits: 1 left, 2 right, 4 top, 8 bottom
    static int e_x, e_y, e_w, e_h;
    bool maxed = glfwGetWindowAttrib(a.win, GLFW_MAXIMIZED) || glfwGetWindowMonitor(a.win);
    if (!maxed && !moving && !sizing) {
        const float band = 5;
        int hover = 0;
        if (io.MousePos.x >= 0 && io.MousePos.x < band) hover |= 1;
        if (io.MousePos.x > vs.x - band && io.MousePos.x <= vs.x) hover |= 2;
        if (io.MousePos.y >= 0 && io.MousePos.y < band) hover |= 4;
        if (io.MousePos.y > vs.y - band && io.MousePos.y <= vs.y) hover |= 8;
        if (edge == 0 && hover && !ImGui::IsAnyItemActive()) {
            ImGui::SetMouseCursor((hover & 3) && (hover & 12) ? ((hover == 5 || hover == 10) ? ImGuiMouseCursor_ResizeNWSE : ImGuiMouseCursor_ResizeNESW)
                                  : (hover & 3) ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
            if (ImGui::IsMouseClicked(0)) { edge = hover; e_x = wx; e_y = wy; e_w = ww; e_h = wh; start_gx = gx; start_gy = gy; }
        }
    }
    if (edge) {
        if (!ImGui::IsMouseDown(0)) edge = 0;
        else {
            int dx = int(gx - start_gx), dy = int(gy - start_gy), x = e_x, y = e_y, w = e_w, h = e_h;
            if (edge & 1) { w = std::max(900, e_w - dx); x = e_x + e_w - w; }
            if (edge & 2) w = std::max(900, e_w + dx);
            if (edge & 4) { h = std::max(560, e_h - dy); y = e_y + e_h - h; }
            if (edge & 8) h = std::max(560, e_h + dy);
            glfwSetWindowPos(a.win, x, y);
            glfwSetWindowSize(a.win, w, h);
        }
    }
}

// Help: what each kind of data is, keys, search syntax, credits.

// The tag list CLIP chooses from: a paged, searchable table with add / edit / delete, saved to tags.txt.
static void draw_tag_editor(App& a) {
    if (a.tag_editor_open) {
        ImGui::OpenPopup(tr("Tag list###tagedit"));
        a.tag_editor_open = false;
        a.ed_tags = load_tag_vocabulary();
        a.ed_loaded = true;
        a.ed_dirty = false;
        a.ed_page = 0;
        a.ed_edit = -1;
        a.ed_name[0] = a.ed_search[0] = a.ed_newcat[0] = 0;
    }
    ImGui::SetNextWindowSize(ImVec2(760, 620), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(tr("Tag list###tagedit"), nullptr)) return;
    auto& tags = a.ed_tags;
    std::vector<std::string> cats;
    for (auto& t : tags)
        if (std::find(cats.begin(), cats.end(), t.category) == cats.end()) cats.push_back(t.category);
    ImGui::TextWrapped("%s", tr("CLIP picks each photo's tags from this list. Tags compete only with the other tags of their category, so put a tag in the category it belongs to (people, scene, objects...). 'medium' is what kind of picture it is."));
    // search + category filter
    ImGui::SetNextItemWidth(260);
    if (ImGui::InputTextWithHint("##eds", tr("Search tags"), a.ed_search, sizeof a.ed_search)) a.ed_page = 0;
    tip(tr("Show only tags containing these letters"));
    ImGui::SameLine();
    std::vector<const char*> cat_items{tr("All categories")};
    for (auto& c : cats) cat_items.push_back(c.c_str());
    ImGui::SetNextItemWidth(200);
    if (ImGui::Combo("##edc", &a.ed_cat, cat_items.data(), int(cat_items.size()))) a.ed_page = 0;
    tip(tr("Show only one category"));
    a.ed_cat = std::clamp(a.ed_cat, 0, int(cats.size()));
    std::string f = util::lower(util::trim(a.ed_search));
    std::vector<int> vis;
    for (int i = 0; i < int(tags.size()); i++) {
        if (a.ed_cat > 0 && tags[size_t(i)].category != cats[size_t(a.ed_cat - 1)]) continue;
        if (!f.empty() && util::lower(tags[size_t(i)].tag).find(f) == std::string::npos) continue;
        vis.push_back(i);
    }
    const int per = 15;
    int pages = std::max(1, (int(vis.size()) + per - 1) / per);
    a.ed_page = std::clamp(a.ed_page, 0, pages - 1);
    ImGui::SameLine();
    ImGui::TextDisabled("%zu / %zu", vis.size(), tags.size());
    // table
    int remove = -1;
    if (ImGui::BeginTable("edtags", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingFixedFit, ImVec2(0, per * 27.0f))) {
        ImGui::TableSetupColumn(tr("Tag"), ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(tr("Category"), 0, 170);
        ImGui::TableSetupColumn("", 0, 140);
        ImGui::TableHeadersRow();
        for (int k = a.ed_page * per; k < std::min(int(vis.size()), (a.ed_page + 1) * per); k++) {
            int i = vis[size_t(k)];
            ImGui::TableNextRow();
            ImGui::PushID(i);
            ImGui::TableNextColumn();
            if (a.ed_edit == i) ImGui::TextColored(g_accent, "%s", tags[size_t(i)].tag.c_str());
            else ImGui::TextUnformatted(tags[size_t(i)].tag.c_str());
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", tags[size_t(i)].category.c_str());
            ImGui::TableNextColumn();
            if (ImGui::SmallButton(tr("Edit"))) {
                a.ed_edit = i;
                snprintf(a.ed_name, sizeof a.ed_name, "%s", tags[size_t(i)].tag.c_str());
                a.ed_form_cat = int(std::find(cats.begin(), cats.end(), tags[size_t(i)].category) - cats.begin());
            }
            tip(tr("Rename this tag or move it to another category (below)"));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, g_warn_red);
            if (ImGui::SmallButton(tr("Delete"))) remove = i;
            ImGui::PopStyleColor();
            tip(tr("Delete and block this tag: photos lose it now, and it is never suggested or written into files again (unblock it below)."));
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (remove >= 0) {
        set_tag_blocked(tags[size_t(remove)].tag, true);  // deleted = never again (also survives vocabulary upgrades)
        tags.erase(tags.begin() + remove);
        a.ed_dirty = true;
        if (a.ed_edit == remove) a.ed_edit = -1;
        else if (a.ed_edit > remove) a.ed_edit--;
    }
    // pager
    if (ImGui::ArrowButton("##prev", ImGuiDir_Left) && a.ed_page > 0) a.ed_page--;
    tip(tr("Previous page"));
    ImGui::SameLine();
    ImGui::Text("%s %d / %d", tr("page"), a.ed_page + 1, pages);
    ImGui::SameLine();
    if (ImGui::ArrowButton("##next", ImGuiDir_Right) && a.ed_page + 1 < pages) a.ed_page++;
    tip(tr("Next page"));
    // blocked tags: deleted ones, with a way back
    {
        std::set<std::string> blocked = blocked_tags();
        if (!blocked.empty()) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s (%zu):", tr("Blocked"), blocked.size());
            tip(tr("Deleted tags: never suggested, recognised or written into files. Click one to allow it again."));
            for (auto& b : blocked) {
                ImGui::SameLine();
                if (ImGui::SmallButton((b + "  " ICON_X).c_str())) {
                    set_tag_blocked(b, false);
                    a.toast = util::fmt("%s \"%s\"", tr("Unblocked"), b.c_str());
                    a.toast_until = glfwGetTime() + 2.5;
                }
                tip(tr("Allow this tag again (add it back to the list to have it recognised)"));
            }
        }
    }
    // add / edit form
    ImGui::SeparatorText(a.ed_edit >= 0 ? tr("Edit tag") : tr("Add a tag"));
    ImGui::SetNextItemWidth(260);
    ImGui::InputTextWithHint("##edname", tr("tag, e.g. golden retriever"), a.ed_name, sizeof a.ed_name);
    tip(tr("A short English phrase CLIP can recognise in a picture"));
    ImGui::SameLine();
    std::vector<const char*> form_cats;
    for (auto& c : cats) form_cats.push_back(c.c_str());
    form_cats.push_back(tr("new category..."));
    a.ed_form_cat = std::clamp(a.ed_form_cat, 0, int(form_cats.size()) - 1);
    ImGui::SetNextItemWidth(170);
    ImGui::Combo("##edfc", &a.ed_form_cat, form_cats.data(), int(form_cats.size()));
    tip(tr("The category it competes in"));
    bool newcat = a.ed_form_cat == int(cats.size());
    if (newcat) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(140);
        ImGui::InputTextWithHint("##ednc", tr("category name"), a.ed_newcat, sizeof a.ed_newcat);
        tip(tr("Name of the new category"));
    }
    ImGui::SameLine();
    std::string name = util::lower(util::trim(a.ed_name));
    std::string cat = newcat ? util::lower(util::trim(a.ed_newcat)) : (cats.empty() ? "object" : cats[size_t(a.ed_form_cat)]);
    bool dup = false;
    for (int i = 0; i < int(tags.size()); i++) dup |= i != a.ed_edit && util::lower(tags[size_t(i)].tag) == name;
    ImGui::BeginDisabled(name.empty() || cat.empty() || dup);
    if (ImGui::Button(a.ed_edit >= 0 ? tr("Save change") : tr("Add"))) {
        if (a.ed_edit >= 0) tags[size_t(a.ed_edit)] = {name, cat};
        else tags.push_back({name, cat});
        a.ed_dirty = true;
        a.ed_edit = -1;
        a.ed_name[0] = a.ed_newcat[0] = 0;
    }
    tip(tr("Put this tag into the list (saved when you press Save and re-tag)"));
    ImGui::EndDisabled();
    tip(dup ? tr("This tag is already in the list") : tr("Put this tag into the list (saved when you press Save)"));
    if (a.ed_edit >= 0) {
        ImGui::SameLine();
        if (ImGui::Button(tr("Cancel edit"))) { a.ed_edit = -1; a.ed_name[0] = 0; }
        tip(tr("Stop editing this tag"));
    }
    // footer
    ImGui::Separator();
    bool busy = a.pipe->running();
    ImGui::BeginDisabled(!a.ed_dirty || busy);
    if (primary_button(tr("Save and re-tag"), tr("Save the list and recompute every photo's tags from its stored analysis (seconds; no picture is read)."))) {
        if (save_tag_vocabulary(tags)) {
            RunOptions o = a.opts;
            o.stages = ST_TAG;
            o.scope = all_scope(a.cfg);
            o.clip_mode = o.tag_mode = 0;  // the new list makes every photo's tags outdated: all are re-tagged
            submit_task(a, "Re-tag with the new tag list", effective(a.cfg), o, ST_TAG, false, -1, false);
            a.toast = tr("Tag list saved; re-tagging the photos");
            a.toast_until = glfwGetTime() + 3;
            ImGui::CloseCurrentPopup();
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(a.ed_dirty ? tr("Close without saving") : tr("Close")) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    tip(a.ed_dirty ? tr("Forget the changes made here") : tr("Close the tag list"));
    ImGui::SameLine();
    ImGui::TextDisabled("%s", tag_vocabulary_path().c_str());
    ImGui::EndPopup();
}

// ---- CLIP tab: image analysis status and runs, and the smart update (what is worth updating, rules + jev).

static void run_tag(App& a, int clip_mode, int tag_mode) {
    RunOptions o = a.opts;
    o.stages = ST_TAG;
    o.scope = all_scope(a.cfg);
    o.clip_mode = clip_mode;
    o.tag_mode = tag_mode;
    submit_task(a, "Image recognition", effective(a.cfg), o, ST_TAG, false, -1, false);
}

static void start_smart(App& a) {
    a.smart = std::make_shared<App::SmartState>();
    a.smart_open = true;
    auto st = a.smart;
    Config c = a.cfg;
    Db* db = &a.ui_db;
    bool jev_up = a.jev_ok == 1 && c.jev_enabled;
    std::thread([st, c, db, jev_up] {
        std::string cur = clip_choose(c).id;
        std::vector<App::SmartItem> items;
        int asked = 0;
        for (auto& r : db->clip_rows(all_scope(c))) {
            if (r.user_tags) continue;  // your own tags are never second-guessed
            float best = 0;
            int sure = 0;
            std::string tl;
            for (auto& [t, cf] : r.tags) {
                best = std::max(best, cf);
                if (cf >= kTagWordMin) { sure++; tl += (tl.empty() ? "" : ", ") + t; }
            }
            App::SmartItem it;
            it.id = r.id;
            it.path = r.src_path;
            bool tiny = r.width > 0 && std::min(r.width, r.height) < 160;
            if (r.clip_model.empty()) {
                it.action = "reanalyse"; it.worth = tiny ? 0.3 : 1.0; it.by = "rules";
                it.reason = tiny ? "not analysed yet, but a tiny picture" : "not analysed yet";
            } else if (r.clip_model != cur) {
                it.action = "reanalyse";
                if (best >= 0.5 && sure >= 3 && !tiny) {
                    // Already well tagged by the other model: is the better model worth the time? A close call: jev.
                    it.reason = util::fmt("analysed with %s; tags already good (%s, best %.0f%%)", r.clip_model.c_str(), tl.c_str(), best * 100);
                    if (jev_up && asked < 80) {
                        asked++;
                        std::string d = "a photo whose tags " + tl + util::fmt(" (best %.0f%%) came from a fast, less accurate model", best * 100);
                        JevChoice jc = jev_choice(c, "Which statement is true?",
                                                  {{"yes", "Analysing " + d + " again with the slower, more accurate model is worth the time."},
                                                   {"no", "Analysing " + d + " again with the slower, more accurate model is not worth the time."}},
                                                  "{}");
                        if (jc.ok) { it.worth = jc.probs[0]; it.by = "jev"; }
                    }
                    if (it.by.empty()) { it.worth = 0.5; it.by = "rules"; }
                } else {
                    it.worth = tiny ? 0.3 : 0.85;
                    it.by = "rules";
                    it.reason = util::fmt("analysed with %s; tags weak or few (%d confident, best %.0f%%)", r.clip_model.c_str(), sure, best * 100);
                }
            } else if (sure < 2 || best < kTagWordMin) {
                // Same model: analysing again gives the same result. Re-tagging with the current list may help.
                it.action = "retag";
                it.worth = 0.6;
                it.by = "rules";
                it.reason = util::fmt("few or weak tags (%d confident, best %.0f%%): re-tag with the current list; if still empty, add tags yourself", sure, best * 100);
            } else {
                continue;  // up to date
            }
            it.tick = it.worth >= 0.6;
            items.push_back(std::move(it));
        }
        std::stable_sort(items.begin(), items.end(), [](auto& x, auto& y) { return x.worth > y.worth; });
        std::lock_guard<std::mutex> l(st->mu);
        st->items = std::move(items);
        st->note = asked ? util::fmt("jev judged %d close calls", asked) : (jev_up ? "" : "jev is down: close calls left at 50%");
        st->done = true;
        glfwPostEmptyEvent();
    }).detach();
}

static void draw_smart_modal(App& a) {
    if (a.smart_open) {
        ImGui::OpenPopup(tr("Smart update###smart"));
        a.smart_open = false;
    }
    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(tr("Smart update###smart"), nullptr)) return;
    auto st = a.smart;
    std::lock_guard<std::mutex> l(st->mu);
    ImGui::TextWrapped("%s", tr("Which photos are worth updating: not analysed yet, analysed with the quick model, or with weak or few tags. Rules decide the clear cases; jev decides the close calls (photos already well tagged by the quick model). Ticked at 60% or more."));
    if (!st->done) {
        ImGui::TextDisabled("%s", tr("Checking..."));
    } else {
        int re = 0, rt = 0;
        for (auto& it : st->items)
            if (it.tick) (it.action == "reanalyse" ? re : rt)++;
        if (!st->note.empty()) ImGui::TextDisabled("%s", st->note.c_str());
        if (ImGui::BeginTable("smart", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit,
                              ImVec2(0, ImGui::GetContentRegionAvail().y - 50))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("", 0, 24);
            ImGui::TableSetupColumn(tr("Worth it"), 0, 80);
            ImGui::TableSetupColumn(tr("Photo"), 0, 230);
            ImGui::TableSetupColumn(tr("Update"), 0, 90);
            ImGui::TableSetupColumn(tr("Why (who decided)"), ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            ImGuiListClipper clip;
            clip.Begin(int(st->items.size()));
            while (clip.Step())
                for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                    auto& it = st->items[size_t(i)];
                    ImGui::TableNextRow();
                    ImGui::PushID(i);
                    ImGui::TableNextColumn();
                    ImGui::Checkbox("##t", &it.tick);
                    tip(tr("Update this photo"));
                    ImGui::TableNextColumn();
                    ImGui::ProgressBar(float(it.worth), ImVec2(-1, 0), util::fmt("%.0f%%", it.worth * 100).c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(util::basename(it.path).c_str());
                    tip(it.path);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(it.action == "reanalyse" ? tr("re-analyse") : tr("re-tag"));
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted((it.reason + "  (" + it.by + ")").c_str());
                    ImGui::PopID();
                }
            ImGui::EndTable();
        }
        double per = (clip_choose(a.cfg).id.find("large") != std::string::npos ? 0.35 : 0.06) * std::max(1.0, 8.0 / std::max(1, a.cfg.clip_threads));
        ImGui::BeginDisabled(a.pipe->running() || re + rt == 0);
        if (primary_button(util::fmt("%s (%d %s, ~%s · %d %s)", tr("Update ticked"), re, tr("re-analyse"), fmt_duration(re * per).c_str(), rt, tr("re-tag")).c_str(),
                           tr("Re-analyse and re-tag the ticked photos. Nothing in your files changes."))) {
            RunOptions o = a.opts;
            o.stages = ST_TAG;
            o.scope = all_scope(a.cfg);
            for (auto& it : st->items)
                if (it.tick) (it.action == "reanalyse" ? o.reanalyse_ids : o.retag_ids).push_back(it.id);
            submit_task(a, "Smart update", effective(a.cfg), o, ST_TAG, false, -1, false);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
    }
    if (ImGui::Button(tr("Close")) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    tip(tr("Close without updating"));
    ImGui::EndPopup();
}

static void draw_clip_tab(App& a) {
    Config& c = a.cfg;
    bool busy = a.pipe->running();
    ClipInfo info = clip_choose(c);
    if (a.clip_counts_gen != a.view_gen) {  // refresh the counts when the catalog changed
        Config q = c;
        q.clip_model = "b32";
        a.clip_counts = a.ui_db.clip_counts(all_scope(c), info.id, tag_vocabulary_hash(info.id));
        a.clip_counts_quick = a.ui_db.clip_counts(all_scope(c), clip_choose(q).id, tag_vocabulary_hash(clip_choose(q).id));
        a.clip_counts_gen = a.view_gen;
    }
    const ClipCounts& k = a.clip_counts;
    bool large = info.id.find("large") != std::string::npos;
    double scale = std::max(1.0, 8.0 / std::max(1, c.clip_threads));
    double per = (large ? 0.35 : 0.06) * scale, per_quick = 0.06 * scale;
    std::string cd;
    {
        std::lock_guard<std::mutex> l(a.health_mu);
        cd = a.clip_detail;
    }
    ImGui::TextWrapped("%s", tr("CLIP reads each picture once (the slow part) and keeps the result. Tags and meaning search are worked out from it in seconds."));
    ImGui::Text("%s: %s  ·  %d %s  ·  %d %s  ·  %d %s  ·  %d %s", tr("Model"), info.label.c_str(), k.total, tr("photos"), k.total - k.no_vec, tr("analysed"),
                k.no_vec, tr("not analysed with this model"), k.few, tr("with few tags"));
    ImGui::TextDisabled("%s", cd.c_str());
    ImGui::Spacing();
    ImGui::BeginDisabled(busy || !c.clip_enabled || a.clip_ok == 0);
    ImGui::BeginDisabled(k.no_vec == 0);
    bool go_missing = primary_button(util::fmt("%s (%d, ~%s)", tr("Analyse missing"), k.no_vec, fmt_duration(k.no_vec * per).c_str()).c_str(),
                       tr("Analyse the photos not analysed with this model yet, then tag them"), ImVec2(260, 0));
    ImGui::EndDisabled();
    if (go_missing) run_tag(a, 0, 0);
    ImGui::SameLine();
    if (ImGui::Button(util::fmt("%s (%d, ~%s)", tr("Quick: small model"), a.clip_counts_quick.no_vec, fmt_duration(a.clip_counts_quick.no_vec * per_quick).c_str()).c_str(), ImVec2(260, 0)))
        run_tag(a, 2, 0);
    tip(tr("ViT-B/32 for the photos not analysed yet: about 6 times faster, less accurate. A later normal run upgrades them."));
    ImGui::SameLine();
    if (ImGui::Button(util::fmt("%s (%d, ~%s)", tr("Re-analyse all"), k.total, fmt_duration(k.total * per).c_str()).c_str(), ImVec2(260, 0))) run_tag(a, 1, 0);
    tip(tr("Read every picture again. Only needed after changing the model or if the analysis looks wrong."));
    if (ImGui::Button(util::fmt("%s (%d)", tr("Re-tag all"), k.total - k.no_vec).c_str(), ImVec2(260, 0))) run_tag(a, 3, 1);
    tip(tr("Recompute every photo's tags from its stored analysis (seconds; no picture is read). Your own tags are kept."));
    ImGui::SameLine();
    if (ImGui::Button(util::fmt("%s (%d)", tr("Re-tag photos with few tags"), k.few).c_str(), ImVec2(260, 0))) run_tag(a, 3, 2);
    tip(tr("Photos with fewer than 2 confident tags get another look with the current tag list"));
    ImGui::SameLine();
    bool jev_down = a.jev_ok != 1;
    if (ImGui::Button(tr("Smart update..."), ImVec2(260, 0))) start_smart(a);
    tip(jev_down ? tr("jev is down: the clear cases are still decided by the rules, and the close calls are left at 50%.")
                 : tr("Lists the photos worth updating, each with why: rules decide the clear cases, jev the close calls. You tick and run."));
    ImGui::EndDisabled();
    ImGui::SetNextItemWidth(200);
    ImGui::SliderInt(tr("CPU threads for image analysis"), &c.clip_threads, 1, 16);
    tip(tr("More threads analyse faster but leave less of the machine for other work"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Edit tag list..."))) a.tag_editor_open = true;
    tip(tr("The words CLIP chooses tags from: search, add, rename, move or delete them"));
}

static void draw_corr_confirm(App& a) {
    if (a.corr_confirm) {
        ImGui::OpenPopup(tr("Check corrections###corrc"));
        a.corr_confirm = false;
    }
    if (!ImGui::BeginPopupModal(tr("Check corrections###corrc"), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    int ai = 0, n = a.stats.total;
    for (auto& r : a.tag_rows) ai += !r.gen_tool.empty();
    bool llm = a.llm_ok == 1;
    ImGui::TextWrapped("%s", util::fmt(tr("Re-checks the generated tags and names of %d photos. The rules are instant; AI images (%d known) are also read by the LLM, which runs on the GPU server (about 1-3 s each)."), n, ai).c_str());
    if (!llm) ImGui::TextColored(g_warn, "%s", tr("The LLM is down: only the rules are checked."));
    if (primary_button(tr("Check now"), tr("Start the check; the proposals appear in Corrections"), ImVec2(140, 0))) {
        RunOptions o = a.opts;
        o.stages = ST_FIX;
        o.scope = all_scope(a.cfg);
        o.recheck_all = true;
        Config run = effective(a.cfg);
        if (!llm) run.llm_enabled = false;
        submit_task(a, "Check tags", run, o, ST_FIX, false, -1, false);
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    tip(tr("Do not check now"));
    ImGui::EndPopup();
}

// One line that says what each kind of data is and where it is handled.
// ===========================================================================
// Main window: sidebar | content | inspector, with one top bar.

static const char* section_title(const App& a) {
    switch (a.section) {
        case SEC_ALL: return tr("All photos");
        case SEC_FAV: return tr("Favorites");
        case SEC_MONTH: return a.month == "undated" ? tr("Undated") : a.month.c_str();
        case SEC_PLACE: return a.place.c_str();
        case SEC_AI: return tr("AIGC");
        case SEC_FOLDER: return a.section_folder.c_str();
        case SEC_DUPES: return tr("Deduplicate");
        case SEC_DATES: return tr("Fix dates");
        case SEC_SUGG: return tr("Set tags");
        case SEC_ORGANIZE: return tr("Organize path");
        case SEC_SAVE: return tr("Auto set meta");
        case SEC_TAGS: return tr("Set tags");
        case SEC_RECOG: return tr("Image recognition");
        case SEC_ACTIVITY: return tr("Activity");
    }
    return "";
}

static void set_section(App& a, int sec, const std::string& arg = "") {
    a.section = sec;
    a.fav_only = sec == SEC_FAV;
    a.ai_only = sec == SEC_AI;
    a.month = sec == SEC_MONTH ? arg : "";
    a.place = sec == SEC_PLACE ? arg : "";
    a.section_folder = sec == SEC_FOLDER ? arg : "";
    a.review_only = false;
    a.selected = 0;
    a.multi.clear();
    a.rows_dirty = a.dups_dirty = true;
    if (sec == SEC_TAGS) a.on_tags = true;
}

// One sidebar row: icon, label, optional count on the right; returns true when clicked.
static bool side_item(App& a, const char* icon, const std::string& label, bool selected, int count = -1, bool count_accent = false, const char* tip_text = nullptr) {
    (void)a;
    ImGui::PushID(label.c_str());
    bool clicked = ImGui::Selectable("##side", selected, 0, ImVec2(0, ImGui::GetTextLineHeight() + 6));
    if (tip_text) tip(tip_text);
    ImVec2 p0 = ImGui::GetItemRectMin(), p1 = ImGui::GetItemRectMax();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float y = p0.y + 3;
    ImU32 col = ImGui::GetColorU32(ImGuiCol_Text), dim = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    dl->AddText(ImVec2(p0.x + 6, y), selected ? ImGui::GetColorU32(g_accent) : dim, icon);
    std::string shown = label;
    float lx = 12 + ImGui::GetFontSize() * 1.15f;  // label after the icon, at any text size
    float maxw = p1.x - p0.x - lx - 40;
    while (shown.size() > 4 && ImGui::CalcTextSize(shown.c_str()).x > maxw) shown = shown.substr(0, shown.size() - 4) + "…";
    dl->AddText(ImVec2(p0.x + lx, y), col, shown.c_str());
    if (count >= 0) {
        std::string n = count >= 10000 ? util::fmt("%.0fk", count / 1000.0) : std::to_string(count);
        ImVec2 ts = ImGui::CalcTextSize(n.c_str());
        dl->AddText(ImVec2(p1.x - ts.x - 6, y), count_accent && count > 0 ? ImGui::GetColorU32(g_accent) : dim, n.c_str());
    }
    ImGui::PopID();
    return clicked;
}

static void side_heading(const char* t) {
    ImGui::Dummy(ImVec2(0, 4));
    ImGui::TextDisabled("%s", t);
}

static void draw_sidebar(App& a) {
    const DbStats& st = a.stats;
    side_heading(tr("Library"));
    if (side_item(a, ICON_PHOTO, tr("All photos"), a.section == SEC_ALL, st.total, false, tr("Every photo in your folders"))) set_section(a, SEC_ALL);
    if (side_item(a, ICON_STAR, tr("Favorites"), a.section == SEC_FAV, int(a.favorites.size()), false, tr("Photos you starred (f)"))) set_section(a, SEC_FAV);
    if (a.ai_count && side_item(a, ICON_SPARKLES, tr("AIGC"), a.section == SEC_AI, a.ai_count, false, tr("AI-generated images: pictures with generation data (prompt, model) in the file")))
        set_section(a, SEC_AI);
    // Years, then their months
    static std::string open_year;
    std::map<std::string, std::pair<int, std::vector<const MonthInfo*>>> years;
    for (auto& mi : a.months) {
        std::string y = mi.month == "undated" ? std::string(tr("Undated")) : mi.month.substr(0, 4);
        years[y].first += mi.count;
        years[y].second.push_back(&mi);
    }
    if (!years.empty()) side_heading(tr("Years"));
    for (auto it = years.rbegin(); it != years.rend(); ++it) {
        const std::string& y = it->first;
        bool open = open_year == y;
        bool sel_year = a.section == SEC_MONTH && (a.month == y || util::starts_with(a.month, y + "-") || (y == tr("Undated") && a.month == "undated"));
        if (side_item(a, open ? ICON_CHEVRON_DOWN : ICON_CHEVRON_RIGHT, y, sel_year && !open, it->second.first)) {
            open_year = open ? "" : y;
            if (it->second.second.size() == 1 || y == tr("Undated")) set_section(a, SEC_MONTH, it->second.second[0]->month);
        }
        if (open_year == y)
            for (auto* mi : it->second.second) {
                ImGui::Indent(14);
                std::string label = mi->month == "undated" ? std::string(tr("Undated")) : mi->month;
                if (side_item(a, ICON_CALENDAR, label, a.section == SEC_MONTH && a.month == mi->month, mi->count)) set_section(a, SEC_MONTH, mi->month);
                ImGui::Unindent(14);
            }
    }
    if (!a.places.empty()) {
        side_heading(tr("Places"));
        std::vector<std::pair<std::string, int>> pl(a.places.begin(), a.places.end());
        std::sort(pl.begin(), pl.end(), [](auto& x, auto& y) { return x.second > y.second; });
        for (size_t i = 0; i < pl.size() && i < 8; i++)
            if (side_item(a, ICON_MAP_PIN, pl[i].first, a.section == SEC_PLACE && a.place == pl[i].first, pl[i].second)) set_section(a, SEC_PLACE, pl[i].first);
    }
    // What needs you
    int nd = 0;
    for (auto& r : a.dup_rows) nd += r.dup_of != 0;
    int pending = count_plan(a.plan, false);
    side_heading(tr("Tools"));
    if (side_item(a, ICON_COPY, tr("Deduplicate"), a.section == SEC_DUPES, nd, true, tr("Identical copies: keep one, move the others to the Trash"))) set_section(a, SEC_DUPES);
    if (side_item(a, ICON_CALENDAR_QUESTION, tr("Fix dates"), a.section == SEC_DATES, st.review + st.undated, true,
                  tr("Photos whose date is a guess or missing: confirm it, pick another, or accept the best guesses")))
        set_section(a, SEC_DATES);
    if (side_item(a, ICON_TAGS, tr("Set tags"), a.section == SEC_TAGS || a.section == SEC_SUGG, int(a.corrections.size()), true,
                  tr("Each photo's tags, your own tags, suggested fixes and the tag list")))
        set_section(a, SEC_TAGS);
    if (side_item(a, ICON_FOLDERS, tr("Organize path"), a.section == SEC_ORGANIZE, a.plan_op != OP_METADATA ? pending : -1, true,
                  tr("Copy, move or rename photos into month folders with date names")))
        set_section(a, SEC_ORGANIZE);
    if (side_item(a, ICON_FILE_EXPORT, tr("Auto set meta"), a.section == SEC_SAVE, a.plan_op == OP_METADATA ? pending : -1, true,
                  tr("All in one go: write missing dates, keywords, descriptions and places into the files")))
        set_section(a, SEC_SAVE);
    if (side_item(a, ICON_EYE, tr("Image recognition"), a.section == SEC_RECOG, -1, false, tr("Analyse pictures: new ones, again, or a quick pass")))
        set_section(a, SEC_RECOG);
    if (side_item(a, ICON_HISTORY, tr("Activity"), a.section == SEC_ACTIVITY, -1, false, tr("What ran, what changed (with Undo), and the decisions jev made")))
        set_section(a, SEC_ACTIVITY);
    side_heading(tr("Folders"));
    for (auto& f : std::vector<std::string>(a.cfg.folders)) {
        bool here = util::dir_exists(f);
        std::string ftip = here ? f : f + "\n" + tr("Not found: the drive may not be connected. Photos from it show their saved thumbnails; connect it to open them.");
        if (!here) ImGui::PushStyleColor(ImGuiCol_Text, g_warn);
        if (side_item(a, here ? ICON_FOLDER : ICON_ALERT_TRIANGLE, util::basename(f), a.section == SEC_FOLDER && a.section_folder == f, -1, false, ftip.c_str()))
            set_section(a, SEC_FOLDER, f);
        if (!here) ImGui::PopStyleColor();
        if (ImGui::BeginPopupContextItem(("fold" + f).c_str())) {
            if (ImGui::MenuItem(tr("Open in file manager"))) util::open_path(f);
            if (ImGui::MenuItem(tr("Remove from the list"))) {
                remove_folder(a.cfg, f);
                save_config(a.cfg, a.config_file);
                set_section(a, SEC_ALL);
            }
            tip(tr("No file is touched; its photos stay in the catalog"));
            ImGui::EndPopup();
        }
    }
    if (side_item(a, ICON_FOLDER_PLUS, tr("Add folder..."), false, -1, false, tr("Add a folder of photos; Analyze scans all your folders together"))) start_dir_dialog(a, 3);
}

// ---- search field: chips + text, suggestions while typing

struct PalItem {
    const char* icon;
    std::string label, where;
    std::function<void(App&)> go;
};
static std::vector<PalItem> palette_items(App& a);
static std::vector<const PalItem*> palette_match(const std::vector<PalItem>& items, const std::string& typed, size_t max_n);
static void ensure_settings_index(App& a);

struct Suggestion {
    const char* icon;
    std::string label, query, kind;
    std::function<void(App&)> go;  // a "Go to" entry: navigates instead of becoming a filter chip
};

static std::vector<Suggestion> suggestions(App& a, const std::string& typed_in) {
    std::vector<Suggestion> out;
    std::string t = util::lower(util::trim(typed_in));
    if (t.empty()) return out;
    if (a.sugg_tags_gen != a.view_gen) {  // tags seen in the catalog, plus the tag list
        std::set<std::string> all;
        for (auto& r : a.rows)
            for (auto& x : util::split(r.tags, ','))
                if (!util::trim(x).empty()) all.insert(util::lower(util::trim(x)));
        static std::vector<TagDef> vocab = load_tag_vocabulary();
        for (auto& v : vocab) all.insert(util::lower(v.tag));
        a.sugg_tags.assign(all.begin(), all.end());
        a.sugg_tags_gen = a.view_gen;
    }
    auto quoted = [](const std::string& v) { return v.find(' ') != std::string::npos ? "\"" + v + "\"" : v; };
    for (auto& tg : a.sugg_tags)
        if (out.size() < 5 && tg.find(t) != std::string::npos && !tag_blocked(tg)) out.push_back({ICON_TAG, tg, "tag:" + quoted(tg), tr("Tag")});
    for (auto& [p, n] : a.places)
        if (out.size() < 7 && util::lower(p).find(t) != std::string::npos) out.push_back({ICON_MAP_PIN, p, "place:" + quoted(util::lower(p)), tr("Place")});
    std::set<std::string> years;
    for (auto& mi : a.months)
        if (mi.month != "undated") years.insert(mi.month.substr(0, 4));
    for (auto& y : years)
        if (out.size() < 8 && util::starts_with(y, t)) out.push_back({ICON_CALENDAR, y, "date:" + y, tr("Year")});
    for (auto& mi : a.months)
        if (out.size() < 8 && t.size() >= 5 && util::starts_with(mi.month, t)) out.push_back({ICON_CALENDAR, mi.month, "date:" + mi.month, tr("Month")});
    if (std::string("favorites").find(t) == 0 || std::string("starred").find(t) == 0) out.push_back({ICON_STAR, tr("Favorites"), "is:fav", tr("Filter")});
    if (t == "ai" || t == "aigc" || std::string("generated").find(t) == 0) out.push_back({ICON_SPARKLES, tr("AIGC"), "is:ai", tr("Filter")});
    // Pages, settings and actions by name ("dedup", "theme", "screenshot"...): go there
    if (t.size() >= 3) {
        static std::vector<PalItem> items;
        static int built = -1;
        if (built != ImGui::GetFrameCount() / 120 || items.empty()) {  // rebuilt now and then (folders, undo change)
            ensure_settings_index(a);
            items = palette_items(a);
            built = ImGui::GetFrameCount() / 120;
        }
        for (auto* p : palette_match(items, t, 3)) out.push_back({p->icon, p->label, "", tr("Go to"), p->go});
    }
    return out;
}

static bool sentence_like(const std::string& s) {
    int words = 0;
    bool in = false;
    for (unsigned char ch : s) {
        bool sp = isspace(ch);
        if (!sp && !in) words++;
        in = !sp;
        if (ch == ':' || ch == '(' || ch == '|' || ch == '"') return false;  // query syntax
    }
    return words >= 4;
}

static void draw_search(App& a, float width) {
    ImGuiStyle& st = ImGui::GetStyle();
    float h = ImGui::GetFrameHeight();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + h), ImGui::GetColorU32(ImGuiCol_FrameBg), st.FrameRounding);
    ImGui::BeginGroup();
    ImGui::SetCursorScreenPos(ImVec2(p0.x + 8, p0.y + st.FramePadding.y));
    ImGui::TextDisabled(ICON_SEARCH);
    ImGui::SameLine(0, 6);
    // chips
    int remove = -1;
    for (size_t i = 0; i < a.tokens.size(); i++) {
        ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, p0.y + 2));
        ImGui::PushID(int(i));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.28f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, st.FramePadding.y - 2));
        if (ImGui::Button((a.tokens[i].first + "  " ICON_X).c_str())) remove = int(i);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        tip(tr("Remove this filter"));
        ImGui::PopID();
        ImGui::SameLine(0, 4);
    }
    if (remove >= 0) {
        a.tokens.erase(a.tokens.begin() + remove);
        a.rows_dirty = true;
    }
    float x = ImGui::GetCursorScreenPos().x;
    ImGui::SetCursorScreenPos(ImVec2(x, p0.y));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0, 0, 0, 0));
    ImGui::SetNextItemWidth(std::max(60.0f, p0.x + width - x - h - 6));  // room for the × at the end
    if (a.focus_search) {
        ImGui::SetKeyboardFocusHere();
        a.focus_search = false;
    }
    const char* hint = a.tokens.empty() ? tr("Search photos, or ask: my dog at the beach last summer") : "";
    bool changed = ImGui::InputTextWithHint("##q", hint, a.search, sizeof a.search);
    bool active = ImGui::IsItemActive();
    bool focused = ImGui::IsItemFocused() || active;
    ImVec2 in_min = ImGui::GetItemRectMin(), in_max = ImGui::GetItemRectMax();
    ImGui::PopStyleColor(3);
    tip(tr("Words, a description of the picture, or a sentence (press Enter to ask). Pick a suggestion to add it as a filter.\nPower syntax: beach | sea, -night, tag:dog, date:2019, size:>2mb, like:#12, ~typo (see ? > Search)."));
    if (changed) {
        a.rows_dirty = true;
        a.ask_go = false;
        a.search_edit_at = glfwGetTime();
        a.search_translate = false;
        a.sugg_sel = -1;
    }
    // clear button
    if (a.search[0] || !a.tokens.empty()) {
        float xb = h - 4;  // a square, frameless × inside the box's right end
        ImGui::SetCursorScreenPos(ImVec2(p0.x + width - xb - 4, p0.y + 2));
        ImGui::PushStyleColor(ImGuiCol_Text, g_text_dim);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        bool clr = ImGui::Button(ICON_X "##clr", ImVec2(xb, xb));
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        if (clr) {
            a.search[0] = 0;
            a.tokens.clear();
            a.ask_go = false;
            a.rows_dirty = true;
        }
        ImGui::PopStyleColor();
        tip(tr("Clear the search"));
    }
    ImGui::EndGroup();
    // keys: Enter = ask (a sentence) or take the highlighted suggestion; Backspace on empty removes the last chip
    std::vector<Suggestion> sg = focused ? suggestions(a, a.search) : std::vector<Suggestion>{};
    if (focused) {
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && !sg.empty()) a.sugg_sel = std::min(int(sg.size()) - 1, a.sugg_sel + 1);
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && !sg.empty()) a.sugg_sel = std::max(-1, a.sugg_sel - 1);
        if (ImGui::IsKeyPressed(ImGuiKey_Backspace) && !a.search[0] && !a.tokens.empty()) { a.tokens.pop_back(); a.rows_dirty = true; }
    }
    auto take = [&](const Suggestion& g) {
        if (g.go) {  // navigate, and leave the search as it was before typing
            auto go = g.go;
            a.search[0] = 0;
            a.sugg_sel = -1;
            go(a);
            return;
        }
        a.tokens.push_back({std::string(g.icon) + " " + g.label, g.query});
        a.search[0] = 0;
        a.sugg_sel = -1;
        a.rows_dirty = true;
    };
    if (ImGui::IsItemDeactivated() || (focused && (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)))) {
        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
            if (a.sugg_sel >= 0 && a.sugg_sel < int(sg.size())) take(sg[size_t(a.sugg_sel)]);
            else if (sentence_like(a.search) && a.llm_ok == 1 && a.cfg.llm_enabled) { a.ask_go = true; a.rows_dirty = true; }
            ImGui::SetKeyboardFocusHere(-1);
        }
    }
    // suggestion list under the field
    static bool hovered_last = false;
    if ((focused || hovered_last) && !sg.empty()) {
        ImGui::SetNextWindowPos(ImVec2(in_min.x, in_max.y + 4));
        ImGui::SetNextWindowSize(ImVec2(std::max(260.0f, width * 0.6f), 0));
        ImGui::Begin("##suggest", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                              ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_Tooltip);
        for (size_t i = 0; i < sg.size(); i++) {
            ImGui::PushID(int(i));
            if (ImGui::Selectable(util::fmt("%s  %s", sg[i].icon, sg[i].label.c_str()).c_str(), int(i) == a.sugg_sel)) take(sg[i]);
            ImGui::SameLine(ImGui::GetWindowWidth() - 70);
            ImGui::TextDisabled("%s", sg[i].kind.c_str());
            ImGui::PopID();
        }
        if (sentence_like(a.search) && a.llm_ok == 1) ImGui::TextDisabled("%s", tr("Enter: ask the assistant"));
        hovered_last = ImGui::IsWindowHovered();
        ImGui::End();
    } else {
        hovered_last = false;
    }
    // The box is `width` wide whatever is in it: the next item (the filter button) goes after it, never inside.
    ImGui::SetCursorScreenPos(p0);
    ImGui::Dummy(ImVec2(width, h));
}

// ---- top bar

// ---- Status · tasks · logs: the second button of the top-right cluster (Ctrl+J). The ring around it is the real
// overall progress of the queue, the badge counts waiting work; amber when paused, a red dot after errors.
static const char* stage_label(int stage) {
    static const int order[] = {ST_SCAN, ST_DUPES, ST_DECIDE, ST_TAG, ST_VISION, ST_FIX, ST_ORGANIZE};
    static const char* names[] = {"Scanning", "Duplicates", "Dates and places", "Recognising", "Describing", "Checking tags", "Organizing"};
    for (int k = 0; k < 7; k++)
        if (stage == order[k]) return names[k];
    return "Working";
}

static void draw_task_row(App& a, QTask& t, int idx, bool& changed) {
    Progress& p = a.pipe->progress;
    ImGui::PushID(t.id);
    const char* icon = t.state == TQ_RUNNING ? ICON_PLAY : t.state == TQ_PAUSED ? ICON_PAUSE : t.state == TQ_QUEUED ? ICON_CLOCK
                     : t.state == TQ_STOPPED ? ICON_PLAYER_STOP : t.errors ? ICON_ALERT_TRIANGLE : ICON_CIRCLE_CHECK;
    ImVec4 col = t.state == TQ_RUNNING ? g_accent : t.state == TQ_PAUSED ? g_warn : t.errors ? g_warn : t.state == TQ_DONE ? g_ok : g_text_dim;
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(col, "%s", icon);
    ImGui::SameLine();
    ImGui::TextUnformatted(tr(t.name.c_str()));
    float right = ImGui::GetContentRegionMax().x;
    auto act = [&](const char* ic, const char* tipt, float& x) {  // icon buttons from the right edge
        float w = ImGui::GetFrameHeight();
        x -= w + 2;
        ImGui::SameLine(x);
        bool r = ImGui::Button(ic, ImVec2(w, 0));
        tip(tipt);
        return r;
    };
    float x = right;
    if (t.state == TQ_RUNNING) {
        if (act(ICON_PLAYER_STOP, tr("Stop after the current photo; what is done is kept"), x)) { a.task_stop_as = TQ_STOPPED; a.pipe->stop(); }
        if (t.resumable && act(ICON_PAUSE, tr("Pause: stops now and waits in the queue; Resume continues where it stopped"), x)) { a.task_stop_as = TQ_PAUSED; a.pipe->stop(); }
    } else if (t.state == TQ_QUEUED || t.state == TQ_PAUSED) {
        if (act(ICON_X, tr("Cancel: take it out of the queue"), x)) { a.tasks.erase(a.tasks.begin() + idx); changed = true; ImGui::PopID(); return; }
        if (t.state == TQ_PAUSED && act(ICON_PLAY, tr("Resume"), x)) { t.state = TQ_QUEUED; changed = true; }
        if (t.state == TQ_QUEUED && t.resumable && act(ICON_PAUSE, tr("Hold: it waits until you resume it"), x)) t.state = TQ_PAUSED;
        if (idx + 1 < int(a.tasks.size()) && act(ICON_ARROW_DOWN, tr("Move down"), x)) { std::swap(a.tasks[size_t(idx)], a.tasks[size_t(idx + 1)]); ImGui::PopID(); return; }
        if (idx > 0 && a.tasks[size_t(idx - 1)].state != TQ_RUNNING && act(ICON_ARROW_UP, tr("Move up"), x)) {
            std::swap(a.tasks[size_t(idx)], a.tasks[size_t(idx - 1)]);
            ImGui::PopID();
            return;
        }
    }
    // second line: progress bar (running), or the state and numbers
    if (t.state == TQ_RUNNING) {
        int done = p.done, total = p.total;
        float frac = total > 0 ? float(done) / float(total) : 0.0f;
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - p.stage_start).count();
        double eta = done > 0 && total > done ? el / done * (total - done) : -1;
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, g_accent);
        ImGui::ProgressBar(total > 0 ? frac : -1.0f * float(ImGui::GetTime()), ImVec2(-1, 6), "");
        ImGui::PopStyleColor();
        ImGui::TextDisabled("%s  ·  %d / %d  ·  %.0f%%%s", tr(stage_label(p.stage)), done, total, frac * 100,
                            eta >= 0 ? (std::string("  ·  ") + fmt_eta(eta) + " " + tr("remaining")).c_str() : "");
    } else {
        const char* st = t.state == TQ_QUEUED ? tr("queued") : t.state == TQ_PAUSED ? tr("paused") : t.state == TQ_STOPPED ? tr("stopped")
                                                                                                   : t.errors ? tr("done with errors") : tr("done");
        std::string extra = t.total ? util::fmt("  ·  %d / %d", t.done, t.total) : "";
        if (t.t_end > 0) extra += "  ·  " + fmt_duration(t.t_end - t.t_start);
        if (!t.note.empty()) extra += "  ·  " + t.note;
        ImGui::TextDisabled("%s%s", st, extra.c_str());
    }
    ImGui::PopID();
}

static void draw_status_popover(App& a) {
    static int tab = 0;
    static bool lv[3] = {true, true, true};  // info, warning, error
    static char lq[96] = "";
    Progress& p = a.pipe->progress;
    // header: the app's state in one sentence
    int running = 0, queued = 0, paused = 0;
    for (auto& t : a.tasks) (t.state == TQ_RUNNING ? running : t.state == TQ_PAUSED ? paused : queued)++;
    std::string head = running ? util::fmt("%s %s", tr("Running:"), tr(a.tasks.front().name.c_str()))
                     : paused && !queued ? std::string(tr("Paused"))
                                         : std::string(tr("Ready"));
    if (queued) head += util::fmt("  ·  %d %s", queued, tr("queued"));
    if (paused) head += util::fmt("  ·  %d %s", paused, tr("paused"));
    ImGui::TextUnformatted(head.c_str());
    prefs::seg("statustab", &tab, {tr("Tasks"), tr("Logs")});
    ImGui::Separator();
    if (tab == 0) {
        ImGui::BeginDisabled(a.tasks.empty());
        if (ImGui::SmallButton((std::string(ICON_PAUSE " ") + tr("Pause all")).c_str())) {
            for (auto& t : a.tasks)
                if (t.state == TQ_QUEUED && t.resumable) t.state = TQ_PAUSED;
            if (a.pipe->running() && !a.tasks.empty() && a.tasks.front().resumable) { a.task_stop_as = TQ_PAUSED; a.pipe->stop(); }
        }
        tip(tr("Pause the running task and hold the queue"));
        ImGui::SameLine();
        if (ImGui::SmallButton((std::string(ICON_PLAY " ") + tr("Resume all")).c_str())) {
            for (auto& t : a.tasks)
                if (t.state == TQ_PAUSED) t.state = TQ_QUEUED;
            start_next_task(a);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton((std::string(ICON_X " ") + tr("Cancel all")).c_str())) {
            a.tasks.erase(std::remove_if(a.tasks.begin(), a.tasks.end(), [](const QTask& t) { return t.state != TQ_RUNNING; }), a.tasks.end());
            if (a.pipe->running()) { a.task_stop_as = TQ_STOPPED; a.pipe->stop(); }
        }
        tip(tr("Empty the queue and stop the running task (what is done is kept)"));
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(a.task_done.empty());
        if (ImGui::SmallButton(tr("Clear done"))) a.task_done.clear();
        ImGui::EndDisabled();
        ImGui::BeginChild("tasklist", ImVec2(0, std::min(360.0f, ImGui::GetTextLineHeightWithSpacing() * 2.6f * float(a.tasks.size() + a.task_done.size() + 1))));
        bool changed = false;
        if (a.tasks.empty() && a.task_done.empty()) ImGui::TextDisabled("%s", tr("Nothing has run yet."));
        for (int i = 0; i < int(a.tasks.size()); i++) draw_task_row(a, a.tasks[size_t(i)], i, changed);
        if (!a.task_done.empty()) ImGui::SeparatorText(tr("Done"));
        for (int i = 0; i < int(a.task_done.size()); i++) {
            QTask& t = a.task_done[size_t(i)];
            ImGui::PushID(1000000 + t.id);
            draw_task_row(a, t, -1, changed);
            ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::GetFrameHeight() * 2 - 4);
            if (ImGui::Button(ICON_RETRY, ImVec2(ImGui::GetFrameHeight(), 0))) {
                QTask copy = t;
                submit_task(a, copy.name, copy.cfg, copy.o, copy.last_stages, copy.planned, copy.plan_op, copy.show_plan);
            }
            tip(tr("Run it again"));
            ImGui::SameLine(0, 2);
            bool rm = ImGui::Button(ICON_X, ImVec2(ImGui::GetFrameHeight(), 0));
            tip(tr("Remove from the list"));
            ImGui::PopID();
            if (rm) { a.task_done.erase(a.task_done.begin() + i); break; }
        }
        if (changed) start_next_task(a);
        ImGui::EndChild();
        // helpers
        ImGui::SeparatorText(tr("Helpers"));
        std::string cd, ld, jd;
        {
            std::lock_guard<std::mutex> l(a.health_mu);
            cd = a.clip_detail;
            ld = a.llm_detail;
            jd = a.jev_detail;
        }
        auto hrow = [](int v, bool on, const char* name, const std::string& detail) {
            ImVec4 col = !on ? g_text_dim : v == 1 ? g_ok : v == 0 ? g_warn : g_text_dim;
            ImGui::TextColored(col, "%s", v == 1 && on ? ICON_CIRCLE_CHECK : ICON_CIRCLE_X);
            ImGui::SameLine();
            ImGui::Text("%s: %s", name, !on ? tr("off") : v == 1 ? tr("ready") : v == 0 ? tr("down") : tr("checking"));
            if (!detail.empty()) tip(detail.c_str());
        };
        hrow(a.clip_ok, a.cfg.clip_enabled, tr("Image recognition"), cd);
        hrow(a.llm_ok, a.cfg.llm_enabled, tr("Language model"), a.cfg.llm_url + "  " + ld);
        hrow(a.jev_ok, a.cfg.jev_enabled, tr("jev decisions"), a.cfg.jev_url + "  " + jd);
        if (ImGui::SmallButton((std::string(ICON_FLASK " ") + tr("Model tests...")).c_str())) { a.show_model_tests = true; ImGui::CloseCurrentPopup(); }
        tip(tr("Send your own input to each model and see the answer and how long it took"));
        ImGui::SameLine();
        if (ImGui::SmallButton((std::string(ICON_HISTORY " ") + tr("Open Activity")).c_str())) { set_section(a, SEC_ACTIVITY); ImGui::CloseCurrentPopup(); }
        tip(tr("The full log, what changed (with Undo), and the decisions jev made"));
    } else {
        const char* names[3] = {tr("Info"), tr("Warning"), tr("Error")};
        ImVec4 cols[3] = {ImGui::GetStyleColorVec4(ImGuiCol_Text), g_warn, g_err};
        for (int k = 2; k >= 0; k--) {  // level filter chips
            if (k != 2) ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, lv[k] ? ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.25f) : ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
            if (ImGui::SmallButton(names[k])) lv[k] = !lv[k];
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160);
        ImGui::InputTextWithHint("##lq", tr("filter"), lq, sizeof lq);
        std::vector<LogLine> lines = a.log.tail(2000);
        std::string ql = util::lower(lq), text;
        std::vector<const LogLine*> shown;
        for (auto it = lines.rbegin(); it != lines.rend(); ++it)  // newest first
            if (lv[std::clamp(it->level, 0, 2)] && (ql.empty() || util::lower(it->text).find(ql) != std::string::npos)) shown.push_back(&*it);
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_COPY)) {
            for (auto* l : shown) text += l->time + "  " + l->text + "\n";
            ImGui::SetClipboardText(text.c_str());
        }
        tip(tr("Copy these lines"));
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FOLDER)) util::open_path(util::dirname(a.log.file.empty() ? util::data_dir() + "/x" : a.log.file));
        tip(tr("Open the folder with the log file"));
        ImGui::BeginChild("loglines", ImVec2(0, 340), ImGuiChildFlags_Borders);
        ImGuiListClipper clip;
        clip.Begin(int(shown.size()));
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                const LogLine& l = *shown[size_t(i)];
                int lvl = std::clamp(l.level, 0, 2);
                ImGui::TextColored(cols[lvl], "%s", lvl == 2 ? ICON_CIRCLE_X : lvl == 1 ? ICON_ALERT_TRIANGLE : ICON_INFO_CIRCLE);
                ImGui::SameLine();
                ImGui::TextDisabled("%s", l.time.c_str());
                ImGui::SameLine();
                ImGui::TextUnformatted(l.text.substr(0, 160).c_str());
                if (l.text.size() > 160 || ImGui::IsItemHovered()) tip(l.text.c_str());
            }
        ImGui::EndChild();
    }
    (void)p;
}

static void draw_status_button(App& a, float bw) {
    int active = int(a.tasks.size());
    bool paused_only = active > 0 && !a.pipe->running();
    bool errors = false;
    for (auto& t : a.task_done) errors |= t.errors > 0 && t.t_end > a.status_seen_at;
    if (ImGui::Button(ICON_LIST_CHECK "##status", ImVec2(bw, 0)) || a.status_open_req) {
        ImGui::OpenPopup("status");
        a.status_open_req = false;
        a.status_seen_at = glfwGetTime();
    }
    ImVec2 b0 = ImGui::GetItemRectMin(), b1 = ImGui::GetItemRectMax();
    {
        std::string t = active ? util::fmt("%s: %d", tr("Tasks"), active) : std::string(tr("Tasks and logs"));
        tipk(t.c_str(), "Ctrl+J");
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 c((b0.x + b1.x) / 2, (b0.y + b1.y) / 2);
    float r = (b1.y - b0.y) / 2 + 1;
    if (active) {  // the ring: overall progress of the queue (real numbers, no endless spinner)
        Progress& p = a.pipe->progress;
        float cur = a.pipe->running() && p.total > 0 ? float(p.done) / float(p.total) : 0.0f;
        float overall = std::clamp((float(a.batch_done) + cur) / float(std::max(1, a.batch_total)), 0.0f, 1.0f);
        dl->PathArcTo(c, r, 0, 6.2832f, 40);
        dl->PathStroke(ImGui::GetColorU32(ImGuiCol_FrameBg), ImDrawFlags_Closed, 2.0f);
        if (overall > 0.001f) {
            dl->PathArcTo(c, r, -1.5708f, -1.5708f + 6.2832f * overall, 40);
            dl->PathStroke(ImGui::GetColorU32(paused_only ? g_warn : g_accent), 0, 2.0f);
        }
        std::string n = std::to_string(active);  // badge
        ImVec2 ts = ImGui::CalcTextSize(n.c_str());
        ImVec2 bc(b1.x - 2, b0.y + 2);
        float br = std::max(ts.x, ts.y) * 0.5f + 2;
        dl->AddCircleFilled(bc, br, ImGui::GetColorU32(paused_only ? g_warn : g_accent));
        ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 0.72f);
        ImVec2 ts2 = ImGui::CalcTextSize(n.c_str());
        dl->AddText(ImVec2(bc.x - ts2.x / 2, bc.y - ts2.y / 2), IM_COL32(255, 255, 255, 255), n.c_str());
        ImGui::PopFont();
    }
    if (errors) dl->AddCircleFilled(ImVec2(b0.x + 4, b0.y + 4), 3.5f, ImGui::GetColorU32(kDanger));
    if (ImGui::IsPopupOpen("status")) {  // only then: SetNextWindow* would otherwise move the next window drawn (the sidebar)
        ImGui::SetNextWindowPos(ImVec2(b1.x, b1.y + 6), ImGuiCond_Always, ImVec2(1, 0));
        ImGui::SetNextWindowSize(ImVec2(480, 0));
    }
    if (anim::begin_popup("status")) {
        draw_status_popover(a);
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
        anim::end_popup();
    }
}

static void draw_topbar(App& a) {
    bool busy = a.pipe->running();
    float full = ImGui::GetContentRegionAvail().x;
    ImGui::AlignTextToFramePadding();
    if (ImGui::Button(ICON_MENU_2)) a.sidebar_on = !a.sidebar_on;
    tip(tr("Show or hide the sidebar"));
    ImGui::SameLine(0, 10);
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.25f);
    std::string title = section_title(a);
    if (title.size() > 28) title = "…" + title.substr(title.size() - 26);
    ImGui::TextUnformatted(title.c_str());
    ImGui::PopFont();
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 12, full * 0.22f));
    float right_w = 330 + ImGui::GetFrameHeight() * 1.3f * (a.cfg.system_titlebar ? 4 : 7) + (a.cfg.system_titlebar ? 0 : 20);
    float sw = std::max(240.0f, full - ImGui::GetCursorPosX() - right_w);
    draw_search(a, sw);
    ImGui::SameLine(0, 6);
    if (ImGui::Button(ICON_ADJUSTMENTS_HORIZONTAL)) ImGui::OpenPopup("filters");
    tip(tr("Search options: which fields, a date range, regular expressions"));
    if (anim::begin_popup("filters")) {
        ImGui::TextDisabled("%s", tr("Look for words in"));
        if (ImGui::Checkbox(tr("File and folder names"), &a.in_name)) a.rows_dirty = true;
        if (ImGui::Checkbox(tr("Descriptions and tags"), &a.in_desc)) a.rows_dirty = true;
        if (ImGui::Checkbox(tr("Info inside files (EXIF / XMP)"), &a.in_exif)) a.rows_dirty = true;
        if (ImGui::Checkbox(tr("AI generation prompts"), &a.in_prompt)) a.rows_dirty = true;
        ImGui::Separator();
        ImGui::TextDisabled("%s", tr("Taken between"));
        if (date_picker("from", a.date_from, tr("any start"))) a.rows_dirty = true;
        ImGui::SameLine();
        ImGui::TextUnformatted(tr("and"));
        ImGui::SameLine();
        if (date_picker("to", a.date_to, tr("any end"))) a.rows_dirty = true;
        if (!a.date_from.empty() && !a.date_to.empty() && a.date_from > a.date_to) std::swap(a.date_from, a.date_to);
        ImGui::Separator();
        if (ImGui::Checkbox(tr("Match as a regular expression"), &a.regex)) a.rows_dirty = true;
        tip(tr("For experts: the search text is an ECMAScript regular expression over names, tags, prompts and metadata"));
        if (ImGui::Button(tr("Reset"))) {
            a.in_name = a.in_exif = a.in_desc = a.in_prompt = true;
            a.date_from.clear();
            a.date_to.clear();
            a.regex = false;
            a.rows_dirty = true;
        }
        tip(tr("Search everything, any date"));
        anim::end_popup();
    }
    ImGui::SameLine(0, 6);
    if (ImGui::Button(a.photo_grid ? ICON_LIST : ICON_LAYOUT_GRID)) a.photo_grid = !a.photo_grid;
    tip(a.photo_grid ? tr("Show as a list") : tr("Show as a grid of thumbnails"));
    ImGui::SameLine(0, 2);
    bool insp = a.inspector_on;
    if (insp) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.30f));
    if (ImGui::Button(ICON_PREVIEW_PANE)) a.inspector_on = !a.inspector_on;
    if (insp) ImGui::PopStyleColor();
    tip(tr("Show or hide the preview pane: the selected photo with its date, place, tags and file info (Ctrl+I)"));
    ImGui::SameLine(0, 12);
    {  // The feature button: its own icon, accent-tinted, one per view
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.16f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.28f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.40f));
        ImGui::PushStyleColor(ImGuiCol_Text, g_accent);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.55f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        if (ImGui::Button((std::string(ICON_FEATURE "  ") + tr("Analyze")).c_str())) request_analyze(a);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(5);
        tipk(busy ? tr("Scan your folders again when the running task is done (queued)")
                  : tr("Scan your folders for new or changed photos and update dates, tags and the lists to review. Nothing in your files changes."),
             "F5");
    }
    // The top-right corner: help and settings, then the window buttons at the very edge.
    float bw = ImGui::GetFrameHeight() * 1.3f, sp = 2, gap = 14;
    bool own = !a.cfg.system_titlebar;
    // Top right, the same in every view: [camera] | search · status/tasks/logs · help · settings | window buttons
    float corner = bw * 5 + sp * 4 + 10 + (own ? gap + bw * 3 + sp * 2 : 0);
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 12, ImGui::GetWindowContentRegionMax().x - corner));
    if (ImGui::Button(ICON_CAMERA, ImVec2(bw, 0))) ImGui::OpenPopup("shotmenu");
    tip(tr("Screenshot of this window, or of an area you select (saved in Pictures/jev-photos)"));
    if (anim::begin_popup("shotmenu")) {
        if (ImGui::MenuItem((std::string(ICON_SCREENSHOT "  ") + tr("Whole window")).c_str())) { a.shot_mode = 1; a.shot_frames = 2; }
        tip(tr("The whole window, as it is now"));
        if (ImGui::MenuItem((std::string(ICON_CROP "  ") + tr("Select an area...")).c_str())) { a.shot_mode = 2; a.shot_a = a.shot_b = ImVec2(-1, -1); }
        tip(tr("Drag a rectangle over the part you want; Esc cancels"));
        ImGui::BeginDisabled(a.shot_last.empty());
        if (ImGui::MenuItem((std::string(ICON_FOLDER "  ") + tr("Open the screenshots folder")).c_str())) util::open_path(util::dirname(a.shot_last));
        ImGui::EndDisabled();
        anim::end_popup();
    }
    ImGui::SameLine(0, 10);
    if (ImGui::Button(ICON_SEARCH "##palette", ImVec2(bw, 0))) a.palette_open = true;
    tipk(tr("Search: pages, settings, actions, help"), "Ctrl+P");
    ImGui::SameLine(0, sp);
    draw_status_button(a, bw);
    ImGui::SameLine(0, sp);
    if (ImGui::Button(ICON_HELP, ImVec2(bw, 0))) a.show_help = !a.show_help;
    tipk(tr("Help: concepts, glossary, shortcuts"), "F1");
    ImGui::SameLine(0, sp);
    if (ImGui::Button(ICON_SETTINGS, ImVec2(bw, 0))) a.show_settings = !a.show_settings;
    tipk(tr("Settings"), "Ctrl+,");
    if (own) {
        ImGui::SameLine(0, gap / 2);
        {  // 1 px divider between the cluster and the window buttons
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y + 4), ImVec2(p.x, p.y + ImGui::GetFrameHeight() - 4), ImGui::GetColorU32(ImGuiCol_Border));
        }
        ImGui::SameLine(0, gap / 2);
        if (ImGui::Button(ICON_MINUS, ImVec2(bw, 0))) glfwIconifyWindow(a.win);
        tip(tr("Minimise"));
        ImGui::SameLine(0, 2);
        bool maxed = glfwGetWindowAttrib(a.win, GLFW_MAXIMIZED);
        if (ImGui::Button(maxed ? ICON_WINDOW_MINIMIZE : ICON_SQUARE, ImVec2(bw, 0))) maxed ? glfwRestoreWindow(a.win) : glfwMaximizeWindow(a.win);
        tip(maxed ? tr("Restore the window size") : tr("Maximise (or double-click the top bar)"));
        ImGui::SameLine(0, 2);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(kDanger.x, kDanger.y, kDanger.z, 0.9f));
        if (ImGui::Button(ICON_X "##close", ImVec2(bw, 0))) glfwSetWindowShouldClose(a.win, GLFW_TRUE);
        ImGui::PopStyleColor();
        tip(tr("Close jev photos (a running step stops; everything done so far is kept)"));
    }
}

// ---- the library: a header with what is shown, then the grid or the list

static void draw_library_header(App& a) {
    std::string count;
    if (a.view.loading()) count = tr("searching...");
    else if (a.searching) count = util::fmt("%zu %s", a.rows.size(), tr("matches"));
    else count = util::fmt("%zu %s", a.rows.size(), tr("photos"));
    ImGui::TextDisabled("%s", count.c_str());
    if (a.ask_go) {
        ImGui::SameLine();
        ImGui::TextColored(g_accent, "%s %s", ICON_ROBOT, tr("asked the assistant"));
    }
    if (a.section == SEC_DATES) {
        // Every photo here is either dated by a guess or not dated at all; say how many of each, and offer what can be
        // done in one go: confirm the best guesses (all of them, or only the strong ones).
        int guessed = 0, strong = 0, undated = 0;
        for (auto& r : a.rows) {
            if (r.date_value.empty()) undated++;
            else {
                guessed++;
                strong += r.date_conf >= 0.4;
            }
        }
        auto accept = [&](double min_conf) {
            std::vector<std::tuple<int64_t, std::string, std::string>> todo;
            for (auto& r : a.rows)
                if (!r.date_value.empty() && r.date_conf >= min_conf) todo.push_back({r.id, r.date_value, r.date_prec});
            a.view.write([todo](Db& db) {
                for (auto& [id, v, p] : todo) db.confirm_date(id, v, p);
            });
            a.rows_dirty = true;
            a.catalog_gen++;
        };
        ImGui::SameLine(0, 20);
        ImGui::BeginDisabled(guessed == 0);
        if (primary_button(util::fmt("%s (%d)", tr("Accept all best guesses"), guessed).c_str(),
                           tr("Confirm the date shown for every photo here that has one; they leave this list. Photos without any date stay.")))
            accept(0);
        ImGui::EndDisabled();
        if (strong > 0 && strong < guessed) {
            ImGui::SameLine();
            if (ImGui::Button(util::fmt("%s (%d)", tr("Only the strong ones"), strong).c_str())) accept(0.4);
            tip(tr("Confirm only the guesses with good evidence (40% or more); check the others one by one"));
        }
        ImGui::TextDisabled("%s", util::fmt(tr("%d with a best guess (%d strong) · %d without a date: select one to pick its date in the inspector"),
                                            guessed, strong, undated).c_str());
    }
    if (!a.multi.empty() && a.multi.size() > 1) {
        ImGui::SameLine(0, 20);
        ImGui::Text("%zu %s", a.multi.size(), tr("selected"));
    }
    if (!ImGui::GetIO().WantTextInput && !a.viewer.open && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
        bool ctrl = ImGui::GetIO().KeyCtrl;
        if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || (!ctrl && ImGui::IsKeyPressed(ImGuiKey_D, false))) trash_selected(a);  // asks first
        if (ImGui::IsKeyPressed(ImGuiKey_F2, false) && a.selected) begin_rename(a, a.selected);
        if (!ctrl && ImGui::IsKeyPressed(ImGuiKey_C, false)) copy_paths(a);
        if (!ctrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) a.focus_search = true;
    }
    draw_fav_tag_chips(a);
    if (!a.search_error.empty()) ImGui::TextColored(g_err, "%s", a.search_error.c_str());
    else if (!a.search_note.empty()) ImGui::TextDisabled("%s", a.search_note.c_str());
    // A folder that is not there (unplugged drive): say so, instead of leaving blank tiles unexplained.
    static double checked = -10;
    static std::vector<std::string> away;
    if (glfwGetTime() - checked > 5) {
        checked = glfwGetTime();
        away.clear();
        for (auto& f : a.cfg.folders)
            if (!util::dir_exists(f) && (a.section != SEC_FOLDER || a.section_folder == f)) away.push_back(f);
    }
    for (auto& f : away)
        ImGui::TextColored(g_warn, "%s %s: %s", ICON_ALERT_TRIANGLE, tr("Folder not found (drive not connected?)"), f.c_str());
}

// Inspector extras for an unsure date: confirm it, or use one of the other candidates.
static void draw_date_actions(App& a, Photo& p) {
    if (!p.needs_review && !p.date_value.empty()) return;
    ImGui::Spacing();
    ImGui::TextColored(g_warn, "%s %s", ICON_CALENDAR_QUESTION, tr("This date is a guess"));
    if (!p.date_value.empty() && primary_button(tr("The date is right"), tr("Confirm this date; it is then treated as certain"))) {
        int64_t id = p.id;
        std::string v = p.date_value, pr = p.date_prec;
        a.view.write([id, v, pr](Db& db) { db.confirm_date(id, v, pr); });
        p.needs_review = false;
        a.rows_dirty = true;
        a.catalog_gen++;
    }
    json ev = json::parse(p.date_evidence.empty() ? "{}" : p.date_evidence, nullptr, false);
    if (!ev.is_object() || !ev.contains("candidates")) return;
    std::set<std::string> seen{p.date_value.substr(0, 16)};
    bool any = false;
    for (auto& c : ev["candidates"]) {
        std::string when = c.value("when", "");
        if (when.empty() || c.value("weight", 0.0) <= 0) continue;
        util::Civil cv = util::parse_exif_datetime(when);
        if (!cv.valid() || !seen.insert(cv.str().substr(0, 16)).second) continue;
        if (!any) { ImGui::TextDisabled("%s", tr("Or use another date found for it:")); any = true; }
        ImGui::PushID(when.c_str());
        if (ImGui::SmallButton(util::fmt("%s  (%s)", when.c_str(), c.value("src", "").c_str()).c_str())) {
            int64_t id = p.id;
            std::string v = cv.str(), pr = util::precision_name(cv.prec);
            a.view.write([id, v, pr](Db& db) { db.confirm_date(id, v, pr); });
            p.date_value = v;
            p.needs_review = false;
            a.rows_dirty = true;
            a.catalog_gen++;
        }
        tip(tr("Use this date for the photo"));
        ImGui::PopID();
    }
}

static void draw_library(App& a) {
    if (a.stats.total == 0 && !a.pipe->running()) {
        if (a.view.loading()) { ImGui::TextDisabled("%s", tr("Loading...")); return; }
        ImGui::Dummy(ImVec2(0, 40));
        ImGui::TextWrapped("%s", tr("No photos here yet. Analyze reads your folders; browsing and search work while it runs."));
        if (primary_button(tr("Analyze"), tr("Scan your folders"))) request_analyze(a);
        return;
    }
    draw_library_header(a);
    // While Ctrl is held the wheel resizes the thumbnails instead of scrolling.
    ImGui::BeginChild("libcontent", ImVec2(0, 0), 0, ImGui::GetIO().KeyCtrl && a.photo_grid ? ImGuiWindowFlags_NoScrollWithMouse : 0);
    if (a.photo_grid) draw_photo_grid(a);
    else draw_photo_table(a);
    ImGui::EndChild();
}

// ---- first run: folders, what the app may do, start

static void draw_first_run(App& a) {
    float w = std::min(620.0f, ImGui::GetContentRegionAvail().x - 40);
    ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - w) / 2);
    ImGui::BeginChild("welcome", ImVec2(w, 0));
    ImGui::Dummy(ImVec2(0, 30));
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.6f);
    ImGui::TextUnformatted(tr("Welcome to jev photos"));
    ImGui::PopFont();
    ImGui::TextWrapped("%s", tr("Find any photo by what is in it, fix missing dates, clear out duplicates, and organize your library. Your originals are only changed if you choose to."));
    ImGui::Dummy(ImVec2(0, 12));
    ImGui::TextColored(g_accent, "1");
    ImGui::SameLine();
    ImGui::TextUnformatted(tr("Add your photo folders"));
    for (auto& f : a.cfg.folders) ImGui::BulletText("%s", f.c_str());
    if (ImGui::Button(util::fmt("%s %s", ICON_FOLDER_PLUS, tr("Add folder...")).c_str())) start_dir_dialog(a, 3);
    tip(tr("Choose a folder; add as many as you like"));
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::TextColored(g_accent, "2");
    ImGui::SameLine();
    ImGui::TextUnformatted(tr("What may jev photos do with your files?"));
    static int goal = 0;
    ImGui::RadioButton(tr("Only browse and search (nothing in my folders changes)"), &goal, 0);
    tip(tr("The catalog lives in the app's own data folder. You can organize later."));
    ImGui::RadioButton(tr("Also make organized copies with date names"), &goal, 1);
    tip(tr("Copies go to a jev-organized folder inside each photo folder; your originals stay as they are."));
    ImGui::PushStyleColor(ImGuiCol_Text, g_warn_red);
    ImGui::RadioButton(tr("Rename my originals with date names"), &goal, 2);
    ImGui::PopStyleColor();
    tip(tr("Changes your original files' names (you can undo it from Activity)."));
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::TextColored(g_accent, "3");
    ImGui::SameLine();
    ImGui::TextUnformatted(tr("Start"));
    ImGui::TextDisabled("%s", tr("Analysis runs in the background; you can browse and search right away."));
    ImGui::BeginDisabled(a.cfg.folders.empty());
    if (primary_button(tr("Start"), tr("Save these choices and analyze your folders"), ImVec2(160, 0))) {
        a.cfg.file_op = goal == 2 ? OP_RENAME : OP_COPY;
        a.cfg.goal = goal;
        a.cfg.first_run_done = true;
        save_config(a.cfg, a.config_file);
        a.opts.clip_mode = a.opts.tag_mode = 0;
        { start_run(a, analyze_stages(a.cfg)); a.auto_meta_step = a.cfg.auto_meta ? -1 : 0; }
        set_section(a, SEC_ALL);
    }
    ImGui::EndDisabled();
    ImGui::EndChild();
}

// ---- Settings in their own window

static void draw_settings_window(App& a) {
    if (!a.show_settings) return;
    ImGui::SetNextWindowSize(ImVec2(820, 640), ImGuiCond_FirstUseEver);
    ImVec2 vs = ImGui::GetMainViewport()->Size;
    ImGui::SetNextWindowPos(ImVec2(vs.x / 2, vs.y / 2), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(600, 360), ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::Begin(tr("Settings###settingswin"), &a.show_settings, ImGuiWindowFlags_NoCollapse)) {
        draw_settings(a);
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) &&
            ImGui::IsKeyPressed(ImGuiKey_Escape, false))
            a.show_settings = false;  // Esc closes, like a native preferences window
    }
    ImGui::End();
}

// ---- Activity: what ran, Undo, jev's decisions

static void draw_history(App& a);
static void draw_activity(App& a) {
    ImGui::TextWrapped("%s", tr("Everything that changed your files is listed here and can be undone, newest first."));
    draw_history(a);
    ImGui::SeparatorText(tr("How jev helped"));
    const DecisionStats& ds = a.decision_stats;
    if (!ds.total) ImGui::TextDisabled("%s", tr("No jev decisions yet."));
    else {
        auto kind = [&](const char* k, const char* label) {
            auto it = ds.by_kind.find(k);
            return util::fmt("%s %d", label, it == ds.by_kind.end() ? 0 : it->second.first);
        };
        ImGui::Text("%d %s · %s · %s · %s · %s · %d %s", ds.total, tr("decisions"), kind("date", tr("dates")).c_str(), kind("place", tr("places")).c_str(),
                    kind("search", tr("search")).c_str(), kind("description", tr("descriptions")).c_str(), ds.changed, tr("changed the result"));
    }
    if (ImGui::BeginTable("decisions2", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0, 240))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(tr("Kind"), 0, 80);
        ImGui::TableSetupColumn(tr("About"), 0, 300);
        ImGui::TableSetupColumn(tr("Rules / LLM said"), 0, 200);
        ImGui::TableSetupColumn(tr("jev chose"), 0, 220);
        ImGui::TableSetupColumn(tr("Result"), 0, 150);
        ImGui::TableHeadersRow();
        for (auto& d : a.decisions) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(tr(d.kind.c_str()));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted((d.kind == "search" ? "\"" + d.subject + "\"  " + d.question : rel_path(a.cfg, d.subject)).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(d.rules_pick.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(d.jev_pick.substr(0, 70).c_str());
            ImGui::TableNextColumn();
            if (d.changed) ImGui::TextColored(g_warn, "%s", d.final_pick.c_str());
            else ImGui::TextUnformatted(d.final_pick.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::SeparatorText(tr("Log"));
    ImGui::BeginChild("logpane", ImVec2(0, 0), ImGuiChildFlags_Borders);
    for (auto& l : a.log.tail(400)) {
        ImVec4 col = l.level == 2 ? g_err : l.level == 1 ? g_warn : ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImGui::TextColored(col, "%s  %s", l.time.c_str(), l.text.c_str());
    }
    ImGui::EndChild();
}

// ---- Undo: every applied run is journaled; undo puts it back, newest change first

static void start_undo(App& a, const std::string& run) {
    RunOptions o = a.opts;
    o.stages = 0;
    o.undo_run = run;
    a.undo_until = 0;
    submit_task(a, "Undo", effective(a.cfg), o, 0, false, -1, false);
    a.log.add(0, "undo of the run applied " + run);
}

static void draw_history(App& a) {
    static std::vector<JournalRun> runs;
    static int runs_gen = -1;
    if (runs_gen != a.catalog_gen) {
        runs = a.ui_db.journal_runs(10);
        runs_gen = a.catalog_gen;
    }
    if (runs.empty()) {
        ImGui::TextDisabled("%s", tr("Nothing applied yet."));
        return;
    }
    if (ImGui::BeginTable("history", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn(tr("When"), 0, 170);
        ImGui::TableSetupColumn(tr("What"), ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(tr("Changes"), 0, 70);
        ImGui::TableSetupColumn("", 0, 110);
        ImGui::TableHeadersRow();
        for (auto& r : runs) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.at.substr(0, 19).c_str());
            ImGui::TableNextColumn();
            std::string what;
            for (auto& k : util::split(r.kinds, ',')) {
                const char* w = k == "copy" ? tr("copied") : k == "move" ? tr("moved") : k == "rename" ? tr("renamed") : k == "refile" ? tr("re-filed")
                              : k == "meta" ? tr("info saved in files") : k.c_str();
                what += (what.empty() ? "" : ", ") + std::string(w);
            }
            ImGui::TextUnformatted(what.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", r.count);
            ImGui::TableNextColumn();
            ImGui::PushID(r.run.c_str());
            if (r.undone) ImGui::TextDisabled("%s", tr("undone"));
            else {
                ImGui::BeginDisabled(a.pipe->running());
                if (ImGui::SmallButton(util::fmt("%s %s", ICON_ARROW_BACK_UP, tr("Undo")).c_str())) start_undo(a, r.run);
                ImGui::EndDisabled();
                tip(tr("Put everything this run changed back: files return to their old names and places, copies go to the Trash, files whose info was written get their earlier version back"));
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

// Choosing a screenshot area: everything outside the rectangle is dimmed; drag, release to take it, Esc to cancel.
static void draw_shot_overlay(App& a) {
    if (a.shot_mode != 2) return;
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::SetNextWindowFocus();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##shotarea", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoMove);
    ImGui::PopStyleVar(2);
    ImGui::InvisibleButton("##drag", vp->Size);
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsItemActivated()) a.shot_a = a.shot_b = io.MousePos;
    if (ImGui::IsItemActive()) a.shot_b = io.MousePos;
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImU32 dim = IM_COL32(0, 0, 0, 110);
    ImVec2 lo(std::min(a.shot_a.x, a.shot_b.x), std::min(a.shot_a.y, a.shot_b.y)), hi(std::max(a.shot_a.x, a.shot_b.x), std::max(a.shot_a.y, a.shot_b.y));
    ImVec2 v0 = vp->Pos, v1(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);
    if (a.shot_a.x < 0) dl->AddRectFilled(v0, v1, dim);
    else {  // dim around the rectangle
        dl->AddRectFilled(v0, ImVec2(v1.x, lo.y), dim);
        dl->AddRectFilled(ImVec2(v0.x, hi.y), v1, dim);
        dl->AddRectFilled(ImVec2(v0.x, lo.y), ImVec2(lo.x, hi.y), dim);
        dl->AddRectFilled(ImVec2(hi.x, lo.y), ImVec2(v1.x, hi.y), dim);
        dl->AddRect(lo, hi, ImGui::GetColorU32(g_accent), 0.0f, ImDrawFlags_None, 2.0f);
        std::string sz = util::fmt("%.0f x %.0f", (hi.x - lo.x) * io.DisplayFramebufferScale.x, (hi.y - lo.y) * io.DisplayFramebufferScale.y);
        dl->AddText(ImVec2(lo.x + 4, std::max(v0.y, lo.y - ImGui::GetTextLineHeight() - 4)), IM_COL32(255, 255, 255, 255), sz.c_str());
    }
    const char* hint = tr("Drag over the area to capture  ·  Esc cancels");
    ImVec2 ts = ImGui::CalcTextSize(hint);
    ImVec2 h0((v0.x + v1.x - ts.x) / 2 - 12, v0.y + 14);
    dl->AddRectFilled(h0, ImVec2(h0.x + ts.x + 24, h0.y + ts.y + 12), IM_COL32(20, 20, 24, 220), 8);
    dl->AddText(ImVec2(h0.x + 12, h0.y + 6), IM_COL32(255, 255, 255, 255), hint);
    if (ImGui::IsItemDeactivated()) {
        if (hi.x - lo.x >= 4 && hi.y - lo.y >= 4) { a.shot_mode = 3; a.shot_frames = 2; a.shot_a = lo; a.shot_b = hi; }
        else a.shot_mode = 0;  // a click is not an area
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) a.shot_mode = 0;
    ImGui::End();
}

static void draw_undo_banner(App& a) {
    static std::string shown_run;
    static double shown_at = 0;
    if (a.undo_run.empty() || glfwGetTime() > a.undo_until || a.pipe->running()) { shown_run.clear(); return; }
    if (shown_run != a.undo_run) { shown_run = a.undo_run; shown_at = glfwGetTime(); }
    // State: slides up from the bottom edge and fades in (220 ms ease-out); leaves the way it came (160 ms).
    float in = anim::progress(shown_at, 0.22f);
    float left = float(a.undo_until - glfwGetTime());
    float out = left < 0.16f ? anim::ease_out(1 - left / 0.16f) : 0.0f;
    if (out > 0) anim::keep_drawing();
    float dy = anim::g_reduce ? 0.0f : (1 - in) * 16 + out * 16;
    ImVec2 vs = ImGui::GetMainViewport()->Size, vp = ImGui::GetMainViewport()->Pos;
    ImGui::SetNextWindowPos(ImVec2(vp.x + vs.x / 2, vp.y + vs.y - 24 + dy), 0, ImVec2(0.5f, 1));
    ImGui::SetNextWindowBgAlpha(0.95f);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.01f, in * (1 - out)));
    ImGui::Begin("##undo", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(g_ok, ICON_CIRCLE_CHECK);
    ImGui::SameLine();
    ImGui::TextUnformatted(a.undo_text.c_str());
    ImGui::SameLine(0, 16);
    if (ImGui::Button(util::fmt("%s %s", ICON_ARROW_BACK_UP, tr("Undo")).c_str())) start_undo(a, a.undo_run);
    tip(tr("Put it all back (Ctrl+Z). Later runs can be undone from Activity."));
    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_X)) a.undo_until = std::min(a.undo_until, glfwGetTime() + 0.16);  // leave the way it came
    ImGui::End();
    ImGui::PopStyleVar();
}

// ---- the window

// ---- Model tests (for development): try each model with your own input, see the raw answer and how long it took

struct ModelTest {
    std::mutex mu;
    std::atomic<bool> busy{false};
    std::string out, err;
    double secs = 0;
};

static void draw_model_tests(App& a) {
    if (!a.show_model_tests) return;
    static int tab = 0;
    static char prompt[2048] = "Reply with one word: are you ready?";
    static int max_tokens = 64;
    static bool json_reply = false;
    static char query[256] = "a photo of a dog";
    static char jev_q[512] = "Which of these is a date a camera would write?";
    static char jev_opts[1024] = "2019:05:12 14:30:22\n1970:01:01 00:00:00\n2099:12:31 23:59:59";
    static auto st = std::make_shared<ModelTest>();
    static ClipModel* clip = nullptr;  // loaded once, kept for the next test
    static TagIndex clip_idx;
    static std::mutex clip_mu;

    ImGui::SetNextWindowSize(ImVec2(760, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(560, 360), ImVec2(FLT_MAX, FLT_MAX));
    if (!ImGui::Begin(tr("Model tests###modeltests"), &a.show_model_tests)) { ImGui::End(); return; }
    ImGui::TextDisabled("%s", tr("For development: send your own input to each model and see the raw answer and the time it took."));
    prefs::seg("mt_tab", &tab, {tr("Language model"), tr("Vision model"), tr("Image recognition"), "jev"});
    ImGui::Spacing();
    std::string photo;
    if (a.sel_loaded && a.sel.id == a.selected) photo = a.sel.src_path;
    auto run = [&](std::function<std::string(std::string&)> fn) {
        if (st->busy.exchange(true)) return;
        {
            std::lock_guard<std::mutex> l(st->mu);
            st->out.clear();
            st->err.clear();
        }
        std::thread([fn, s = st] {
            auto t0 = std::chrono::steady_clock::now();
            std::string err, out = fn(err);
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            {
                std::lock_guard<std::mutex> l(s->mu);
                s->out = out;
                s->err = err;
                s->secs = secs;
            }
            s->busy = false;
            glfwPostEmptyEvent();
        }).detach();
    };
    Config c = effective(a.cfg);
    if (tab == 0) {
        ImGui::TextDisabled("%s  ·  %s", c.llm_url.c_str(), c.llm_model.empty() ? tr("(server default model)") : c.llm_model.c_str());
        ImGui::InputTextMultiline("##prompt", prompt, sizeof prompt, ImVec2(-1, ImGui::GetTextLineHeight() * 5));
        ImGui::SetNextItemWidth(140);
        ImGui::InputInt(tr("Max tokens"), &max_tokens, 16);
        max_tokens = std::clamp(max_tokens, 1, 8192);
        ImGui::SameLine(0, 20);
        ImGui::Checkbox(tr("JSON answer"), &json_reply);
        ImGui::BeginDisabled(st->busy);
        if (primary_button(tr("Send"), tr("Ask the language model (thinking off when that is set in Settings)"))) {
            std::string p = prompt;
            int mt = max_tokens;
            bool js = json_reply;
            run([c, p, mt, js](std::string& err) { return llm_ask(c, p, mt, js, err); });
        }
        ImGui::EndDisabled();
    } else if (tab == 1) {
        ImGui::TextDisabled("%s  ·  %s", c.vl_url.c_str(), c.vl_model.c_str());
        if (!c.vl_enabled) ImGui::TextColored(g_warn, "%s", tr("The vision model is off in Settings (Advanced); the test still sends the request."));
        ImGui::TextWrapped("%s %s", tr("Photo:"), photo.empty() ? tr("select a photo in the library first") : photo.c_str());
        ImGui::BeginDisabled(st->busy || photo.empty());
        if (primary_button(tr("Describe the selected photo"), tr("Send the photo to the vision model and show what it answers"))) {
            int orient = meta_orientation(MetaMap::from_json(a.sel.meta_json));
            run([c, photo, orient](std::string& err) {
                std::string jpeg;
                if (!prepare_image(photo, orient, c.vl_max_side, jpeg, err)) return std::string();
                VisionResult v = vision_describe(c, jpeg);
                if (!v.ok) { err = v.error; return v.raw; }
                std::string o = "caption: " + v.caption + "\nscene: " + v.scene + "\nlandmark: " + v.landmark + "\ntext: " + v.text + "\nobjects: ";
                for (auto& x : v.objects) o += x + ", ";
                o += "\ntags: ";
                for (auto& x : v.tags) o += x + ", ";
                return o + "\n\nraw:\n" + v.raw;
            });
        }
        ImGui::EndDisabled();
    } else if (tab == 2) {
        ClipInfo info = clip_choose(c);
        ImGui::TextDisabled("%s  ·  %s", info.label.c_str(), info.dir.c_str());
        ImGui::TextWrapped("%s %s", tr("Photo:"), photo.empty() ? tr("select a photo in the library first") : photo.c_str());
        ImGui::SetNextItemWidth(320);
        ImGui::InputTextWithHint("##q", tr("a text to compare, e.g. a photo of a dog"), query, sizeof query);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", tr("(optional: its similarity to the photo)"));
        ImGui::BeginDisabled(st->busy || photo.empty() || !clip_files_present(info.dir));
        if (primary_button(tr("Tag the selected photo"), tr("Run image recognition on the photo: tags, scene and timings"))) {
            std::string q = query;
            int maxt = c.clip_max_tags;
            run([c, info, photo, q, maxt](std::string& err) {
                std::lock_guard<std::mutex> l(clip_mu);
                std::string o;
                auto t0 = std::chrono::steady_clock::now();
                if (!clip || clip->info().id != info.id) {
                    delete clip;
                    clip = new ClipModel();
                    if (!clip->load(info, c.clip_device, c.clip_threads, true, true, err)) { delete clip; clip = nullptr; return o; }
                    if (!build_tag_index(*clip, clip_idx, err)) { delete clip; clip = nullptr; return o; }
                    o += util::fmt("loaded %s on %s in %.1f s (%zu tags)\n", info.label.c_str(), clip->device_used().c_str(),
                                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), clip_idx.tags.size());
                }
                auto t1 = std::chrono::steady_clock::now();
                std::vector<float> chw;
                std::vector<Vec> emb;
                if (!clip_preprocess(photo, chw, err) || !clip->encode_images({chw}, emb, err)) return o;
                ImageTags t = pick_tags(clip_idx, emb[0], maxt);
                o += util::fmt("image: %.0f ms\nscene: %s\ntags:\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() * 1000,
                               t.scene.c_str());
                for (auto& [tag, p] : t.tags) o += util::fmt("  %-28s %3.0f%%\n", tag.c_str(), p * 100);
                if (!util::trim(q).empty()) {
                    Vec tv;
                    if (clip->encode_text(q, tv, err)) o += util::fmt("\n\"%s\": cosine %.3f\n", q.c_str(), dot(emb[0], tv));
                }
                return o;
            });
        }
        ImGui::EndDisabled();
        if (!clip_files_present(info.dir)) ImGui::TextColored(g_warn, "%s", tr("Model not downloaded: Settings > Image recognition > Models."));
    } else {
        ImGui::TextDisabled("%s  ·  %s", c.jev_url.c_str(), c.jev_model.c_str());
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##jq", tr("the question"), jev_q, sizeof jev_q);
        ImGui::TextDisabled("%s", tr("Options, one per line (each should make sense on its own):"));
        ImGui::InputTextMultiline("##jo", jev_opts, sizeof jev_opts, ImVec2(-1, ImGui::GetTextLineHeight() * 5));
        ImGui::BeginDisabled(st->busy);
        if (primary_button(tr("Ask jev"), tr("Send the question and options; see its choice and probabilities"))) {
            std::string q = jev_q;
            std::vector<std::pair<std::string, std::string>> opts;
            int k = 0;
            for (auto& line : util::split(jev_opts, '\n'))
                if (!util::trim(line).empty()) opts.push_back({util::fmt("o%d", ++k), util::trim(line)});
            run([c, q, opts](std::string& err) {
                JevChoice r = jev_choice(c, q, opts, "{}");
                if (!r.ok) { err = r.error; return std::string(); }
                std::string o = "choice: " + r.choice + "\n";
                for (size_t i = 0; i < opts.size(); i++)
                    o += util::fmt("  %5.1f%%  %s\n", i < r.probs.size() ? r.probs[i] * 100 : 0.0, opts[i].second.c_str());
                return o;
            });
        }
        ImGui::EndDisabled();
    }
    ImGui::Separator();
    std::string out, err;
    double secs;
    {
        std::lock_guard<std::mutex> l(st->mu);
        out = st->out;
        err = st->err;
        secs = st->secs;
    }
    if (st->busy) ImGui::TextDisabled("%s", tr("Waiting for the answer..."));
    else if (!err.empty()) ImGui::TextColored(kDanger, "%s (%.2f s): %s", tr("Failed"), secs, err.c_str());
    else if (!out.empty()) ImGui::TextColored(g_ok, "%s %.2f s", tr("Answered in"), secs);
    if (!out.empty()) {
        ImGui::SameLine();
        if (ImGui::SmallButton(tr("Copy"))) ImGui::SetClipboardText(out.c_str());
        ImGui::BeginChild("mtout", ImVec2(0, 0), ImGuiChildFlags_Borders);
        ImGui::TextUnformatted(out.c_str());
        ImGui::EndChild();
    }
    ImGui::End();
}

// ---- Find a menu entry, a setting or an action by name (Ctrl+K, the top-bar button, or the search box) and go there


static void apply_theme(App& a, int t) {
    a.cfg.theme = t;
    if (t == THEME_SYSTEM) g_system_dark = -1;  // look again
    theme_default_accent(theme_kind(a.cfg), a.cfg.accent);
    apply_style(a.cfg);
    save_config(a.cfg, a.config_file);
}

static std::vector<PalItem> palette_items(App& a) {
    std::vector<PalItem> v;
    auto sec = [&](const char* icon, const char* label, int s, const char* where) {
        v.push_back({icon, label, where, [s](App& x) { set_section(x, s); }});
    };
    const char* lib = tr("Library");
    const char* tools = tr("Tools");
    sec(ICON_PHOTO, tr("All photos"), SEC_ALL, lib);
    sec(ICON_STAR, tr("Favorites"), SEC_FAV, lib);
    sec(ICON_SPARKLES, tr("AIGC"), SEC_AI, lib);
    sec(ICON_COPY, tr("Deduplicate"), SEC_DUPES, tools);
    sec(ICON_CALENDAR_QUESTION, tr("Fix dates"), SEC_DATES, tools);
    sec(ICON_TAGS, tr("Set tags"), SEC_TAGS, tools);
    v.push_back({ICON_TAGS, tr("Suggestions"), std::string(tools) + " › " + tr("Set tags"), [](App& x) { set_section(x, SEC_SUGG); }});
    sec(ICON_FOLDERS, tr("Organize path"), SEC_ORGANIZE, tools);
    sec(ICON_FILE_EXPORT, tr("Auto set meta"), SEC_SAVE, tools);
    sec(ICON_EYE, tr("Image recognition"), SEC_RECOG, tools);
    sec(ICON_HISTORY, tr("Activity"), SEC_ACTIVITY, tools);
    for (auto& f : a.cfg.folders) v.push_back({ICON_FOLDER, util::basename(f), tr("Folders"), [f](App& x) { set_section(x, SEC_FOLDER, f); }});
    const char* act = tr("Action");
    v.push_back({ICON_SEARCH, tr("Analyze"), act, [](App& x) { request_analyze(x); }});
    v.push_back({ICON_FOLDER_PLUS, tr("Add folder..."), act, [](App& x) { start_dir_dialog(x, 3); }});
    v.push_back({ICON_TAG, tr("Edit tag list..."), act, [](App& x) { x.tag_editor_open = true; }});
    if (!a.undo_run.empty()) v.push_back({ICON_ARROW_BACK_UP, tr("Undo the last run"), act, [](App& x) { start_undo(x, x.undo_run); }});
    v.push_back({ICON_SCREENSHOT, tr("Screenshot of the window"), act, [](App& x) { x.shot_mode = 1; x.shot_frames = 3; }});
    v.push_back({ICON_CROP, tr("Screenshot of an area..."), act, [](App& x) { x.shot_mode = 2; x.shot_a = x.shot_b = ImVec2(-1, -1); }});
    v.push_back({ICON_FLASK, tr("Model tests"), act, [](App& x) { x.show_model_tests = true; }});
    v.push_back({ICON_HELP, tr("Help"), act, [](App& x) { x.show_help = true; }});
    v.push_back({ICON_SETTINGS, tr("Settings"), act, [](App& x) { x.show_settings = true; }});
    v.push_back({ICON_PREVIEW_PANE, tr("Show / hide the preview pane"), act, [](App& x) { x.inspector_on = !x.inspector_on; }});
    v.push_back({ICON_MENU_2, tr("Show / hide the sidebar"), act, [](App& x) { x.sidebar_on = !x.sidebar_on; }});
    v.push_back({ICON_LAYOUT_GRID, tr("Grid / list"), act, [](App& x) { x.photo_grid = !x.photo_grid; }});
    v.push_back({ICON_ACTIVITY, tr("Log"), act, [](App& x) { x.show_log = true; }});
    const char* th[] = {tr("Dark"), tr("Tokyo Night"), tr("Light"), tr("System")};
    for (int t = 0; t < 4; t++) v.push_back({ICON_EYE, std::string(tr("Theme")) + ": " + th[t], act, [t](App& x) { apply_theme(x, t); }});
    // every row of Settings, found where it is drawn
    for (auto& [section, title] : prefs::g_index) {
        std::string key = section + "\x1f" + title;
        bool adv = prefs::g_advanced_sections.count(section) > 0;
        v.push_back({ICON_SETTINGS, title, std::string(tr("Settings")) + " › " + section, [key, adv](App& x) {
                         x.show_settings = true;
                         if (adv) prefs::g_advanced = true;
                         prefs::g_find = key;
                     }});
    }
    return v;
}

// Best matches for a typed text: every word must appear in the name or where it is; names that start with it first.
static std::vector<const PalItem*> palette_match(const std::vector<PalItem>& items, const std::string& typed, size_t max_n) {
    std::vector<std::pair<int, const PalItem*>> scored;
    std::vector<std::string> words;
    for (auto& w : util::split(util::lower(util::trim(typed)), ' '))
        if (!w.empty()) words.push_back(w);
    for (auto& it : items) {
        std::string label = util::lower(it.label), all = label + " " + util::lower(it.where);
        int score = 0;
        bool ok = true;
        for (auto& w : words) {
            size_t p = all.find(w);
            if (p == std::string::npos) { ok = false; break; }
            score += p == 0 ? 30 : label.find(w) != std::string::npos ? (label.find(" " + w) != std::string::npos ? 20 : 10) : 2;
        }
        if (ok) scored.push_back({score, &it});
    }
    std::stable_sort(scored.begin(), scored.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::vector<const PalItem*> out;
    for (auto& [s, p] : scored)
        if (out.size() < max_n) out.push_back(p);
    return out;
}

static void ensure_settings_index(App& a) {
    if (!prefs::g_index.empty()) return;
    // Lay Settings out once, off screen, so every row is known (and the advanced ones too).
    bool adv = prefs::g_advanced;
    prefs::g_advanced = true;
    prefs::g_indexing = true;
    ImGui::SetNextWindowPos(ImVec2(-20000, -20000));
    ImGui::SetNextWindowSize(ImVec2(820, 600));
    ImGui::Begin("##settings_index", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings |
                                                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
    draw_settings(a);
    ImGui::End();
    prefs::g_indexing = false;
    prefs::g_advanced = adv;
}

static void draw_palette(App& a) {
    static char q[128] = "";
    static int sel = 0;
    if (a.palette_open) {
        ensure_settings_index(a);
        ImGui::OpenPopup("##palette");
        a.palette_open = false;
        q[0] = 0;
        sel = 0;
    }
    ImGuiViewport* vp = ImGui::GetMainViewport();
    float w = std::min(620.0f, vp->Size.x - 80);
    ImGui::SetNextWindowPos(ImVec2(vp->Pos.x + (vp->Size.x - w) / 2, vp->Pos.y + 70));
    ImGui::SetNextWindowSize(ImVec2(w, 0));
    if (!ImGui::BeginPopup("##palette", ImGuiWindowFlags_NoMove)) return;
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##pq", tr("Go to a page, a setting or an action..."), q, sizeof q)) sel = 0;
    static std::vector<PalItem> items;
    if (ImGui::IsWindowAppearing() || items.empty()) items = palette_items(a);
    auto m = palette_match(items, q, 12);
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) sel = std::min(int(m.size()) - 1, sel + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) sel = std::max(0, sel - 1);
    const PalItem* chosen = nullptr;
    for (int i = 0; i < int(m.size()); i++) {
        ImGui::PushID(i);
        if (ImGui::Selectable("##it", i == sel, 0, ImVec2(0, ImGui::GetTextLineHeight() + 6))) chosen = m[size_t(i)];
        ImVec2 r0 = ImGui::GetItemRectMin();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddText(ImVec2(r0.x + 6, r0.y + 3), ImGui::GetColorU32(ImGuiCol_TextDisabled), m[size_t(i)]->icon);
        dl->AddText(ImVec2(r0.x + 12 + ImGui::GetFontSize(), r0.y + 3), ImGui::GetColorU32(ImGuiCol_Text), m[size_t(i)]->label.c_str());
        ImVec2 ws = ImGui::CalcTextSize(m[size_t(i)]->where.c_str());
        dl->AddText(ImVec2(ImGui::GetItemRectMax().x - ws.x - 8, r0.y + 3), ImGui::GetColorU32(ImGuiCol_TextDisabled), m[size_t(i)]->where.c_str());
        ImGui::PopID();
    }
    if (m.empty()) ImGui::TextDisabled("%s", tr("Nothing by that name"));
    if ((ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) && sel < int(m.size())) chosen = m[size_t(sel)];
    if (chosen) {
        auto go = chosen->go;
        ImGui::CloseCurrentPopup();
        go(a);
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ---- Keyboard: one list, read by Help > Shortcuts, the palette and the tooltips
struct KeyDef {
    const char* group;
    const char* keys;
    const char* action;
};
static const KeyDef kKeys[] = {
    {"General", "Ctrl+P", "Find a page, a setting or an action (also Ctrl+K)"},
    {"General", "F5", "Analyze the folders"},
    {"General", "Ctrl+J", "Tasks and logs"},
    {"General", "F1", "Help (Ctrl+/ opens the shortcuts)"},
    {"General", "Ctrl+,", "Settings"},
    {"General", "Ctrl+Shift+T", "Switch theme"},
    {"General", "Ctrl+Z", "Undo what was just applied"},
    {"General", "Esc", "Close a dialog, the palette or the viewer"},
#ifdef _WIN32
    {"General", "Alt+F4", "Quit"},
#else
    {"General", "Ctrl+Q", "Quit"},
#endif
    {"View", "Ctrl+B", "Show / hide the sidebar"},
    {"View", "Ctrl+I", "Show / hide the preview pane"},
    {"View", "Ctrl+= / Ctrl+- / Ctrl+0", "Text size: larger / smaller / normal"},
    {"View", "F11", "Full screen"},
    {"View", "Ctrl+wheel", "Thumbnail size in the grid"},
    {"Photos", "Arrows, h j k l", "Move between photos (and months in the list)"},
    {"Photos", "Enter, Space", "Open the viewer"},
    {"Photos", "Ctrl+click, Shift+click", "Select several photos"},
    {"Photos", "f", "Star / unstar"},
    {"Photos", "c", "Copy the paths"},
    {"Photos", "d, Delete", "Move to the Trash (asks; Enter confirms)"},
    {"Photos", "F2", "Rename"},
    {"Photos", "s, Ctrl+F", "Go to the search box"},
    {"Photos", "Backspace", "In an empty search: remove the last filter"},
    {"Viewer", "Left, Right", "Previous / next photo"},
    {"Viewer", "Wheel", "Zoom at the pointer"},
    {"Viewer", "Drag", "Move the zoomed picture"},
    {"Viewer", "Double-click, 0", "Fit / zoom in"},
    {"Viewer", "Esc, q", "Close"},
};

static bool help_match(const std::string& q, std::initializer_list<const char*> fields) {
    if (q.empty()) return true;
    for (auto* f : fields)  // the translated text and the English one: people type English in any language
        if (util::lower(tr(f)).find(q) != std::string::npos || util::lower(f).find(q) != std::string::npos) return true;
    return false;
}

static void draw_help(App& a) {
    if (!a.show_help) return;
    static char q[96] = "";
    ImGui::SetNextWindowSize(ImVec2(780, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetMainViewport()->Size.x - 810, 60), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(tr("Help###help"), &a.show_help)) { ImGui::End(); return; }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Escape, false))
        a.show_help = false;
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##hq", (std::string(ICON_SEARCH "  ") + tr("Search help: a word, a feature, a key")).c_str(), q, sizeof q);
    std::string query = util::lower(util::trim(q));
    int want = a.help_tab;
    a.help_tab = -1;
    if (ImGui::BeginTabBar("helptabs")) {
        auto tab = [&](const char* label, int idx) {
            return ImGui::BeginTabItem(label, nullptr, want == idx ? ImGuiTabItemFlags_SetSelected : 0);
        };
        // 1. Concepts, in the order you meet them
        if (tab((std::string(ICON_LIGHTBULB " ") + tr("Concepts")).c_str(), 0)) {
            struct Card { const char* icon; const char* title; const char* text; const char* where; int section; };
            static const Card cards[] = {
                {ICON_FOLDER, "Folders and the library", "Add the folders where your photos are. jev photos keeps a catalog of them; your files only change when you apply something.", "Sidebar > Library, Folders", SEC_ALL},
                {ICON_FEATURE, "Analyze", "Reads new and changed photos: dates, places, duplicates and what each picture shows. Nothing in your files changes.", "Top bar > Analyze (F5)", -1},
                {ICON_LIST_CHECK, "Tools", "Each tool lists what needs you, with a count: deduplicate, fix dates, set tags, organize path, auto set meta.", "Sidebar > Tools", SEC_DUPES},
                {ICON_HISTORY, "Tasks and Undo", "Long work runs in the background as tasks you can pause, stop and resume. Every change to your files can be undone.", "Top right > Tasks (Ctrl+J), Sidebar > Activity", SEC_ACTIVITY},
                {ICON_SEARCH, "Search and Go to", "Search photos by words, by what they show, or with a sentence. Ctrl+P finds any page, setting or action.", "Top bar > search box, Ctrl+P", -2},
            };
            for (auto& c : cards) {
                if (!help_match(query, {c.title, c.text, c.where})) continue;
                ImGui::PushID(c.title);
                ImGui::BeginGroup();
                ImGui::TextColored(g_accent, "%s", c.icon);
                ImGui::SameLine();
                ImGui::TextUnformatted(tr(c.title));
                ImGui::Indent(ImGui::GetFontSize() * 1.6f);
                ImGui::TextWrapped("%s", tr(c.text));
                ImGui::TextDisabled("%s %s", tr("Where you see it:"), tr(c.where));
                if (c.section != -1 && ImGui::SmallButton(tr("Show me"))) {
                    if (c.section == -2) a.focus_search = true;
                    else set_section(a, c.section);
                    if (c.section == SEC_ACTIVITY) a.status_open_req = true;
                }
                ImGui::Unindent(ImGui::GetFontSize() * 1.6f);
                ImGui::EndGroup();
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::PopID();
            }
            ImGui::EndTabItem();
        }
        // 2. Glossary, A-Z in the current language
        if (tab((std::string(ICON_BOOK " ") + tr("Glossary")).c_str(), 1)) {
            struct C { const char* name; const char* text; };
            static const C cs[] = {
                {"AIGC", "AI-generated images: pictures whose file carries generation data (prompt, model, LoRAs). Sidebar > Library > AIGC."},
                {"AI prompt", "What an AI image was generated from (ComfyUI, A1111, Civitai downloads...). Read-only; searchable."},
                {"Auto set meta", "All in one go: writes missing dates, keywords, descriptions and places into the files where they are, after you review the list."},
                {"Blocked tags", "A tag you delete (tag list, or the x next to a photo's tag) is blocked: never recognised, suggested or written into files again. Unblock it in the tag list."},
                {"Deduplicate", "Identical copies of the same photo, found by size, hash and a byte comparison. One stays; the others can go to the Trash."},
                {"Description", "A sentence written into the file (XMP dc:description): the AI prompt for generated images, plus \"Shows: tags\". An existing description is only changed as you decide."},
                {"EXIF / XMP", "The metadata inside the file. Camera, lens, dates, GPS and AI prompts are the file's own and are never changed."},
                {"File name", "The name on disk. Organize gives photos date names (20190512_00001.jpg), optionally keeping the original name in front."},
                {"Fix dates", "Photos whose date is a guess or missing. Accept the best guesses, or confirm or pick a date in the preview pane."},
                {"Go to", "Ctrl+P (or the search icon at the top right, or typing in the search box) finds a page, a setting or an action by name and takes you there."},
                {"Image recognition", "CLIP: the image analysis that reads each picture once (slow) and keeps the result, so tags and meaning search are fast afterwards."},
                {"jev", "The decision model that settles close calls: dates, places, search results, and cross-checks of the LLM."},
                {"Keywords", "Tags written into the file (XMP dc:subject) so other apps see them. Only added; the file's own keywords stay."},
                {"LLM", "The language model (local server) that reads prompts, picks keywords and judges search results."},
                {"Model tests", "For development (Settings > Advanced, or the tasks popover): send your own input to the language, vision, recognition and jev models."},
                {"Organize path", "Renames photos to date names where they are (the default), or copies / moves them into month folders. Listed first, then applied; undoable."},
                {"Photo folders", "The folders you add (sidebar > Folders). Analyze scans them all together; click one to see only its photos."},
                {"Preview pane", "The right-hand pane: the selected photo with its date, place, tags, file info and why the app decided what it did (Ctrl+I)."},
                {"Red buttons", "They change or remove your original files (move, rename, write into them, move to Trash, replace a description)."},
                {"Screenshot", "The camera at the top right saves the whole window or an area you drag (Pictures/jev-photos); the path is copied."},
                {"Set tags > Suggestions", "Fixes of generated tags and names, proposed with a confidence and applied only when you tick them."},
                {"Tags", "Words for what the picture shows: proposed by CLIP with a confidence, or your own. Searchable."},
                {"Tasks", "Long work (analysing, applying, undoing) waits in a queue and runs one at a time. Pause, resume, stop, retry and reorder them at the top right (Ctrl+J)."},
                {"Undo", "Every applied change is recorded: Activity lists the last 10 runs, each with Undo (Ctrl+Z right after applying)."},
            };
            std::vector<const C*> sorted;
            for (auto& c : cs)
                if (help_match(query, {c.name, c.text})) sorted.push_back(&c);
            std::sort(sorted.begin(), sorted.end(), [](const C* x, const C* y) { return util::lower(tr(x->name)) < util::lower(tr(y->name)); });
            static std::vector<PalItem> items;
            if (items.empty() || ImGui::IsWindowAppearing()) items = palette_items(a);
            if (ImGui::BeginTable("glossary", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH)) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 150);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 80);
                for (auto* c : sorted) {
                    ImGui::TableNextRow();
                    ImGui::PushID(c->name);
                    ImGui::TableNextColumn();
                    ImGui::TextColored(g_accent, "%s", tr(c->name));
                    ImGui::TableNextColumn();
                    ImGui::TextWrapped("%s", tr(c->text));
                    ImGui::TableNextColumn();
                    auto m = palette_match(items, c->name, 1);  // the feature it names, if there is one
                    if (!m.empty() && ImGui::SmallButton(tr("Show me"))) m[0]->go(a);
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            ImGui::EndTabItem();
        }
        // 3. Shortcuts, from the key list
        if (tab((std::string(ICON_KEYBOARD " ") + tr("Shortcuts")).c_str(), 2)) {
            if (ImGui::SmallButton(tr("Copy"))) {
                std::string all;
                for (auto& k : kKeys) all += std::string(tr(k.group)) + "\t" + k.keys + "\t" + tr(k.action) + "\n";
                ImGui::SetClipboardText(all.c_str());
            }
            tip(tr("Copy every shortcut as text"));
            const char* group = "";
            for (auto& k : kKeys) {
                if (!help_match(query, {k.action, k.group}) && util::lower(k.keys).find(query) == std::string::npos) continue;
                if (strcmp(group, k.group) != 0) {
                    group = k.group;
                    ImGui::SeparatorText(tr(group));
                }
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(tr(k.action));
                ImGui::SameLine(ImGui::GetFontSize() * 22);
                ImGui::Dummy(ImVec2(0, 0));
                key_chips(k.keys);
            }
            ImGui::EndTabItem();
        }
        if (tab((std::string(ICON_FILTER " ") + tr("Search syntax")).c_str(), 3)) {
            ImGui::TextWrapped("%s", tr("beach sunset (both) · beach | sea, OR (either) · -night, NOT (without) · ( ) grouping · \"a phrase\" · ~typo"));
            ImGui::TextWrapped("%s", tr("tag: name: desc: prompt: exif: place: camera: model: lora: — one field"));
            ImGui::TextWrapped("%s", tr("date:2019 date:2019-05..2019-08 date:>2020 year:2018..2020 · size:>2mb w:>3000 mp:>12 ext:raw"));
            ImGui::TextWrapped("%s", tr("is:fav is:ai is:untagged is:uncertain is:portrait · has:gps has:prompt has:desc · like:#123 (looks like)"));
            ImGui::TextWrapped("%s", tr("Chinese, Japanese and Korean words also find English tags, and the other way round."));
            ImGui::TextWrapped("%s", tr("Pick a suggestion while typing to add it as a filter chip. A sentence (four words or more) + Enter asks the assistant."));
            ImGui::EndTabItem();
        }
        if (tab((std::string(ICON_INFO_CIRCLE " ") + tr("About")).c_str(), 4)) {
            ImGui::Text("jev photos %s", JEV_VERSION);
            ImGui::TextDisabled("%s", tr("Organize, tag and search photos, with explainable decisions."));
            if (ImGui::Button("github.com/jstdlee/jev-photos")) util::open_path("https://github.com/jstdlee/jev-photos");
            tip(tr("Open the project page in the browser"));
            ImGui::SeparatorText(tr("Credits"));
            const char* credits[] = {"Dear ImGui (Omar Cornut) · GLFW · OpenGL", "SQLite with FTS5 · nlohmann/json · xxHash · stb_image",
                                     "ONNX Runtime · CLIP (OpenAI), ONNX exports by Xenova", "Font Awesome Free 6 (fonts OFL 1.1, icons CC BY 4.0) · Tokyo Night colours (enkia, MIT)", "exiv2 (metadata) · ffmpeg (video frames)",
                                     "jev / Julia decision API · Qwen3 language model (local)", "Built with Claude Code"};
            for (auto* c : credits) ImGui::BulletText("%s", c);
            ImGui::SeparatorText(tr("License"));
            ImGui::TextDisabled("GPL-3.0 (see LICENSE)");
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

static void open_start_tab(App& a) {
    // Older entry points (--tab, ui scripts, buttons) name a tab; map it onto the sidebar.
    if (a.show_plan_tab) { set_section(a, a.plan_op == OP_METADATA ? SEC_SAVE : SEC_ORGANIZE); a.show_plan_tab = false; }
    a.show_overview = false;
    if (a.start_tab.empty()) return;
    std::string t = a.start_tab;
    a.start_tab.clear();
    if (t == "photos" || t == "overview") set_section(a, a.section == SEC_FAV ? SEC_FAV : SEC_ALL);
    else if (t == "favorites") set_section(a, SEC_FAV);
    else if (t == "preview") set_section(a, a.plan_op == OP_METADATA ? SEC_SAVE : SEC_ORGANIZE);
    else if (t == "write") set_section(a, SEC_SAVE);
    else if (t == "dupes") set_section(a, SEC_DUPES);
    else if (t == "dates") set_section(a, SEC_DATES);
    else if (t == "tags") set_section(a, a.meta_sub == "write" ? SEC_SAVE : SEC_TAGS);
    else if (t == "corrections") set_section(a, SEC_SUGG);
    else if (t == "clip") set_section(a, SEC_RECOG);
    else if (t == "log" || t == "activity") set_section(a, SEC_ACTIVITY);
    else if (t == "settings") a.show_settings = true;
    a.meta_sub.clear();
}

static void draw_content(App& a) {
    switch (a.section) {
        case SEC_DUPES: draw_dupes(a); break;
        case SEC_SUGG:
        case SEC_ORGANIZE: draw_plan(a, false); break;
        case SEC_SAVE: draw_plan(a, true); break;
        case SEC_TAGS: {
            int v = a.section == SEC_SUGG ? 1 : 0;
            std::string sugg = a.corrections.empty() ? std::string(tr("Suggestions")) : util::fmt("%s (%zu)", tr("Suggestions"), a.corrections.size());
            if (prefs::seg("settags", &v, {tr("Photos and tags"), sugg})) {
                a.section = v ? SEC_SUGG : SEC_TAGS;
                a.on_tags = a.section == SEC_TAGS;
                a.rows_dirty = true;
            }
            prefs::g_dirty = false;
            ImGui::SameLine(0, 14);
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", v ? tr("Fixes of tags the app generated, each with how sure it is; tick and apply")
                                        : tr("What each photo shows; edit them, fill in missing ones, edit or block tags in the tag list"));
            ImGui::Dummy(ImVec2(0, 4));
            if (v) draw_corrections(a);
            else draw_tags(a);
            break;
        }
        case SEC_RECOG: draw_clip_tab(a); break;
        case SEC_ACTIVITY: draw_activity(a); break;
        default: draw_library(a); break;
    }
}

static void draw_ui(App& a) {
    g_accent = ImVec4(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 1);
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("jev-photos", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImGui::PopStyleVar(2);
    open_start_tab(a);
    bool was_tags = a.on_tags;
    a.on_tags = a.section == SEC_TAGS;
    if (a.on_tags && !was_tags) a.rows_dirty = true;  // entering Tags: fetch its rows
    // keys: Ctrl+I inspector, Ctrl+, settings, Ctrl+Z undo the last run, Ctrl+F search
    ImGuiIO& io = ImGui::GetIO();
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_I, false)) a.inspector_on = !a.inspector_on;
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Comma, false)) a.show_settings = !a.show_settings;
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F, false)) a.focus_search = true;
    if (io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_K, false) || ImGui::IsKeyPressed(ImGuiKey_P, false))) a.palette_open = true;  // feature search
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) a.show_help = !a.show_help;
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Slash, false)) { a.show_help = true; a.help_tab = 2; }  // shortcuts
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_J, false)) a.status_open_req = true;
    if (io.KeyCtrl && !io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_B, false)) a.sidebar_on = !a.sidebar_on;
    if (io.KeyCtrl && io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_T, false)) {  // Dark -> Tokyo Night -> Light -> System
        static const int next[] = {THEME_TOKYO, THEME_LIGHT, THEME_SYSTEM, THEME_DARK};
        apply_theme(a, next[std::clamp(a.cfg.theme, 0, 3)]);
        a.toast = std::string(tr("Theme")) + ": " + tr(a.cfg.theme == THEME_DARK ? "Dark" : a.cfg.theme == THEME_TOKYO ? "Tokyo Night" : a.cfg.theme == THEME_LIGHT ? "Light" : "System");
        a.toast_until = glfwGetTime() + 1.5;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) request_analyze(a);
    if (ImGui::IsKeyPressed(ImGuiKey_F11, false)) {  // full screen on the monitor the window is on
        static int wx, wy, ww, wh;
        if (glfwGetWindowMonitor(a.win)) glfwSetWindowMonitor(a.win, nullptr, wx, wy, ww, wh, 0);
        else {
            glfwGetWindowPos(a.win, &wx, &wy);
            glfwGetWindowSize(a.win, &ww, &wh);
            GLFWmonitor* mon = glfwGetPrimaryMonitor();
            int mc = 0;
            GLFWmonitor** mons = glfwGetMonitors(&mc);
            for (int i = 0; i < mc; i++) {  // the monitor holding the window's centre
                int mx, my;
                const GLFWvidmode* vm = glfwGetVideoMode(mons[i]);
                glfwGetMonitorPos(mons[i], &mx, &my);
                if (vm && wx + ww / 2 >= mx && wx + ww / 2 < mx + vm->width && wy + wh / 2 >= my && wy + wh / 2 < my + vm->height) mon = mons[i];
            }
            if (const GLFWvidmode* vm = glfwGetVideoMode(mon)) glfwSetWindowMonitor(a.win, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
        }
    }
#ifndef _WIN32
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) glfwSetWindowShouldClose(a.win, GLFW_TRUE);
#endif
    {  // Ctrl+ / Ctrl- / Ctrl+0: text size in the same steps as Settings (100, 110, 125, 150 %)
        static const float sizes[] = {16, 17.6f, 20, 24};
        int si = 0;
        for (int i = 0; i < 4; i++)
            if (std::fabs(a.cfg.font_size - sizes[i]) < std::fabs(a.cfg.font_size - sizes[si])) si = i;
        int to = si;
        if (io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_Equal, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false))) to = std::min(3, si + 1);
        if (io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_Minus, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false))) to = std::max(0, si - 1);
        if (io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_0, false) || ImGui::IsKeyPressed(ImGuiKey_Keypad0, false))) to = 0;
        if (to != si && !io.WantTextInput) {
            a.cfg.font_size = sizes[to];
            save_config(a.cfg, a.config_file);
        }
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z, false) && !io.WantTextInput && !a.undo_run.empty() && glfwGetTime() < a.undo_until) start_undo(a, a.undo_run);
    bool first_run = !a.cfg.first_run_done && a.stats.total == 0 && !a.pipe->running();

    // top bar (also the window's drag area when there is no system title bar)
    ImGui::PushStyleColor(ImGuiCol_ChildBg, g_sidebar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
    float bar_h = ImGui::GetFrameHeight() + 16;
    ImGui::BeginChild("topbar", ImVec2(0, bar_h), ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar);
    draw_topbar(a);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (!a.cfg.system_titlebar) window_chrome(a, bar_h);
    g_prof.mark("topbar");

    if (first_run) {
        draw_first_run(a);
    } else {
        // sidebar | content | inspector
        if (a.sidebar_on) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, g_sidebar);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
            ImGui::BeginChild("sidebar", ImVec2(a.sidebar_w, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
            draw_sidebar(a);
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
            bool collapse = false;
            vsplit("##split_side", a.sidebar_w, 160, 420, false, 220, &collapse);
            if (collapse) a.sidebar_on = false;  // dragged away: Ctrl+B or the menu button brings it back
            a.cfg.sidebar_w = a.sidebar_w;
        }
        bool insp = a.inspector_on && library_section(a.section);
        float avail = ImGui::GetContentRegionAvail().x;
        a.inspector_w = std::clamp(a.inspector_w, 260.0f, std::max(260.0f, avail - 360));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));
        ImGui::BeginChild("content", ImVec2(insp ? avail - a.inspector_w - 8 : 0, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
        draw_content(a);
        ImGui::EndChild();
        if (insp) {
            vsplit("##split_insp", a.inspector_w, 260, 2000, true, 340);
            a.cfg.inspector_w = a.inspector_w;
            ImGui::BeginChild("inspector", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding);
            if (!a.selected) ImGui::TextDisabled("%s", tr("Select a photo to see its date, place, tags and file info."));
            else {
                draw_detail(a);
            }
            ImGui::EndChild();
        }
        ImGui::PopStyleVar();
    }
    g_prof.mark("content");
    draw_trash_modal(a);
    draw_analyze_dialog(a);
    draw_tag_editor(a);
    draw_photo_trash(a);
    draw_rename_modal(a);
    draw_palette(a);
    draw_smart_modal(a);
    draw_corr_confirm(a);
    ImGui::End();
    draw_settings_window(a);
    draw_model_tests(a);
    draw_help(a);
    draw_undo_banner(a);
    draw_viewer(a);
    draw_shot_overlay(a);
    if (a.show_log) {
        ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x * 0.7f, vp->WorkSize.y * 0.55f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(tr("Log"), &a.show_log)) draw_log(a);
        ImGui::End();
    }
    ImGui::Begin("jev-photos");
    static double toast_in = -1, toast_prev_until = 0;
    if (glfwGetTime() < a.toast_until) {
        // Feedback: rises 8 px and fades in (200 ms ease-out), leaves the same way (150 ms). A new message while one is
        // showing only changes the text (no restart).
        if (toast_in < 0 || glfwGetTime() > toast_prev_until) toast_in = glfwGetTime();
        toast_prev_until = a.toast_until;
        float in = anim::progress(toast_in, 0.20f);
        float left = float(a.toast_until - glfwGetTime());
        float out = left < 0.15f ? anim::ease_out(1 - left / 0.15f) : 0.0f;
        if (out > 0) anim::keep_drawing();
        float alpha = in * (1 - out), dy = anim::g_reduce ? 0.0f : (1 - in) * 8 + out * 8;
        ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 20, vp->WorkPos.y + vp->WorkSize.y - 20 + dy), 0, ImVec2(1, 1));
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.01f, alpha));
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(a.toast.c_str());
        ImGui::EndTooltip();
        ImGui::PopStyleVar();
    } else {
        toast_in = -1;
    }
    ImGui::End();
}

// Where screenshots go: <Pictures>/jev-photos/screenshot-YYYYMMDD-HHMMSS.png
static std::string screenshot_path() {
    std::string dir = util::home() + "/Pictures";
#ifndef _WIN32
    util::ProcResult r = util::run({"xdg-user-dir", "PICTURES"}, "", 3);
    if (r.rc == 0 && !util::trim(r.out).empty()) dir = util::trim(r.out);
#endif
    dir += "/jev-photos";
    util::mkdirs(dir);
    util::Civil c = util::local_from_epoch(util::now_epoch());
    std::string base = dir + util::fmt("/screenshot-%04d%02d%02d-%02d%02d%02d", c.y, c.mo, c.d, c.h, c.mi, c.s), p = base + ".png";
    for (int k = 2; util::file_exists(p); k++) p = base + util::fmt("-%d.png", k);
    return p;
}

// Part of the framebuffer (x, y, w, h in framebuffer pixels, from the top left) as a PNG.
static bool save_screenshot_area(const std::string& path, int fw, int fh, int x, int y, int w, int h) {
    x = std::clamp(x, 0, fw - 1);
    y = std::clamp(y, 0, fh - 1);
    w = std::clamp(w, 1, fw - x);
    h = std::clamp(h, 1, fh - y);
    std::vector<unsigned char> px(size_t(w) * h * 4), out(px.size());
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(x, fh - y - h, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    for (int r = 0; r < h; r++) memcpy(&out[size_t(r) * w * 4], &px[size_t(h - 1 - r) * w * 4], size_t(w) * 4);
    for (size_t i = 3; i < out.size(); i += 4) out[i] = 255;
    return stbi_write_png(path.c_str(), w, h, 4, out.data(), w * 4) != 0;
}

static void save_screenshot(const std::string& path, int w, int h) {
    std::vector<unsigned char> px(size_t(w) * h * 4), out(px.size());
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h; y++) memcpy(&out[size_t(y) * w * 4], &px[size_t(h - 1 - y) * w * 4], size_t(w) * 4);
    for (size_t i = 3; i < out.size(); i += 4) out[i] = 255;
    if (stbi_write_png(path.c_str(), w, h, 4, out.data(), w * 4)) fprintf(stderr, "saved %s (%dx%d)\n", path.c_str(), w, h);
}

// ---------------------------------------------------------------------------
// CLI

// GUI supervisor. The NVIDIA GL driver can segfault while creating/drawing the first frames when GPU memory is
// nearly exhausted (e.g. a large model server holds most of the unified memory). That cannot be caught in-process,
// so the window runs in a child: if it dies from a crash in its first 20 s, it is restarted once with Mesa software
// rendering, which is plenty for this UI.
#ifndef _WIN32
static void set_software_gl() {
    setenv("__GLX_VENDOR_LIBRARY_NAME", "mesa", 1);
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
}

static volatile pid_t g_child = 0;
static void forward_signal(int sig) {  // the supervisor passes stop requests on to the window process
    if (g_child > 0) kill(g_child, sig);
}

static int supervise(int argc, char** argv, int renderer) {
    auto launch = [&](bool software) -> std::pair<int, double> {
        std::vector<std::string> env_s;
        for (char** e = environ; *e; e++) env_s.push_back(*e);
        env_s.push_back("JEV_PHOTOS_CHILD=1");
        if (software) {
            env_s.push_back("__GLX_VENDOR_LIBRARY_NAME=mesa");
            env_s.push_back("LIBGL_ALWAYS_SOFTWARE=1");
            env_s.push_back("JEV_PHOTOS_SOFTWARE=1");
        }
        std::vector<char*> envp;
        for (auto& e : env_s) envp.push_back(e.data());
        envp.push_back(nullptr);
        std::vector<char*> args(argv, argv + argc);
        args.push_back(nullptr);
        pid_t pid;
        auto t0 = std::chrono::steady_clock::now();
        if (posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, args.data(), envp.data()) != 0) return {-1, 0};
        g_child = pid;
        int st = 0;
        while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (WIFSIGNALED(st)) return {128 + WTERMSIG(st), secs};
        return {WIFEXITED(st) ? WEXITSTATUS(st) : 1, secs};
    };
    signal(SIGINT, SIG_IGN);  // the child handles Ctrl-C and exits cleanly
    signal(SIGTERM, forward_signal);
    signal(SIGHUP, forward_signal);
    auto [rc, secs] = launch(renderer == 2);
    bool crashed = rc == 128 + SIGSEGV || rc == 128 + SIGABRT || rc == 128 + SIGBUS;
    if (crashed && secs < 20 && renderer == 0) {
        fprintf(stderr, "jev-photos: the GPU driver crashed while opening the window (GPU memory may be exhausted); "
                        "restarting with software rendering\n");
        rc = launch(true).first;
    }
    return rc;
}

#endif

// --clip-test IMG...: load CLIP, print the tags it picks and the time it takes (for tuning the vocabulary).
static int clip_test(const Config& c, const std::vector<std::string>& files) {
    ClipModel m;
    std::string err;
    ClipInfo info = clip_choose(c);
    double t0 = glfwGetTime();
    if (!m.load(info, c.clip_device, c.clip_threads, true, true, err)) { fprintf(stderr, "clip: %s\n", err.c_str()); return 1; }
    TagIndex idx;
    double t1 = glfwGetTime();
    if (!build_tag_index(m, idx, err)) { fprintf(stderr, "tags: %s\n", err.c_str()); return 1; }
    printf("%s on %s, load %.1fs, %zu tag prompts %.1fs\n", info.label.c_str(), m.device_used().c_str(), t1 - t0, idx.tags.size(), glfwGetTime() - t1);
    for (auto& f : files) {
        std::vector<float> chw;
        std::vector<Vec> emb;
        double a = glfwGetTime();
        if (!clip_preprocess(f, chw, err) || !m.encode_images({chw}, emb, err)) { printf("%s: %s\n", f.c_str(), err.c_str()); continue; }
        ImageTags t = pick_tags(idx, emb[0], c.clip_max_tags);
        printf("%-40s %4.0f ms  scene=%s\n   ", util::basename(f).substr(0, 40).c_str(), (glfwGetTime() - a) * 1000, t.scene.c_str());
        for (auto& [tag, p] : t.tags) printf("%s %.2f, ", tag.c_str(), p);
        if (getenv("JEV_CLIP_DEBUG")) {  // per category: mass, then the best three by share
            TagScores sc = score_tags(idx, emb[0]);
            for (auto& [cat, m] : sc.mass) {
                std::vector<size_t> in;
                for (size_t i = 0; i < idx.tags.size(); i++)
                    if (idx.tags[i].category == cat) in.push_back(i);
                std::sort(in.begin(), in.end(), [&](size_t x, size_t y) { return sc.prob[x] > sc.prob[y]; });
                printf("\n     %-18s %.2f:", cat.c_str(), m);
                for (size_t k = 0; k < std::min<size_t>(3, in.size()); k++) printf(" %s %.2f", idx.tags[in[k]].tag.c_str(), sc.share(idx, in[k]));
            }
        }
        printf("\n");
    }
    return 0;
}

static void usage() {
    printf(
        "Usage: jev-photos [FOLDER]                 (GUI)\n"
        "       jev-photos --cli FOLDER [options]   (headless run with a progress bar)\n\n"
        "  FOLDER                 the folder with your photos; organized copies go to FOLDER/jev-organized\n"
        "  --out DIR              put organized copies somewhere else (alias --lib)\n"
        "  --db FILE              catalog database (default ~/.local/share/jev-photos/catalog.sqlite)\n"
        "  --config FILE          settings file (default ~/.config/jev-photos/config.ini)\n"

        "  --stages LIST          scan,dupes,decide,tag,vision,organize or all (default all)\n"
        "  --scope DIR            scan and process only this folder (inside or outside the sources)\n"
        "  --dry-run              print the preview (copies, renames, metadata additions) and change nothing\n"
        "  --move                 move instead of copy\n"
        "  --op copy|move|rename|metadata   what organize does (metadata = only fill in missing fields, in place)\n"
        "  --names date|keep      20190512_00001, or keep the original name as a prefix (IMG_20190512_00001)\n"
        "  --no-rewrite           keep keywords jev-photos wrote earlier even when the tags changed\n"
        "  --retag                recompute every photo's tags from its stored CLIP embedding\n"
        "  --write embed|sidecar|db   where new metadata goes (default embed into the library copy)\n"
        "  --no-vision / --no-jev disable the VL model / jev decisions for this run\n"
        "  --redecide             re-run date/location decisions for every photo\n"
        "  --retry-vision         retry photos whose vision call failed\n"
        "  --save-config          write the effective settings back to the config file\n"
        "  --explain FILE [--root DIR]   print the date/location decision for one file and exit\n"
        "  --search TEXT          search the folder's photos and exit; with --mode smart|keyword|semantic|regex,\n"
        "                         --in name,exif,desc (fields for words/regex), --from 2019-05-01 --to 2019-08-31\n"
        "                         (or --years 2018-2020); --mode ask: natural language via the LLM + decider\n"
        "  --stats                print library statistics (counts and sizes) and exit\n"
        "  --dupes                print exact-duplicate and similar-photo groups and exit\n"
        "  --similar              also find near duplicates (resized/re-saved) during this run\n"
        "  --preview-path DIR     GUI: open and immediately preview this folder\n"
        "  --tab NAME             GUI: start on photos, preview, dupes, log or settings\n"
        "  --screenshot FILE [--delay SEC]   GUI: save a PNG of the window and exit\n");
}

static int cli_explain(const Config& c, const std::string& file, const std::string& root_in) {
    if (!util::file_exists(file)) { fprintf(stderr, "no such file: %s\n", file.c_str()); return 1; }
    std::string path = util::canonical(file);
    std::string root = root_in.empty() ? util::dirname(path) : util::canonical(root_in);
    MetaMap m = read_meta(path);
    struct stat st{};
    stat(path.c_str(), &st);
    DateInputs in;
    in.filename = util::basename(path);
    in.dirs = rel_dirs(root, path);
    in.meta = &m;
    in.mtime = ST_MTIME(st);
    in.now = util::now_epoch();
    in.min_year = c.min_year;
    DateDecision d = decide_date(in, c.review_below);
    printf("file: %s\n", path.c_str());
    if (!m.error.empty()) printf("metadata error: %s\n", m.error.c_str());
    printf("\ncandidates:\n");
    for (size_t i = 0; i < d.candidates.size(); i++) {
        auto& k = d.candidates[i];
        std::string notes;
        for (auto& n : k.notes) notes += " [" + n + "]";
        printf("  %s %-15s %-19s w=%.2f/%.2f S=%.2f  %s%s\n", !d.alternatives.empty() && d.alternatives[0] == int(i) ? "*" : " ",
               date_source_id(k.src), k.when.valid() ? k.when.pretty().c_str() : "-", k.weight, k.base, k.support, k.raw.c_str(), notes.c_str());
    }
    if (c.jev_enabled && d.alternatives.size() >= 2 && d.margin < c.jev_margin) {
        std::vector<std::pair<std::string, std::string>> opts;
        for (size_t k = 0; k < d.alternatives.size(); k++) opts.push_back({"d" + std::to_string(k), date_option_text(d, d.alternatives[k])});
        printf("\nasking jev (margin %.2f < %.2f):\n", d.margin, c.jev_margin);
        for (auto& o : opts) printf("  %s: %s\n", o.first.c_str(), o.second.c_str());
        JevChoice jc = jev_choice(c, "Which is the real date this photo was taken?", opts, "{}");
        if (jc.ok) apply_jev_date(d, jc.probs, c.jev_date_weight, c.review_below);
        else printf("  jev unavailable: %s\n", jc.error.c_str());
    }
    printf("\ndecision: %s  (%s, precision %s)\n  confidence %.2f  margin %.2f  decider %s%s\n", d.when.valid() ? d.when.str().c_str() : "undated",
           d.source.c_str(), util::precision_name(d.when.prec), d.confidence, d.margin, d.decider.c_str(), d.needs_review ? "  NEEDS REVIEW" : "");
    if (!d.jev_note.empty()) printf("  %s\n", d.jev_note.c_str());

    auto pc = place_candidates(in.dirs, util::stem(path));
    printf("\nplace candidates:");
    for (auto& p : pc) printf(" [%s prior=%.2f%s]", p.text.c_str(), p.prior, p.travel_cue ? " trip" : "");
    printf("\n");
    double lat, lon;
    if (meta_gps(m, lat, lon)) printf("GPS: %.6f, %.6f\n", lat, lon);
    return 0;
}

static int run_cli(App& a, bool dry) {
    if (a.cfg.folders.empty()) {
        fprintf(stderr, "usage: jev-photos --cli FOLDER [FOLDER...]   (the folders with your photos)\n");
        return 2;
    }
    if (!open_db(a)) return 2;
    a.log.echo = true;
    a.opts.scope = a.opts.scope.empty() ? all_scope(a.cfg) : util::canonical(a.opts.scope);
    a.opts.plan_only = dry;
    for (auto& f : a.cfg.folders) fprintf(stderr, "folder %s\norganized copies -> %s\n", f.c_str(), library_for(effective(a.cfg), f).c_str());
    a.pipe->start(effective(a.cfg), a.opts);
    bool tty = isatty(2);
    while (a.pipe->running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if (g_quit) a.pipe->stop();
        Progress& p = a.pipe->progress;
        int done = p.done, total = p.total;
        if (tty && p.stage) {
            int w = 30, fill = total ? done * w / total : 0;
            std::string bar(size_t(fill), '#');
            bar += std::string(size_t(w - fill), '.');
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - p.stage_start).count();
            fprintf(stderr, "\r\033[K%-8s [%s] %d/%d  eta %s  err %d", stage_name(p.stage), bar.c_str(), done, total,
                    fmt_eta(done ? el / done * (total - done) : -1).c_str(), p.errors.load());
        }
    }
    a.pipe->join();
    if (tty) fprintf(stderr, "\r\033[K");
    if (dry) {
        for (auto& it : a.pipe->plan()) {
            printf("%-9s %-19s %.2f%s  %s\n          -> %s\n", it.action.c_str(), it.date.c_str(), it.conf, it.needs_review ? " review" : "",
                   it.src.c_str(), it.action == "duplicate" ? it.note.c_str() : it.dest.c_str());
            std::set<std::string> removed(it.meta_removed.begin(), it.meta_removed.end());
            for (auto& k : it.meta_removed) printf("          - Xmp.dc.subject %s  (written by jev-photos earlier)\n", k.c_str());
            if (!it.desc_old.empty() && !it.desc_action.empty())
                printf("          description \"%s\": %s (%s) %s\n", it.desc_old.substr(0, 60).c_str(),
                       it.desc_action == "decide" ? "YOU DECIDE in Review" : it.desc_action.c_str(), it.desc_by.c_str(), it.desc_why.c_str());
            for (auto& c : it.meta_cmds)
                if (!util::starts_with(c, "set Xmp.jev.") && !util::starts_with(c, "del ")) printf("          + %s\n", c.c_str());
        }
    }
    return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    util::attach_console();  // a GUI program: print to the terminal it was started from (CLI modes, --help)
#endif
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif
    App app;
    Pipeline pipe(app.db, app.log);
    app.pipe = &pipe;
    bool cli = false, save_cfg = false, stats = false;
    std::string explain, explain_root, search, shot, startup_preview;
    bool dupes_report = false, frame_stats = false, show_corrections = false;
    int64_t view_id = 0;  // GUI: open the viewer on this photo at start
    std::string ui_script;
    SearchQuery sq;
    int ui_test = 0;
    double shot_delay = 4;
    std::vector<std::string> srcs;
    std::string lib;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--config") && i + 1 < argc) app.config_file = argv[++i];
    load_config(app.cfg, app.config_file);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--config") next();
        else if (a == "--cli") cli = true;
        else if (a == "--src") srcs.push_back(next());
        else if (a == "--lib" || a == "--out") lib = next();
        else if (a == "--stages") {
            std::string s = next();
            app.opts.stages = 0;
            for (auto& t : util::split(s, ','))
                app.opts.stages |= t == "scan" ? ST_SCAN : t == "decide" ? ST_DECIDE : t == "vision" ? ST_VISION : t == "organize" ? ST_ORGANIZE : t == "dupes" || t == "duplicates" ? ST_DUPES : t == "tag" ? ST_TAG : t == "check" || t == "fix" ? ST_FIX : t == "all" ? ST_ALL : 0;
        } else if (a == "--dry-run") app.cfg.dry_run = true;
        else if (a == "--move") app.cfg.file_op = OP_MOVE;
        else if (a == "--op" && i + 1 < argc) {
            std::string v = argv[++i];
            app.cfg.file_op = v == "move" ? OP_MOVE : v == "rename" ? OP_RENAME : v == "metadata" ? OP_METADATA : OP_COPY;
        } else if (a == "--names" && i + 1 < argc) app.cfg.name_style = std::string(argv[++i]) == "keep" ? NAME_KEEP_PREFIX : NAME_DATE_SN;
        else if (a == "--no-rewrite") app.cfg.rewrite_tags = false;
        else if (a == "--retag") app.opts.retag_all = true;
        else if (a == "--recheck") app.opts.recheck_all = true;
        else if (a == "--corrections") show_corrections = true;
        else if (a == "--write") {
            std::string w = next();
            app.cfg.write_mode = w == "sidecar" ? WRITE_SIDECAR : w == "db" ? WRITE_DB_ONLY : WRITE_EMBED;
        } else if (a == "--no-vision") app.cfg.vl_enabled = false;
        else if (a == "--no-jev") app.cfg.jev_enabled = false;
        else if (a == "--redecide") app.opts.redecide_all = true;
        else if (a == "--scope" || a == "--only") app.opts.scope = next();
        else if (a == "--retry-vision") app.opts.retry_failed_vision = true;
        else if (a == "--save-config") save_cfg = true;
        else if (a == "--explain") explain = next();
        else if (a == "--root") explain_root = next();
        else if (a == "--search") search = next();
        else if (a == "--stats") stats = true;
        else if (a == "--dupes") dupes_report = true;
        else if (a == "--mode") {
            std::string m = next();
            sq.mode = m == "keyword" ? SM_KEYWORD : m == "semantic" ? SM_SEMANTIC : m == "regex" ? SM_REGEX : m == "ask" ? SM_ASK : SM_SMART;
        } else if (a == "--in") {
            std::string f = next();
            sq.in_name = f.find("name") != std::string::npos;
            sq.in_exif = f.find("exif") != std::string::npos;
            sq.in_desc = f.find("desc") != std::string::npos;
        } else if (a == "--years") {  // shorthand: 2018-2020 or 2019
            std::string y = next();
            int a1 = 0, b1 = 0;
            if (sscanf(y.c_str(), "%d-%d", &a1, &b1) != 2) b1 = a1 = atoi(y.c_str());
            sq.date_from = util::fmt("%04d-01-01", a1);
            sq.date_to = util::fmt("%04d-12-31", b1);
        } else if (a == "--from") sq.date_from = next();
        else if (a == "--to") sq.date_to = next();
        else if (a == "--similar") app.cfg.similar_check = true;
        else if (a == "--screenshot") shot = next();
        else if (a == "--delay") shot_delay = atof(next().c_str());
        else if (a == "--select") app.selected = atoll(next().c_str());
        else if (a == "--preview-path") startup_preview = next();
        else if (a == "--plan-select") app.plan_sel = atoi(next().c_str());
        else if (a == "--tab") app.start_tab = next();
        else if (a == "--gui-search") snprintf(app.search, sizeof app.search, "%s", next().c_str());
        else if (a == "--view") view_id = atoll(next().c_str());
        else if (a == "--ui-script") ui_script = next();  // test: "j,j,Return,shot:/tmp/a.png,Escape" fed to the UI itself
        else if (a == "--frame-stats") frame_stats = true;
        else if (a == "--ui-test") ui_test = atoi(next().c_str());  // seconds: click through the tabs while running Start
        else if (a == "--dup-select") app.dup_sel = atoll(next().c_str());
        else if (a == "--db") app.cfg.db_override = next();
        else if (a == "--clip-model" && i + 1 < argc) app.cfg.clip_model = argv[++i];
        else if (a == "--clip-test") { std::vector<std::string> fs; while (i + 1 < argc) fs.push_back(argv[++i]); glfwInit(); return clip_test(app.cfg, fs); }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--version") { printf("jev-photos %s\n", JEV_VERSION); return 0; }
        else if (!a.empty() && a[0] != '-') srcs.push_back(a);  // positional: the photo folder
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    if (!srcs.empty()) {
        // Folders on the command line: the CLI works on exactly those (together); the GUI adds them to its list.
        if (cli) app.cfg.folders.clear();
        for (auto& f : srcs) add_folder(app.cfg, util::canonical(f));
        app.cfg.folder = util::canonical(srcs[0]);
        app.cfg.view_all = cli && srcs.size() > 1;
    }
    if (!lib.empty()) app.cfg.output = util::canonical(lib);
    if (save_cfg) save_config(app.cfg, app.config_file);
    signal(SIGINT, [](int) { g_quit = 1; });
    signal(SIGTERM, [](int) { g_quit = 1; });

    if (!explain.empty()) return cli_explain(app.cfg, explain, explain_root);
    if (dupes_report) {
        if (!open_db(app)) return 2;
        std::vector<DupRow> rows;
        for (auto& r : app.db.dup_rows(false))
            if (app.cfg.folders.empty() || path_under(r.src_path, view_scope(app.cfg))) rows.push_back(r);
        std::map<int64_t, const DupRow*> by;
        std::map<int64_t, std::vector<const DupRow*>> exact, similar;
        for (auto& r : rows) by[r.id] = &r;
        for (auto& r : rows) {
            if (r.dup_of) exact[r.dup_of].push_back(&r);
            if (r.similar_to) similar[r.similar_to].push_back(&r);
        }
        int64_t reclaim = 0;
        for (auto& [k, v] : exact) {
            printf("exact: keep %s (%s)\n", by[k]->src_path.c_str(), util::human_size(by[k]->size).c_str());
            for (auto* r : v) { printf("       dup  %s\n", r->src_path.c_str()); reclaim += r->size; }
        }
        for (auto& [k, v] : similar) {
            printf("similar: largest %s (%dx%d, %s)\n", by[k]->src_path.c_str(), by[k]->width, by[k]->height, util::human_size(by[k]->size).c_str());
            for (auto* r : v) printf("         d=%-2d %s (%dx%d, %s)\n", r->similar_dist, r->src_path.c_str(), r->width, r->height, util::human_size(r->size).c_str());
        }
        printf("%zu exact groups, %s reclaimable; %zu similar groups\n", exact.size(), util::human_size(reclaim).c_str(), similar.size());
        return 0;
    }
    if (show_corrections) {  // --corrections: list the pending correction proposals
        if (!open_db(app)) return 2;
        auto cs = app.db.corrections(view_scope(app.cfg), true);
        for (auto& k : cs) {
            std::string what = k.field == "name" ? "rename " + k.current + " -> " + k.proposed : (k.action == "remove" ? "- " + k.current : "+ " + k.proposed);
            std::string path = k.path;
            path = rel_path(app.cfg, path);
            printf("%3.0f%%  %-34s %-28s %s  [%s]\n", k.confidence * 100, path.substr(0, 34).c_str(), what.c_str(), k.reason.c_str(), k.source.c_str());
        }
        printf("%zu pending corrections\n", cs.size());
        return 0;
    }
    if (!search.empty() || stats || !sq.date_from.empty() || !sq.date_to.empty()) {
        if (!open_db(app)) return 2;
        if (stats) {
            DbStats s = app.db.stats(view_scope(app.cfg));
            printf("photos %d (%s)  resolved %d  needs-review %d  undated %d  located %d  described %d  organized %d\n"
                   "duplicates %d (%s reclaimable)  similar %d\n",
                   s.total, util::human_size(s.total_bytes).c_str(), s.resolved, s.review, s.undated, s.located, s.vision_done, s.organized,
                   s.dups, util::human_size(s.dup_bytes).c_str(), s.similar);
            for (auto& m : app.db.months(view_scope(app.cfg))) printf("  %-8s %5d  %s\n", m.month.c_str(), m.count, util::human_size(m.bytes).c_str());
        }
        if (!search.empty() || !sq.date_from.empty() || !sq.date_to.empty()) {
            Searcher se;
            se.load(app.db, view_scope(app.cfg), 0);
            sq.text = search;
            sq.translate = true;  // the CLI waits for the LLM's other-language equivalents
            SearchResult sr = se.run(app.cfg, sq);
            if (!sr.error.empty()) { fprintf(stderr, "%s\n", sr.error.c_str()); return 2; }
            if (!sr.note.empty()) fprintf(stderr, "note: %s\n", sr.note.c_str());
            for (auto& d : sr.decisions) {
                app.db.add_decision(d);
                fprintf(stderr, "jev: %s  %s -> jev %s -> %s%s\n", d.question.c_str(), d.rules_pick.c_str(), d.jev_pick.c_str(), d.final_pick.c_str(),
                        d.changed ? " (changed)" : "");
            }
            std::map<int64_t, PhotoRow> rows;
            for (auto& r : app.db.query("", false, "", 1000000, view_scope(app.cfg))) rows[r.id] = r;
            for (auto& h : sr.hits) {
                auto it = rows.find(h.id);
                if (it == rows.end()) continue;
                const PhotoRow& r = it->second;
                std::string path = r.dest_path.empty() ? r.src_path : r.dest_path;
                path = rel_path(app.cfg, path);
                printf("%.2f  %-10s  %-44s  %s\n", h.score, r.date_value.substr(0, 10).c_str(), path.c_str(), h.why.c_str());
            }
            fprintf(stderr, "%zu matches (%s, %zu photos searched)\n", sr.hits.size(), search_mode_name(sq.mode), se.size());
        }
        return 0;
    }
    if (cli) return run_cli(app, app.cfg.dry_run);
    // GUI only: the log also goes to a file; waiting tasks from the last session come back paused
    // (loaded after the catalog opens, below).

    // ---------------- GUI
#ifndef _WIN32  // (the NVIDIA-on-GB10 startup crash this guards against is Linux-only)
    if (!getenv("JEV_PHOTOS_CHILD") && !getenv("JEV_PHOTOS_NO_SUPERVISOR")) return supervise(argc, argv, app.cfg.renderer);
    if (getenv("JEV_PHOTOS_CHILD")) prctl(PR_SET_PDEATHSIG, SIGTERM);  // the window never outlives its supervisor
#endif
    if (!app.cfg.lang.empty()) set_lang(lang_from_code(app.cfg.lang.c_str()));
    else {
        const char* l = getenv("LC_ALL");
        if (!l || !*l) l = getenv("LC_MESSAGES");
        if (!l || !*l) l = getenv("LANG");
        set_lang(lang_from_code(l));
    }
#ifdef GLFW_PLATFORM
    if (glfwPlatformSupported(GLFW_PLATFORM_X11)) glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
    glfwSetErrorCallback([](int c, const char* d) { fprintf(stderr, "GLFW error %d: %s\n", c, d); });
    if (!glfwInit()) {
        fprintf(stderr, "no display: use --cli\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (getenv("JEV_PHOTOS_HIDDEN") && (!shot.empty() || !ui_script.empty())) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // tests
    glfwWindowHint(GLFW_DECORATED, app.cfg.system_titlebar ? GLFW_TRUE : GLFW_FALSE);  // our own top bar: drag, double-click, window buttons, resize grip
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "jev-photos");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "jev-photos");
    app.win = glfwCreateWindow(1400, 860, "jev photos", nullptr, nullptr);
    if (!app.win) {
        glfwTerminate();
        return 1;
    }
    {  // window icon (dock, Alt-Tab, title bar)
        GLFWimage img;
        int n = 0;
        img.pixels = stbi_load_from_memory(kIconPng, int(sizeof kIconPng), &img.width, &img.height, &n, 4);
        if (img.pixels) {
            glfwSetWindowIcon(app.win, 1, &img);
            stbi_image_free(img.pixels);
        }
    }
    glfwMakeContextCurrent(app.win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Floating windows (Settings, Help, Model tests, Log) reopen where you left them.
    static std::string layout_ini = util::config_dir() + "/window-layout.ini";
    io.IniFilename = getenv("JEV_PHOTOS_NO_LAYOUT") || !ui_script.empty() || !shot.empty() ? nullptr : layout_ini.c_str();
    if (!app.cfg.first_run_done && !app.cfg.folders.empty()) app.cfg.first_run_done = true;
    apply_motion(app.cfg);
    if (!cli) app.log.file = util::data_dir() + "/jev-photos.log";
    {  // keep the log file small: start a new one past 2 MB
        struct stat st{};
        if (stat(app.log.file.c_str(), &st) == 0 && st.st_size > 2 * 1024 * 1024) rename(app.log.file.c_str(), (app.log.file + ".1").c_str());
    }
    app.sidebar_w = std::clamp(app.cfg.sidebar_w, 160.0f, 420.0f);
    app.inspector_w = std::clamp(app.cfg.inspector_w, 260.0f, 2000.0f);  // set up before the welcome screen existed
    apply_style(app.cfg);
    build_fonts();
    ImGui_ImplGlfw_InitForOpenGL(app.win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    if (!app.cfg.folder.empty() && util::dir_exists(app.cfg.folder)) remember_folder(app.cfg, app.cfg.folder);
    snprintf(app.folder_buf, sizeof app.folder_buf, "%s", app.cfg.folder.c_str());
    open_db(app);
    check_health(app);
    if (!startup_preview.empty()) {
        std::string p = util::canonical(startup_preview);
        set_folder(app, p);
        start_run(app, ST_SCAN | ST_DUPES | ST_DECIDE | ST_ORGANIZE, true);
    }
    app.log.add(0, exiv2_available() ? "exiv2 found" : "exiv2 NOT found: install it to read/write metadata");
    if (view_id) open_viewer(app, {view_id}, 0);
    if (ui_script.empty() && shot.empty()) load_queue(app);  // waiting tasks from the last session, paused
    if (const GLubyte* r = glGetString(GL_RENDERER)) app.gl_renderer = reinterpret_cast<const char*>(r);
    if (getenv("JEV_PHOTOS_SOFTWARE") && app.cfg.renderer == 0)
        app.log.add(1, "the GPU driver crashed at startup, so this window uses software rendering (" + app.gl_renderer +
                           "). Free GPU memory and restart to use the GPU again.");
    else
        app.log.add(0, "OpenGL: " + app.gl_renderer);
    int frame = 0;
    double last_frame = glfwGetTime(), worst_gap = 0;
    int busy_frames = 0;
    while (!glfwWindowShouldClose(app.win) && !g_quit) {
        if (frame_stats) {  // UI responsiveness while a run is active: the longest gap between two frames
            double t = glfwGetTime();
            if (app.pipe->running()) { worst_gap = std::max(worst_gap, t - last_frame); busy_frames++; }
            last_frame = t;
        }
        bool busy = app.pipe->running() || glfwGetTime() < app.toast_until || app.view.loading() || ui_test;
        // After any input keep drawing at full rate for a moment: ImGui needs a frame or two after a click to show
        // its effect (tab switch, popup), and an idle 0.5 s sleep made that feel like a stall.
        static double hot_until = 0;
        // Idle: sleep until something happens (health checks, worker results and thumbnails post events).
        double wait = ui_test ? 0.02 : glfwGetTime() < hot_until ? 0.016 : busy ? 0.1 : 5.0;
        if (g_wake_at > glfwGetTime()) wait = std::min(wait, std::max(0.005, g_wake_at - glfwGetTime()));  // a tooltip is due
        double w0 = glfwGetTime();
        glfwWaitEventsTimeout(wait);
        if (glfwGetTime() - w0 < wait * 0.9) hot_until = glfwGetTime() + 0.6;  // woken early = an event arrived
        g_prof.begin(glfwGetTime());
        if (ui_test) {  // automated clicking: a new tab every 0.4 s, Start at 1 s, quit after ui_test seconds
            static const char* tabs[] = {"photos", "preview", "dupes", "log", "settings"};
            static double t0 = glfwGetTime(), next_tab = 0;
            static int ti = 0;
            static bool started = false;
            double el = glfwGetTime() - t0;
            if (!started && el > 1) { start_run(app, ST_SCAN | ST_DUPES | ST_DECIDE | ST_VISION | ST_ORGANIZE); started = true; }
            if (el > next_tab) { app.start_tab = tabs[ti++ % 5]; next_tab = el + 0.4; }
            if (el > ui_test) glfwSetWindowShouldClose(app.win, GLFW_TRUE);
        }
        // Typing paused for 0.7 s: search again, now allowed to ask the LLM for other-language equivalents.
        if (app.search_edit_at > 0 && !app.search_translate && glfwGetTime() - app.search_edit_at > 0.7 && app.search[0]) {
            app.search_translate = true;
            app.rows_dirty = true;
        }
        // Views are rebuilt by the worker: on demand, and every 1.5 s while a run is going.
        double now = glfwGetTime();
        if (!app.cfg.folders.empty() && (app.rows_dirty || app.dups_dirty || (app.pipe->running() && now - app.last_refresh > 1.5))) {
            ViewRequest vr;
            vr.month = app.month;
            vr.folder = app.section == SEC_FOLDER && !app.section_folder.empty() ? app.section_folder : all_scope(app.cfg);
            vr.ai_only = app.section == SEC_AI;
            vr.place = app.section == SEC_PLACE ? app.place : "";
            vr.review_only = app.review_only || app.section == SEC_DATES;
            vr.dup_similar = app.dup_show_similar;
            vr.sq.text = compose_query(app);
            vr.sq.mode = app.regex ? SM_REGEX : app.ask_go ? SM_ASK : SM_SMART;
            vr.sq.in_name = app.in_name;
            vr.sq.in_exif = app.in_exif;
            vr.sq.in_desc = app.in_desc;
            vr.sq.in_prompt = app.in_prompt;
            vr.sq.translate = app.search_translate;
            vr.sq.date_from = app.date_from;
            vr.sq.date_to = app.date_to;
            vr.gen = app.catalog_gen.load();
            vr.cfg = app.cfg;
            vr.tags = app.on_tags;
            vr.fav_only = app.fav_only;
            app.view.request(vr);
            app.rows_dirty = app.dups_dirty = false;
            app.last_refresh = now;
        }
        {
            ViewData v;
            if (app.view.take(v)) {
                app.rows = std::move(v.rows);
                app.months = std::move(v.months);
                app.stats = v.stats;
                app.dup_rows = std::move(v.dup_rows);
                app.dup_lines = std::move(v.dup_lines);
                app.dup_summary = std::move(v.dup_summary);
                app.searching = v.searching;
                app.search_note = std::move(v.search_note);
                app.search_error = std::move(v.search_error);
                if (v.tags_built) {
                    app.tag_rows = std::move(v.tag_rows);
                    app.tags_built = true;
                }
                app.decisions = std::move(v.decisions);
                app.places = std::move(v.places);
                app.ai_count = v.ai_count;
                app.favorites = std::move(v.favorites);
                app.corrections = std::move(v.corrections);
                app.decision_stats = v.decision_stats;
                app.view_gen++;
            }
        }
        if (app.want_dupes_run.exchange(false) && !app.pipe->running()) start_run(app, ST_DUPES);
        static bool was_trashing = false;
        if (was_trashing && !app.trash_busy) {
            app.rows_dirty = app.dups_dirty = true;
            app.toast = app.trash_what == "photos"
                            ? util::fmt("%d %s", app.trash_done.load(), tr(app.trash_done == 1 ? "photo moved to the Trash" : "photos moved to the Trash"))
                            : util::fmt("%d %s", app.trash_done.load(), tr("extra copies moved"));
            if (app.trash_failed) app.toast += util::fmt("  ·  %d %s", app.trash_failed.load(), tr("could not be moved"));
            app.toast_until = glfwGetTime() + 3;
        }
        was_trashing = app.trash_busy;
        if (glfwGetTime() - app.health_at > 20) {
            app.health_at = glfwGetTime();
            check_health(app);
        }
        // Folder watching: now and then, while idle, pick up new or changed photos quietly (incremental scan; only
        // new pictures are analysed). Nothing in the files changes, and no list to apply is built.
        {
            static double last_busy = glfwGetTime();
            if (app.pipe->running() || app.view.loading() || !app.tasks.empty()) last_busy = glfwGetTime();
            else if (app.cfg.watch_folders && app.cfg.first_run_done && !app.cfg.folders.empty() && !ui_test && ui_script.empty() &&
                     shot.empty() && glfwGetTime() - last_busy > 600) {
                last_busy = glfwGetTime();
                RunOptions keep = app.opts;
                app.opts.clip_mode = app.opts.tag_mode = 0;
                app.log.add(0, "watching: looking for new photos");
                start_run(app, (ST_SCAN | ST_DUPES | ST_DECIDE | (app.cfg.clip_enabled ? ST_TAG : 0)));
                app.opts = keep;
            }
        }
        static int seen_runs = 0;
        if (app.pipe->progress.finished_runs != seen_runs && !app.pipe->running()) {
            seen_runs = app.pipe->progress.finished_runs;
            app.rows_dirty = true;
            app.sel_loaded = false;
            app.dups_dirty = true;
            app.catalog_gen++;
            app.plan = app.pipe->plan();
            // Auto set meta: after an Analyze, build the metadata list, then apply it (undoable from Activity).
            if (app.auto_meta_step == -1 && app.cfg.auto_meta) {
                app.auto_meta_step = 1;
                app.log.add(0, "auto set meta: listing what to write into the files");
                start_run(app, ST_ORGANIZE, true, OP_METADATA);
            } else if (app.auto_meta_step == 1) {
                int n = 0;
                for (auto& it : app.plan) n += it.include && it.action == "metadata";
                app.auto_meta_step = n ? 2 : 0;
                if (n) {
                    app.log.add(0, util::fmt("auto set meta: writing %d files", n));
                    apply_plan(app);
                    app.show_plan_tab = false;  // stay where you are
                }
            } else if (app.auto_meta_step == 2) {
                app.auto_meta_step = 0;
            }
            static std::string seen_apply;
            if (app.pipe->last_applied_run != seen_apply) {  // a run changed files: offer Undo for a while
                seen_apply = app.pipe->last_applied_run;
                int n = 0;
                for (auto& it : app.plan) n += it.result == "ok";
                app.undo_run = seen_apply;
                app.undo_text = util::fmt("%d %s", n, tr(app.plan_op == OP_METADATA ? "files updated" : app.plan_op == OP_COPY ? "photos copied"
                                                        : app.plan_op == OP_MOVE ? "photos moved" : "photos renamed"));
                if (app.plan_op == OP_COPY)  // the originals keep their names: say where the renamed copies are
                    app.undo_text += std::string("  ·  ") + tr("in jev-organized; your originals keep their names");
                app.undo_until = glfwGetTime() + 30;
            }
            task_finished(app);  // last: the next queued task may change the current list
        }
        if (!app.dlg_running) {
            std::lock_guard<std::mutex> l(app.dlg_mu);
            if (!app.dlg_result.empty()) {
                if (app.dlg_target == 1) snprintf(app.lib_buf, sizeof app.lib_buf, "%s", app.dlg_result.c_str());
                else if (app.dlg_target == 3) set_folder(app, app.dlg_result);
                app.dlg_result.clear();
            }
        }
        g_prof.mark("events+view");
        ImGui::GetStyle().FontSizeBase = app.cfg.font_size;
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        // Scripted input (tests): keys go straight into ImGui, never through the desktop.
        static std::string script_shot;
        if (!ui_script.empty()) {
            static std::vector<std::string> steps = util::split(ui_script, ',');
            static size_t si = 0;
            static double next_t = glfwGetTime() + 3.0;
            static ImGuiKey release = ImGuiKey_None;
            ImGuiIO& sio = ImGui::GetIO();
            if (release != ImGuiKey_None) {
                sio.AddKeyEvent(release, false);
                release = ImGuiKey_None;
            } else if (glfwGetTime() >= next_t) {
                if (si < steps.size()) {
                    std::string t = util::trim(steps[si++]);
                    next_t = glfwGetTime() + 0.6;
                    if (util::starts_with(t, "shot:")) script_shot = t.substr(5);
                    else if (util::starts_with(t, "wait:")) next_t = glfwGetTime() + atoi(t.c_str() + 5) / 1000.0;
                    else {
                        static const std::map<std::string, ImGuiKey> keys = {
                            {"j", ImGuiKey_J}, {"k", ImGuiKey_K}, {"h", ImGuiKey_H}, {"l", ImGuiKey_L}, {"q", ImGuiKey_Q}, {"f", ImGuiKey_F}, {"c", ImGuiKey_C}, {"d", ImGuiKey_D}, {"s", ImGuiKey_S}, {"Delete", ImGuiKey_Delete},
                            {"Up", ImGuiKey_UpArrow}, {"Down", ImGuiKey_DownArrow}, {"Left", ImGuiKey_LeftArrow},
                            {"Right", ImGuiKey_RightArrow}, {"Return", ImGuiKey_Enter}, {"Escape", ImGuiKey_Escape}, {"space", ImGuiKey_Space}};
                        auto k = keys.find(t);
                        if (k != keys.end()) { sio.AddKeyEvent(k->second, true); release = k->second; }
                        else if (util::starts_with(t, "tab:")) app.start_tab = t.substr(4);
                        else if (t == "trash-dupes") trash_duplicates(app);  // same as confirming the dialog
                        else if (t == "trash-dialog") app.trash_confirm = true;
                        else if (util::starts_with(t, "ask:")) {  // Ask search for the given sentence
                            snprintf(app.search, sizeof app.search, "%s", t.substr(4).c_str());
                            app.ask_go = true;
                            app.rows_dirty = true;
                        }
                        else if (t == "plan") start_run(app, ST_ORGANIZE, true);
                        else if (t == "analyze") request_analyze(app);
                        else if (t == "undo") { if (!app.undo_run.empty()) start_undo(app, app.undo_run); }
                        else if (t == "find") app.focus_search = true;
                        else if (t == "rename") { if (app.selected) begin_rename(app, app.selected); }
                        else if (t == "palette") app.palette_open = true;
                        else if (t == "status") app.status_open_req = true;
                        else if (util::starts_with(t, "helptab:")) { app.show_help = true; app.help_tab = atoi(t.substr(8).c_str()); }
                        else if (util::starts_with(t, "lang:")) { set_lang(lang_from_code(t.substr(5).c_str())); app.cfg.lang = t.substr(5); app.want_font_rebuild = true; }
                        else if (t == "modeltests") app.show_model_tests = true;
                        else if (t == "shotarea") { app.shot_mode = 2; app.shot_a = app.shot_b = ImVec2(-1, -1); }
                        else if (util::starts_with(t, "mouse:")) {  // mouse:X;Y  (window pixels)
                            auto xy = util::split(t.substr(6), ';');
                            if (xy.size() == 2) sio.AddMousePosEvent(float(atof(xy[0].c_str())), float(atof(xy[1].c_str())));
                        }
                        else if (util::starts_with(t, "wheel:")) sio.AddMouseWheelEvent(0, float(atof(t.substr(6).c_str())));
                        else if (t == "inspector") app.inspector_on = !app.inspector_on;
                        else if (util::starts_with(t, "type:")) sio.AddInputCharactersUTF8(t.substr(5).c_str());
                        else if (t == "apply") apply_plan(app);
                        else if (t == "help") app.show_help = true;
                        else if (t == "tageditor") app.tag_editor_open = true;
                        else if (t == "smart") start_smart(app);
                    }
                } else if (script_shot.empty()) {
                    glfwSetWindowShouldClose(app.win, GLFW_TRUE);
                }
            }
        }
        if (app.want_font_rebuild) {  // between frames: the atlas may be rebuilt here
            app.want_font_rebuild = false;
            build_fonts();
        }
        ImGui::NewFrame();
        draw_ui(app);
        ImGui::Render();
        g_prof.mark("imgui");
        int fw, fh;
        glfwGetFramebufferSize(app.win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        { ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg]; glClearColor(bg.x, bg.y, bg.z, 1); }
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        g_prof.mark("gl");
        if (!script_shot.empty()) {
            save_screenshot(script_shot, fw, fh);
            script_shot.clear();
        }
        // Screenshot from the top bar: a frame or two later, so the menu (or the area overlay) is gone from the picture.
        if ((app.shot_mode == 1 || app.shot_mode == 3) && --app.shot_frames <= 0) {
            std::string path = screenshot_path();
            bool ok;
            if (app.shot_mode == 1) ok = (save_screenshot(path, fw, fh), util::file_exists(path));
            else {
                ImVec2 sc = ImGui::GetIO().DisplayFramebufferScale;
                ok = save_screenshot_area(path, fw, fh, int(app.shot_a.x * sc.x), int(app.shot_a.y * sc.y), int((app.shot_b.x - app.shot_a.x) * sc.x),
                                          int((app.shot_b.y - app.shot_a.y) * sc.y));
            }
            app.shot_mode = 0;
            if (ok) {
                app.shot_last = path;
                glfwSetClipboardString(app.win, path.c_str());
                app.toast = std::string(tr("Screenshot saved (path copied): ")) + util::basename(path);
            } else {
                app.toast = tr("Could not save the screenshot");
            }
            app.toast_until = glfwGetTime() + 3.5;
        }
        if (!shot.empty() && ++frame > 3 && glfwGetTime() >= shot_delay) {
            save_screenshot(shot, fw, fh);
            glfwSetWindowShouldClose(app.win, GLFW_TRUE);
        }
        glfwSwapBuffers(app.win);
        g_prof.mark("swap");
        static std::map<std::string, std::pair<double, int>> worst;  // section -> (max seconds, frames)
        if (ui_test)
            for (auto& [n, t] : g_prof.parts) {
                auto& w = worst[n];
                w.first = std::max(w.first, t);
                w.second++;
            }
        if (ui_test && glfwWindowShouldClose(app.win)) {
            fprintf(stderr, "worst time per section over the test:\n");
            for (auto& [n, w] : worst) fprintf(stderr, "  %-14s %6.1f ms  (%d frames)\n", n.c_str(), w.first * 1000, w.second);
        }
        if (g_prof.total() > 0.1) {
            static int reported = 0;
            std::string msg = util::fmt("slow UI frame %.0f ms: ", g_prof.total() * 1000) + g_prof.str();
            if (frame_stats || ui_test) fprintf(stderr, "%s\n", msg.c_str());
            if (reported++ < 50) app.log.add(1, msg);
        }
    }
    if (frame_stats) fprintf(stderr, "frames during run: %d, longest frame gap: %.0f ms\n", busy_frames, worst_gap * 1000);
    save_queue(app);
    save_config(app.cfg, app.config_file);
    app.pipe->stop();
    // A model call in flight can take its full timeout; do not hold the closed window hostage for it.
    for (int i = 0; i < 30 && app.pipe->running(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (app.pipe->running()) {
        fprintf(stderr, "exiting while a model call is still in flight (it is abandoned)\n");
        std::_Exit(0);
    }
    app.pipe->join();
    app.view.stop();
    app.thumb.unload();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(app.win);
    glfwTerminate();
    return 0;
}
