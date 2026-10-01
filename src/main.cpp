// jev-photos — organize a photo collection by recovered capture date, with jev decisions and VL tagging.
// Dear ImGui + GLFW + OpenGL 3 (same stack as gpu-hud), SQLite catalogue, exiv2 for metadata.
#include "backends/imgui_impl_opengl3_loader.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <spawn.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

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
extern char** environ;

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

static void build_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    FontFace base;
    for (const char* p : {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                          "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"})
        if (util::file_exists(p)) { base.file = p; break; }
    if (base.file.empty()) base = fc_match("sans-serif:lang=en");
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    bool have = false;
    if (!base.file.empty()) {
        cfg.FontNo = base.index;
        have = io.Fonts->AddFontFromFileTTF(base.file.c_str(), 0.0f, &cfg) != nullptr;
    }
    if (!have) io.Fonts->AddFontDefault();
    // Always merge a CJK face: file and folder names are often Chinese/Japanese even in an English UI.
    for (const char* pat : {"sans-serif:lang=zh-cn", "sans-serif:lang=ja", "sans-serif:lang=ko"}) {
        FontFace cjk = fc_match(pat);
        if (cjk.file.empty() || cjk.file == base.file) continue;
        ImFontConfig m;
        m.MergeMode = true;
        m.FontNo = cjk.index;
        io.Fonts->AddFontFromFileTTF(cjk.file.c_str(), 0.0f, &m);
        break;
    }
}

// ---------------------------------------------------------------------------
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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pw, ph, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
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
    bool fav_only = false;
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
    v.months = db.months(r.folder);
    v.stats = db.stats(r.folder);
    v.decisions = db.decisions(r.folder, 300);
    v.corrections = db.corrections(r.folder, true);
    for (auto& row : db.query("", false, "", 1000000, r.folder))
        if (row.favorite) v.favorites.push_back(std::move(row));
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
    std::set<int64_t> pending;
    std::map<int64_t, Texture> tex;
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
    auto p = [](ImGuiKey k) { return ImGui::IsKeyPressed(k, true); };
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
struct ViewerSlot {
    std::mutex mu;
    std::atomic<int64_t> want{0};
    std::vector<unsigned char> rgba;
    int w = 0, h = 0, stage = 0;
    int64_t id = 0;
    bool ready = false;
};
struct Viewer {
    bool open = false;
    std::vector<int64_t> ids;
    int idx = 0;
    int64_t loaded_id = 0;
    int shown_stage = 0;
    int opened_frame = -1;  // the key that opened the viewer must not also move it
    Photo p;
    std::shared_ptr<ViewerSlot> slot = std::make_shared<ViewerSlot>();
};

static void run_detached(std::vector<std::string> argv) {
    std::thread([argv] { util::run(argv, "", 30); }).detach();
}

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
    int search_mode = SM_SMART;
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
    int64_t thumb_id = 0;
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
    std::string action_sub;  // Actions sub-tab to open next
    // Analyze dialog: how much image analysis (CLIP) and tagging to do
    bool analyze_dlg = false;
    ClipCounts ana_counts, ana_counts_quick;
    int ana_clip = 0, ana_tags = 0;
    bool photo_grid = false; // Photos: thumbnails instead of the table
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
};

static void apply_style(const Config& c) {
    ImGuiStyle& st = ImGui::GetStyle();
    ImGui::StyleColorsDark(&st);
    st.WindowRounding = 0.0f;
    st.FrameRounding = 5.0f;
    st.PopupRounding = 6.0f;
    st.GrabRounding = 4.0f;
    st.ChildRounding = 6.0f;
    st.TabRounding = 5.0f;
    st.WindowPadding = ImVec2(12, 10);
    st.ItemSpacing = ImVec2(8, 6);
    st.WindowBorderSize = 0;
    ImVec4 acc(c.accent[0], c.accent[1], c.accent[2], 1);
    st.Colors[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.08f, 0.10f, 1.0f);
    st.Colors[ImGuiCol_ChildBg] = ImVec4(1, 1, 1, 0.02f);
    st.Colors[ImGuiCol_CheckMark] = acc;
    st.Colors[ImGuiCol_SliderGrab] = acc;
    st.Colors[ImGuiCol_SliderGrabActive] = acc;
    st.Colors[ImGuiCol_PlotHistogram] = acc;
    st.Colors[ImGuiCol_Button] = ImVec4(1, 1, 1, 0.08f);
    st.Colors[ImGuiCol_ButtonHovered] = ImVec4(acc.x, acc.y, acc.z, 0.45f);
    st.Colors[ImGuiCol_ButtonActive] = ImVec4(acc.x, acc.y, acc.z, 0.70f);
    st.Colors[ImGuiCol_FrameBg] = ImVec4(1, 1, 1, 0.08f);
    st.Colors[ImGuiCol_FrameBgHovered] = ImVec4(1, 1, 1, 0.14f);
    st.Colors[ImGuiCol_FrameBgActive] = ImVec4(acc.x, acc.y, acc.z, 0.35f);
    st.Colors[ImGuiCol_Header] = ImVec4(acc.x, acc.y, acc.z, 0.30f);
    st.Colors[ImGuiCol_HeaderHovered] = ImVec4(acc.x, acc.y, acc.z, 0.45f);
    st.Colors[ImGuiCol_HeaderActive] = ImVec4(acc.x, acc.y, acc.z, 0.60f);
    st.Colors[ImGuiCol_Tab] = ImVec4(1, 1, 1, 0.06f);
    st.Colors[ImGuiCol_TabHovered] = ImVec4(acc.x, acc.y, acc.z, 0.45f);
    st.Colors[ImGuiCol_TabSelected] = ImVec4(acc.x, acc.y, acc.z, 0.35f);
    st.Colors[ImGuiCol_TableHeaderBg] = ImVec4(1, 1, 1, 0.06f);
    st.Colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.09f, 0.11f, 0.98f);
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
        util::ProcResult r = util::run({"zenity", "--file-selection", "--directory", "--title=Choose folder"}, "", 3600);
        if (r.rc == 127) r = util::run({"kdialog", "--getexistingdirectory", util::home()}, "", 3600);
        {
            std::lock_guard<std::mutex> l(a.dlg_mu);
            a.dlg_result = r.rc == 0 ? util::trim(r.out) : "";
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
    std::string f = std::filesystem::weakly_canonical(util::trim(f_in), ec).string();
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

static void start_run(App& a, int stages, bool force_preview = false) {
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
    a.last_run_planned = o.plan_only;
    a.last_stages = stages;
    a.pipe->start(effective(a.cfg), o);
}

static int analyze_stages(const Config& c) {
    return ST_SCAN | ST_DUPES | ST_DECIDE | ST_ORGANIZE | ST_FIX | (c.clip_enabled ? ST_TAG : 0) | (c.vl_enabled ? ST_VISION : 0);
}

static void apply_plan(App& a) {
    RunOptions o = a.opts;
    o.stages = ST_ORGANIZE;
    o.apply_plan = true;
    o.scope = all_scope(a.cfg);  // every folder in the list, together
    a.last_run_planned = false;
    a.last_stages = ST_ORGANIZE;
    a.show_plan_tab = true;
    a.pipe->start(effective(a.cfg), o);
}

// ---------------------------------------------------------------------------
// UI pieces

// Tooltips and button styles used everywhere: the accent colour for the main action of a screen, red for anything
// that changes or removes original files.
static void tip(const char* text) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}
static ImVec4 g_accent(0.30f, 0.78f, 0.47f, 1);
static const ImVec4 kDanger(0.86f, 0.30f, 0.27f, 1);
static bool styled_button(const char* label, const ImVec4& col, const ImVec2& size, const char* tip_text) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(col.x, col.y, col.z, 0.70f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(col.x, col.y, col.z, 0.88f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(col.x, col.y, col.z, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
    bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(4);
    if (tip_text) tip(tip_text);
    return clicked;
}
static bool primary_button(const char* label, const char* tip_text, const ImVec2& size = ImVec2(0, 0)) { return styled_button(label, g_accent, size, tip_text); }
static bool danger_button(const char* label, const char* tip_text, const ImVec2& size = ImVec2(0, 0)) { return styled_button(label, kDanger, size, tip_text); }
// Apply of the organize list: red when it moves or renames the originals (or writes into them).
static bool apply_button(const Config& c, const char* label, const ImVec2& size = ImVec2(0, 0)) {
    bool originals = c.file_op == OP_MOVE || c.file_op == OP_RENAME || (c.file_op == OP_METADATA && c.write_mode == WRITE_EMBED);
    const char* t = c.file_op == OP_MOVE ? "Moves your original files into the month folders and renames them. No copies are kept."
                  : c.file_op == OP_RENAME ? "Renames your original files in their folders."
                  : c.file_op == OP_METADATA ? "Adds the listed fields into the files themselves (existing values are never changed)."
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
    ImVec4 col = ok == 1 ? ImVec4(0.3f, 0.85f, 0.45f, 1) : ok == 0 ? ImVec4(0.98f, 0.80f, 0.20f, 1) : ImVec4(0.6f, 0.6f, 0.6f, 1);
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
static void draw_toolbar(App& a) {
    bool busy = a.pipe->running();
    ImVec4 acc(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 1);

    // Folders: a list scanned together; the selector shows one folder or all of them.
    ImGui::BeginDisabled(busy);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(tr(a.cfg.folders.size() > 1 ? "Photo folders" : "Photo folder"));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(220.0f, ImGui::GetContentRegionAvail().x - 470));
    std::string cur = a.cfg.folders.empty() ? std::string(tr("choose the folder with your photos"))
                    : a.cfg.view_all ? util::fmt("%s (%zu)", tr("All folders"), a.cfg.folders.size()) : a.cfg.folder;
    if (ImGui::BeginCombo("##folders", cur.c_str())) {
        if (a.cfg.folders.size() > 1 && ImGui::Selectable(util::fmt("%s (%zu)", tr("All folders"), a.cfg.folders.size()).c_str(), a.cfg.view_all))
            set_view_all(a, true);
        for (auto& f : std::vector<std::string>(a.cfg.folders))
            if (ImGui::Selectable(f.c_str(), !a.cfg.view_all && f == a.cfg.folder)) set_folder(a, f);
        if (a.cfg.folders.empty()) ImGui::TextDisabled("%s", tr("No folders yet: press Add."));
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered() && a.cfg.folders.size() > 1)
        ImGui::SetTooltip("%s", tr("Analyze scans every folder in the list. Here you choose what the tabs show: one folder or all."));
    ImGui::SameLine();
    if (ImGui::Button(tr("+ Add..."))) start_dir_dialog(a, 3);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Add a photo folder to the list"));
    ImGui::SameLine();
    if (ImGui::Button(util::fmt("%s (%zu)", tr("Folders"), a.cfg.folders.size()).c_str())) ImGui::OpenPopup("folders");
    tip(tr("The folder list: open, remove, add by path, or re-add a recent one"));
    if (ImGui::BeginPopup("folders")) {
        ImGui::TextDisabled("%s", tr("Scanned together by Analyze. Removing a folder only takes it off the list: no file is touched."));
        std::string remove;
        if (ImGui::BeginTable("fl", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
            for (auto& f : a.cfg.folders) {
                ImGui::TableNextRow();
                ImGui::PushID(f.c_str());
                ImGui::TableNextColumn();
                bool exists = util::dir_exists(f);
                if (exists) ImGui::TextUnformatted(f.c_str());
                else ImGui::TextColored(ImVec4(0.95f, 0.6f, 0.4f, 1), "%s  (%s)", f.c_str(), tr("not found"));
                ImGui::TableNextColumn();
                if (ImGui::SmallButton(tr("Open"))) run_detached({"xdg-open", f});
                tip(tr("Open this folder in the file manager"));
                ImGui::TableNextColumn();
                if (ImGui::SmallButton(tr("Remove"))) remove = f;
                tip(tr("Take this folder off the list. No file is touched; its photos stay in the catalog."));
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (!remove.empty()) {
            remove_folder(a.cfg, remove);
            save_config(a.cfg, a.config_file);
            a.month.clear();
            a.selected = 0;
            a.rows_dirty = a.dups_dirty = true;
        }
        ImGui::Separator();
        ImGui::SetNextItemWidth(420);
        bool go = ImGui::InputTextWithHint("##addpath", tr("type or paste a folder path"), a.folder_buf, sizeof a.folder_buf, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if ((ImGui::Button(tr("Add")) || go) && a.folder_buf[0]) set_folder(a, a.folder_buf);
        tip(tr("Add the typed folder to the list"));
        ImGui::SameLine();
        if (ImGui::Button(tr("Browse..."))) start_dir_dialog(a, 3);
        tip(tr("Choose a folder with the file dialog"));
        std::vector<std::string> again;
        for (auto& r : a.cfg.recent)
            if (std::find(a.cfg.folders.begin(), a.cfg.folders.end(), r) == a.cfg.folders.end()) again.push_back(r);
        if (!again.empty()) {
            ImGui::TextDisabled("%s", tr("Recent folders (click to add again)"));
            for (auto& r : again)
                if (ImGui::Selectable(r.c_str())) set_folder(a, r);
        }
        ImGui::EndPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 16);

    // The one main action. While running it becomes Stop.
    if (!busy) {
        if (primary_button((std::string("   ") + tr("Analyze") + "   ").c_str(),
                           tr("Scan the folders, find duplicates, work out dates and places, tag the pictures and plan the organized copies. Nothing is changed until you apply it in Actions.")))
            request_analyze(a);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.35f, 0.3f, 0.65f));
        if (ImGui::Button((std::string("    ") + tr("Stop") + "    ").c_str())) a.pipe->stop();
        tip(tr("Stop the running step after the current photo; everything done so far is kept"));
        ImGui::PopStyleColor();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("...")) ImGui::OpenPopup("more");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Run a single step"));
    if (ImGui::BeginPopup("more")) {
        ImGui::TextDisabled("%s", tr("Run one step on this folder"));
        if (ImGui::MenuItem(tr("Scan for new or changed files"))) start_run(a, ST_SCAN);
        if (ImGui::MenuItem(tr("Find duplicates"))) start_run(a, ST_DUPES);
        if (ImGui::MenuItem(tr("Work out dates and places"))) start_run(a, ST_DECIDE);
        if (ImGui::MenuItem(tr("Tag pictures (CLIP)"), nullptr, false, a.cfg.clip_enabled)) start_run(a, ST_TAG);
        if (ImGui::MenuItem(tr("Describe with the vision model"), nullptr, false, a.cfg.vl_enabled)) start_run(a, ST_VISION);
        if (ImGui::MenuItem(tr("Plan the organized copies"))) start_run(a, ST_ORGANIZE, true);
        ImGui::Separator();
        ImGui::Checkbox(tr("Re-decide all dates"), &a.opts.redecide_all);
        tip(tr("Next run works out every date again, not only for new or changed photos"));
        ImGui::Checkbox(tr("Retry failed vision"), &a.opts.retry_failed_vision);
        tip(tr("Next run retries photos the vision model could not describe"));
        ImGui::EndPopup();
    }
    ImGui::EndDisabled();

    // Health of the helpers, right-aligned.
    std::string jd, vd, cd, ld;
    {
        std::lock_guard<std::mutex> l(a.health_mu);
        jd = a.jev_detail;
        vd = a.vl_detail;
        cd = a.clip_detail;
        ld = a.llm_detail;
    }
    float right = ImGui::GetWindowContentRegionMax().x;
    ImGui::SameLine(right - (a.cfg.vl_enabled ? 225 : 170));
    g_prof.mark("tb:buttons");
    status_dot(a.clip_ok, "CLIP", cd);
    ImGui::SameLine(0, 12);
    status_dot(a.llm_ok, "AI", a.cfg.llm_url + "  " + ld);
    ImGui::SameLine(0, 12);
    status_dot(a.jev_ok, "jev", a.cfg.jev_url + "  " + jd);
    if (a.cfg.vl_enabled) {
        ImGui::SameLine(0, 12);
        status_dot(a.vl_ok, "VL", a.cfg.vl_url + "  " + vd);
    }
    g_prof.mark("tb:dots");

    // Status line: progress while running, otherwise the last summary; the log is one click away.
    Progress& p = a.pipe->progress;
    int st = p.stage, done = p.done, total = p.total;
    float frac = total > 0 ? float(done) / float(total) : (busy ? 0.0f : 1.0f);
    std::string label;
    if (busy && st) {
        static const int order[] = {ST_SCAN, ST_DUPES, ST_DECIDE, ST_TAG, ST_VISION, ST_ORGANIZE};
        static const char* names[] = {"Scanning", "Finding duplicates", "Dates and places", "Tagging", "Describing", "Planning"};
        int idx = 0, count = 0, which = 0;
        for (int k = 0; k < 6; k++)
            if (a.last_stages & order[k]) {
                count++;
                if (order[k] == st) { idx = count; which = k; }
            }
        const char* name = order[which] == ST_ORGANIZE && !a.last_run_planned ? "Copying" : names[which];
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - p.stage_start).count();
        double eta = done > 0 ? el / done * (total - done) : -1;
        label = util::fmt("%s %d/%d  ·  %s  ·  %d / %d  ·  %s %s", tr("Step"), idx, std::max(count, 1), tr(name), done, total,
                          fmt_eta(eta).c_str(), tr("remaining"));
        if (p.errors) label += util::fmt("  ·  %d %s", p.errors.load(), tr("errors"));
    } else {
        std::lock_guard<std::mutex> l(p.mu);
        label = p.summary.empty() ? (a.cfg.folders.empty() ? tr("Choose a photo folder, then press Analyze.") : tr("Ready. Press Analyze.")) : p.summary;
    }
    g_prof.mark("tb:label");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
    ImGui::ProgressBar(frac, ImVec2(ImGui::GetContentRegionAvail().x - 60, 0), label.c_str());
    ImGui::SameLine();
    if (ImGui::Button(tr("Log"), ImVec2(52, 0))) a.show_log = !a.show_log;
    tip(tr("Show what the last runs did, step by step"));
    if (busy) ImGui::TextDisabled("%s", p.get_current().c_str());

    // Next step banner: a plan is waiting for approval.
    if (!busy && !a.plan.empty() && !a.on_overview) {
        int n = count_plan(a.plan, false), nr = count_plan(a.plan, true);
        if (n > 0) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(acc.x, acc.y, acc.z, 0.14f));
            ImGui::BeginChild("banner", ImVec2(0, ImGui::GetFrameHeightWithSpacing() + 8), ImGuiChildFlags_None);
            ImGui::SetCursorPos(ImVec2(10, 4));
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(acc, "%s", tr("Ready to organize:"));
            ImGui::SameLine();
            // Short destination (full path on hover), buttons right after the text so nothing overlaps.
            std::string out = output_dir(a.cfg), shown = out;
            if (util::starts_with(out, a.cfg.folder + "/")) shown = util::basename(a.cfg.folder) + "/" + out.substr(a.cfg.folder.size() + 1);
            ImGui::Text("%d %s  →  %s", n, tr(a.cfg.file_op == OP_MOVE ? "photos to move" : a.cfg.file_op == OP_RENAME ? "photos to rename"
                                               : a.cfg.file_op == OP_METADATA ? "photos to add metadata to" : "photos to copy"),
                        a.cfg.file_op == OP_RENAME || a.cfg.file_op == OP_METADATA ? tr("their own folders") : shown.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", out.c_str());
            if (nr) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "·  %d %s", nr, tr("with uncertain dates"));
            }
            ImGui::SameLine(0, 24);
            if (ImGui::Button(tr("Open in Actions"), ImVec2(140, 0))) a.show_plan_tab = true;
            tip(tr("See the list of planned changes, one per row, before applying it"));
            ImGui::SameLine();
            if (apply_button(a.cfg, util::fmt("%s (%d)", tr("Apply"), n).c_str(), ImVec2(120, 0))) apply_plan(a);
            ImGui::EndChild();
            ImGui::PopStyleColor();
        }
    }

    // Log window
    if (a.show_log) {
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x * 0.7f, vp->WorkSize.y * 0.55f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.15f, vp->WorkPos.y + vp->WorkSize.y * 0.25f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(tr("Log"), &a.show_log)) draw_log(a);
        ImGui::End();
    }
}

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
    if (t.id == a.thumb_id && !t.rgba.empty()) a.thumb.upload(t.rgba.data(), t.w, t.h);
    t.rgba.clear();
}

static void viewer_load(App& a) {
    Viewer& v = a.viewer;
    if (v.idx < 0 || v.idx >= int(v.ids.size())) return;
    int64_t id = v.ids[size_t(v.idx)];
    if (v.loaded_id == id) return;
    v.loaded_id = id;
    v.shown_stage = 0;
    a.viewer_tex.unload();
    if (!a.ui_db.load(id, v.p)) return;
    ThumbKey key{v.p.src_path, v.p.size, v.p.mtime};
    std::string dest = v.p.dest_path;
    int orient = meta_orientation(MetaMap::from_json(v.p.meta_json));
    int side = thumb_side_for(v.p.src_path, a.cfg.thumb_side, a.cfg.vl_max_side);
    auto slot = v.slot;
    slot->want = id;
    std::thread([slot, key, dest, orient, side, id] {
        auto publish = [&](const std::string& jpeg, int stage) {
            int w, h, n;
            unsigned char* px = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(jpeg.data()), int(jpeg.size()), &w, &h, &n, 4);
            if (!px) return;
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
            if (a.search_mode == SM_ASK || a.search_mode == SM_SEMANTIC || a.search_mode == SM_REGEX) a.search_mode = SM_SMART;
            a.rows_dirty = true;
        }
        if (on) ImGui::PopStyleColor();
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem(tr("Unstar this tag"))) {
                std::string keep = t;
                a.cfg.fav_tags.erase(std::remove(a.cfg.fav_tags.begin(), a.cfg.fav_tags.end(), keep), a.cfg.fav_tags.end());
                save_config(a.cfg, a.config_file);
                ImGui::EndPopup();
                break;
            }
            ImGui::EndPopup();
        }
    }
}

// A star button for a tag (in tag lists): starred tags become filter chips.
static void tag_star(App& a, const std::string& tag) {
    auto& f = a.cfg.fav_tags;
    bool on = std::find(f.begin(), f.end(), tag) != f.end();
    ImGui::PushID(tag.c_str());
    ImGui::PushStyleColor(ImGuiCol_Text, on ? ImVec4(1, 0.8f, 0.25f, 1) : ImVec4(0.55f, 0.55f, 0.6f, 1));
    if (ImGui::SmallButton(on ? "\xE2\x98\x85" : "\xE2\x98\x86")) {
        if (on) f.erase(std::remove(f.begin(), f.end(), tag), f.end());
        else f.push_back(tag);
        save_config(a.cfg, a.config_file);
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", on ? tr("Unstar this tag") : tr("Star this tag: it becomes a one-click filter"));
    ImGui::PopID();
}

static void draw_viewer(App& a) {
    Viewer& v = a.viewer;
    if (!v.open) return;
    switch (nav_key(ImGui::GetFrameCount() != v.opened_frame)) {
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
            v.shown_stage = v.slot->stage;
            v.slot->ready = false;
        }
    }
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowFocus();
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.03f, 0.97f));
    ImGui::Begin("##viewer", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    const Photo& p = v.p;
    if (ImGui::Button("<")) { if (v.idx > 0) v.idx--; }
    ImGui::SameLine();
    if (ImGui::Button(">")) { if (v.idx + 1 < int(v.ids.size())) v.idx++; }
    ImGui::SameLine();
    std::string name = p.src_path;
    name = rel_path(a.cfg, name);
    ImGui::Text("%d / %zu   %s   ·   %s   ·   %s", v.idx + 1, v.ids.size(), name.c_str(),
                p.date_value.empty() ? tr("undated") : util::parse_db_datetime(p.date_value + "|" + p.date_prec).pretty().c_str(),
                util::human_size(p.size).c_str());
    ImGui::SameLine(ImGui::GetWindowWidth() - 300);
    if (ImGui::Button(tr("Open file"))) run_detached({"xdg-open", p.src_path});
    tip(tr("Open the photo in the default viewer"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Open folder"))) run_detached({"xdg-open", util::dirname(p.src_path)});
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
        float s = std::min(avail.x / a.viewer_tex.w, avail.y / a.viewer_tex.h);
        ImVec2 sz(a.viewer_tex.w * s, a.viewer_tex.h * s);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail.x - sz.x) / 2);
        ImGui::Image(ImTextureRef((ImTextureID)(intptr_t)a.viewer_tex.tex), sz);
    } else {
        ImGui::Dummy(avail);
    }
    ImGui::TextDisabled("%s", tr("Left/Right or H/L: previous/next  ·  Esc or Q: close"));
    ImGui::End();
    ImGui::PopStyleColor();
}

static void trash_duplicates(App& a) {
    // Only exact, verified duplicates inside this folder; the kept copy stays. The desktop Trash keeps them
    // restorable; without gio they go to <folder>/jev-duplicates instead of being deleted.
    std::vector<std::pair<int64_t, std::string>> todo;
    for (auto& r : a.dup_rows)
        if (r.dup_of && path_under(r.src_path, view_scope(a.cfg))) todo.push_back({r.id, r.src_path});
    if (todo.empty() || a.trash_busy.exchange(true)) return;
    a.trash_done = a.trash_failed = 0;
    std::vector<std::string> folders = a.cfg.folders;
    std::thread([&a, todo, folders] {
        bool gio = util::which("gio");
        std::vector<int64_t> gone;
        for (auto& [id, path] : todo) {
            bool ok = false;
            if (!util::file_exists(path)) ok = true;
            else if (gio) ok = util::run({"gio", "trash", "--", path}, "", 30).rc == 0;
            else {
                std::string folder = util::dirname(path);
                for (auto& f : folders)
                    if (path_under(path, f)) folder = f;
                std::string dest = folder + "/jev-duplicates/" + path.substr(folder.size() + 1);
                util::mkdirs(util::dirname(dest));
                ok = !util::file_exists(dest) && rename(path.c_str(), dest.c_str()) == 0;
            }
            if (ok) { gone.push_back(id); a.trash_done++; }
            else a.trash_failed++;
            glfwPostEmptyEvent();
        }
        a.log.add(0, util::fmt("moved %d duplicate copies to the %s (%d failed)", a.trash_done.load(),
                               gio ? "Trash" : "jev-duplicates folder", a.trash_failed.load()));
        a.view.write([gone, &a](Db& db) {
            for (int64_t id : gone) db.remove_photo(id);
            a.catalog_gen++;
            a.trash_busy = false;
            glfwPostEmptyEvent();
        });
    }).detach();
}

// Home: what the analysis found and the next step for each part, with its button.
static void draw_overview(App& a) {
    a.on_overview = true;
    ImVec4 acc(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 1);
    bool busy = a.pipe->running();
    if (a.cfg.folders.empty()) {
        ImGui::Dummy(ImVec2(0, 40));
        ImGui::TextUnformatted(tr("Choose a folder of photos to begin."));
        ImGui::TextDisabled("%s", tr("Nothing in it is changed: organized copies are made in a jev-organized folder inside it, after you approve a preview."));
        if (ImGui::Button(tr("Browse..."), ImVec2(200, 36))) start_dir_dialog(a, 3);
        tip(tr("Choose a folder with the file dialog"));
        return;
    }
    const DbStats& st = a.stats;
    int pending = count_plan(a.plan, false);
    int n_dup = 0;
    int64_t dup_bytes = 0;
    for (auto& r : a.dup_rows)
        if (r.dup_of) { n_dup++; dup_bytes += r.size; }
    int n_similar = 0;
    for (auto& r : a.dup_rows) n_similar += r.similar_to != 0;
    bool analyzed = st.total > 0;

    auto card = [&](int n, const char* title, bool done, auto body) {
        ImGui::PushID(n);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(1, 1, 1, 0.035f));
        ImGui::BeginChild("card", ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
        ImGui::TextColored(done ? ImVec4(0.45f, 0.9f, 0.55f, 1) : acc, "%s %d", done ? "\xE2\x9C\x93" : "\xE2\x97\x8B", n);
        ImGui::SameLine();
        ImGui::TextUnformatted(title);
        ImGui::Indent(28);
        body();
        ImGui::Unindent(28);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
    };
    ImGui::BeginChild("overview");
    card(1, tr("Analyze the folder"), analyzed, [&] {
        if (!analyzed) {
            ImGui::TextWrapped("%s", tr("Scans every photo, finds duplicates, works out dates and places, tags what is in each picture and plans the organized copies. Nothing is changed."));
        } else {
            ImGui::Text("%d %s · %s · %d %s", st.total, tr("photos"), util::human_size(st.total_bytes).c_str(), st.tagged, tr("tagged"));
            ImGui::TextDisabled("%s", tr("Run it again after adding photos: only new or changed files are read."));
            if (!a.corrections.empty()) {
                ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1), "%zu %s", a.corrections.size(),
                                   tr("corrections proposed for generated tags or names (with how sure each one is)"));
                ImGui::SameLine();
                if (ImGui::SmallButton(tr("Review corrections"))) a.start_tab = "corrections";
                tip(tr("Open Actions > Corrections"));
            }
        }
        ImGui::BeginDisabled(busy);
        if (ImGui::Button(analyzed ? tr("Analyze again") : tr("Analyze"), ImVec2(160, 0))) request_analyze(a);
        ImGui::EndDisabled();
    });
    if (!analyzed) { ImGui::EndChild(); return; }
    card(2, tr("Check uncertain dates"), st.review == 0 && st.undated == 0, [&] {
        ImGui::Text("%d %s  ·  %d %s  ·  %d %s", st.resolved - st.review - st.undated, tr("dated with confidence"), st.review, tr("uncertain"),
                    st.undated, tr("undated"));
        ImGui::TextDisabled("%s", tr("Uncertain photos are still filed under the best guess. Open one to see the evidence, or type the right date in its panel."));
        if (st.review + st.undated > 0 && ImGui::Button(tr("Show uncertain dates"), ImVec2(200, 0))) {
            a.review_only = true;
            a.search[0] = 0;
            a.rows_dirty = true;
            a.start_tab = "photos";
        }
        tip(tr("Show only photos whose date is a guess, to check or correct them"));
    });
    card(3, tr("Duplicates"), n_dup == 0, [&] {
        if (!n_dup) ImGui::TextUnformatted(tr("No exact duplicates."));
        else {
            ImGui::Text("%d %s (%s)", n_dup, tr("extra copies of photos you already have"), util::human_size(dup_bytes).c_str());
            ImGui::TextDisabled("%s", tr("They are left out of the organized copies automatically. Your folder keeps them until you remove them here."));
        }
        if (n_similar) ImGui::TextDisabled("%d %s", n_similar, tr("similar photos (resized copies or bursts) are listed for review only."));
        if (ImGui::Button(tr("Show duplicates"), ImVec2(160, 0))) a.start_tab = "dupes";
        tip(tr("Open Actions > Duplicates"));
        if (n_dup) {
            ImGui::SameLine();
            ImGui::BeginDisabled(busy || a.trash_busy);
            if (danger_button(tr("Move extra copies to Trash..."), tr("Shows exactly which files would move to the Trash (and which copy stays) before anything moves."), ImVec2(240, 0)))
                a.trash_confirm = true;
            ImGui::EndDisabled();
        }
        if (a.trash_busy) {
            ImGui::SameLine();
            ImGui::TextDisabled("%d / %d", a.trash_done.load(), n_dup);
        }
    });
    card(4, tr("Organize"), pending == 0 && st.organized > 0, [&] {
        std::string out = output_dir(a.cfg);
        if (pending) {
            int nn = 0;
            int64_t bb = 0;
            double ss = estimate_plan(a.plan, a.cfg, nn, bb);
            ImGui::Text("%d %s → %s  ·  %s  ·  %s %s", pending,
                        tr(a.cfg.file_op == OP_MOVE ? "photos to move" : a.cfg.file_op == OP_RENAME ? "photos to rename"
                           : a.cfg.file_op == OP_METADATA ? "photos to add metadata to" : "photos to copy"),
                        a.cfg.file_op == OP_RENAME || a.cfg.file_op == OP_METADATA ? tr("their own folders") : out.c_str(), util::human_size(bb).c_str(), tr("about"), fmt_duration(ss).c_str());
            ImGui::TextDisabled("%s", tr("Named by date (20190512_00001.jpg) in one folder per month. Check the list, untick anything you do not want, then apply."));
            if (ImGui::Button(tr("Review the list"), ImVec2(160, 0))) a.show_plan_tab = true;
            tip(tr("Open Actions > Organize files: every copy, rename and metadata addition, one per row"));
            ImGui::SameLine();
            ImGui::BeginDisabled(busy);
            if (apply_button(a.cfg, util::fmt("%s (%d)", tr("Apply"), pending).c_str(), ImVec2(160, 0))) apply_plan(a);
            ImGui::EndDisabled();
        } else if (st.organized) {
            ImGui::Text("%d %s %s", st.organized, tr("photos organized in"), out.c_str());
            if (ImGui::Button(tr("Open organized folder"), ImVec2(200, 0))) run_detached({"xdg-open", out});
            tip(tr("Open the folder with the organized copies"));
        } else {
            ImGui::TextDisabled("%s", tr("No plan yet."));
            ImGui::BeginDisabled(busy);
            if (ImGui::Button(tr("Plan the organized copies"), ImVec2(220, 0))) start_run(a, ST_ORGANIZE, true);
            tip(tr("Build the list of copies and names to review (nothing changes yet)"));
            ImGui::EndDisabled();
        }
    });
    card(5, tr("Find photos"), false, [&] {
        ImGui::TextDisabled("%s", tr("Describe what is in the picture (\"kids on a beach\"), or search names, places, tags, years and camera details."));
        static char q[256] = "";
        ImGui::SetNextItemWidth(420);
        bool go = ImGui::InputTextWithHint("##homesearch", tr("Search photos"), q, sizeof q, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if (ImGui::Button(tr("Search")) || go) {
            snprintf(a.search, sizeof a.search, "%s", q);
            a.review_only = false;
            a.rows_dirty = true;
            a.start_tab = "photos";
        }
        tip(tr("Search the photos for these words"));
    });
    card(6, tr("How jev helped"), false, [&] {
        const DecisionStats& ds = a.decision_stats;
        ImGui::TextDisabled("%s", tr("jev settles close calls: dates when the evidence disagrees, folder names that may be places, search results the LLM is unsure about, and it cross-checks the LLM on existing descriptions (a disagreement goes to you)."));
        if (!ds.total) {
            ImGui::TextUnformatted(tr("No jev decisions yet for this folder (every date and place was clear from the rules)."));
        } else {
            auto kind = [&](const char* k, const char* label) {
                auto it = ds.by_kind.find(k);
                int asked = it == ds.by_kind.end() ? 0 : it->second.first, ch = it == ds.by_kind.end() ? 0 : it->second.second;
                return util::fmt("%s %d (%d %s)", label, asked, ch, tr("changed"));
            };
            ImGui::Text("%d %s  ·  %s  ·  %s  ·  %s  ·  %s", ds.total, tr("decisions"), kind("date", tr("dates")).c_str(), kind("place", tr("places")).c_str(),
                        kind("search", tr("search")).c_str(), kind("description", tr("descriptions")).c_str());
        }
        if (ImGui::Button(a.show_decisions ? tr("Hide decisions") : tr("Show decisions"), ImVec2(160, 0))) a.show_decisions = !a.show_decisions;
        tip(a.show_decisions ? tr("Hide the list of jev decisions") : tr("Every question jev answered: what the rules or the LLM said, what jev chose, and the result"));
        std::string jd;
        {
            std::lock_guard<std::mutex> l(a.health_mu);
            jd = a.jev_detail;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("jev %s  ·  %s", a.jev_ok == 1 ? tr("up") : tr("down"), a.cfg.jev_url.c_str());
        if (a.show_decisions && !a.decisions.empty()) {
            ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV |
                                 ImGuiTableFlags_SizingFixedFit;
            if (ImGui::BeginTable("decisions", 6, tf, ImVec2(0, 320))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn(tr("Kind"), 0, 60);
                ImGui::TableSetupColumn(tr("About"), 0, 300);
                ImGui::TableSetupColumn(tr("Rules / LLM said"), 0, 190);
                ImGui::TableSetupColumn(tr("jev chose"), 0, 260);
                ImGui::TableSetupColumn(tr("Result"), 0, 150);
                ImGui::TableSetupColumn(tr("When"), 0, 140);
                ImGui::TableHeadersRow();
                for (auto& d : a.decisions) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(tr(d.kind.c_str()));
                    ImGui::TableNextColumn();
                    std::string about = d.kind == "search" ? "\"" + d.subject + "\"  " + d.question
                                                           : rel_path(a.cfg, d.subject);
                    ImGui::PushID(int(d.id));
                    ImGui::Selectable(about.c_str(), false, ImGuiSelectableFlags_SpanAllColumns);
                    if (ImGui::IsItemHovered()) {  // the question and every option with jev's probability
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted(d.question.c_str());
                        json o = json::parse(d.options.empty() ? "[]" : d.options, nullptr, false);
                        if (o.is_array())
                            for (auto& x : o)
                                if (x.is_array() && x.size() >= 2)
                                    ImGui::Text("%3.0f%%  %s", x[1].get<double>() * 100, x[0].get<std::string>().substr(0, 140).c_str());
                        ImGui::EndTooltip();
                    }
                    if (d.photo_id && ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, {d.photo_id}, 0);
                    ImGui::PopID();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(d.rules_pick.c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(d.jev_pick.substr(0, 80).c_str());
                    ImGui::TableNextColumn();
                    if (d.changed) ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s (%s)", d.final_pick.c_str(), tr("changed"));
                    else ImGui::TextUnformatted(d.final_pick.c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s", d.at.substr(0, 16).c_str());
                }
                ImGui::EndTable();
            }
        }
    });
    ImGui::EndChild();

}

// Duplicate removal preview (the dry run): a table of exactly what moves, what stays, totals and time.
static void draw_trash_modal(App& a) {
    if (a.trash_confirm) {
        ImGui::OpenPopup("###trash");
        a.trash_confirm = false;
    }
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(std::min(1100.0f, vp->WorkSize.x - 40), std::min(640.0f, vp->WorkSize.y - 40)), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal((std::string(tr("Remove extra copies")) + "###trash").c_str(), nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
    bool gio = util::which("gio");
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
    if (!a.cfg.verify_dupes) ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "%s", tr("Byte-verify is off: turn it on and check duplicates again first."));
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
            ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s", rel(r->src_path).c_str());
            ImGui::TableNextColumn();
            auto k = by.find(r->dup_of);
            ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.6f, 1), "%s", k != by.end() ? rel(k->second->src_path).c_str() : "?");
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

// The generation metadata of an AI image: tool, model, LoRAs, seed and settings, the prompt and its keywords.
static void draw_gen_info(App& a, const Photo& p) {
    (void)a;
    GenInfo g = GenInfo::from_json(p.gen_json);
    if (!g.found()) return;
    if (!ImGui::CollapsingHeader(util::fmt("%s (%s)###gen", tr("AI generation"), g.tool.c_str()).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (!g.model.empty()) ImGui::TextWrapped("%s: %s", tr("Model"), g.model.c_str());
    for (auto& [l, w] : g.loras) ImGui::TextWrapped("LoRA: %s (%.2f)", l.c_str(), w);
    std::string settings;
    if (!g.seed.empty()) settings += "seed " + g.seed + "  ";
    if (g.steps) settings += util::fmt("%d steps  ", g.steps);
    if (g.cfg > 0) settings += util::fmt("cfg %.1f  ", g.cfg);
    if (!g.sampler.empty()) settings += g.sampler + (g.scheduler.empty() ? "" : "/" + g.scheduler) + "  ";
    if (!g.size.empty()) settings += g.size;
    if (!settings.empty()) ImGui::TextDisabled("%s", settings.c_str());
    for (auto& src : g.sources) ImGui::TextDisabled("%s %s", tr("made from"), src.c_str());
    if (!g.prompt.empty()) {
        ImGui::TextUnformatted(tr("Prompt"));
        ImGui::SameLine();
        if (ImGui::SmallButton((std::string(tr("Copy")) + "##prompt").c_str())) ImGui::SetClipboardText(g.prompt.c_str());
        tip(tr("Copy the prompt to the clipboard"));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.85f, 1, 1));
        ImGui::TextWrapped("%s", g.prompt.c_str());
        ImGui::PopStyleColor();
    }
    if (!g.negative.empty() && ImGui::TreeNode(tr("Negative prompt"))) {
        ImGui::TextWrapped("%s", g.negative.c_str());
        ImGui::TreePop();
    }
    std::string kw;
    for (auto& [k, c] : parse_tag_list(p.gen_keywords)) kw += (kw.empty() ? "" : ", ") + k;
    if (!kw.empty()) ImGui::TextWrapped("%s: %s", tr("Prompt keywords"), kw.c_str());
    ImGui::TextDisabled("%s %s (%s)", tr("found in"), g.field.c_str(), tr("read-only: never changed"));
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
        ImGui::Image(ImTextureRef((ImTextureID)(intptr_t)a.thumb.tex), ImVec2(a.thumb.w * s, a.thumb.h * s));
    } else {
        ImGui::Dummy(ImVec2(avail, 60));
    }
    ImGui::TextWrapped("%s", p.src_path.c_str());
    if (!p.dest_path.empty()) ImGui::TextColored(ImVec4(0.55f, 0.85f, 0.6f, 1), "-> %s", p.dest_path.c_str());
    std::string open_target = p.dest_path.empty() ? p.src_path : p.dest_path;
    ImGui::TextDisabled("%s  ·  %d x %d  ·  #%lld", util::human_size(p.size).c_str(), p.width, p.height, (long long)p.id);
    if (p.dup_of) ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1), "%s #%lld", tr("duplicate of"), (long long)p.dup_of);
    if (p.similar_to) ImGui::TextColored(ImVec4(0.7f, 0.75f, 1, 1), "%s #%lld (dHash %d)", tr("similar to"), (long long)p.similar_to, p.similar_dist);
    ImGui::PushStyleColor(ImGuiCol_Text, p.favorite ? ImVec4(1, 0.8f, 0.25f, 1) : ImGui::GetStyleColorVec4(ImGuiCol_Text));
    if (ImGui::SmallButton(p.favorite ? "\xE2\x98\x85 Favorite" : "\xE2\x98\x86 Favorite")) set_favorite(a, p.id, !p.favorite);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("f also stars the selected photo"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Find similar"))) {  // CLIP image-to-image: like:#id in the search box
        snprintf(a.search, sizeof a.search, "like:#%lld", (long long)p.id);
        if (a.search_mode != SM_SMART && a.search_mode != SM_KEYWORD) a.search_mode = SM_SMART;
        a.month.clear();
        a.rows_dirty = true;
        a.start_tab = "photos";
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Photos that look like this one (CLIP); also: like:#id or like:<file name> in the search box"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Open file"))) run_detached({"xdg-open", open_target});
    tip(tr("Open the photo in the default viewer"));
    ImGui::SameLine();
    if (ImGui::SmallButton(tr("Open folder"))) run_detached({"xdg-open", util::dirname(open_target)});
    tip(tr("Open the folder that holds this photo"));
    ImGui::Separator();

    json ev = json::parse(p.date_evidence.empty() ? "{}" : p.date_evidence, nullptr, false);
    ImGui::Text("%s: %s", tr("Date"), p.date_value.empty() ? tr("undated") : p.date_value.c_str());
    ImGui::Text("%s: %s  ·  %s: %.2f  ·  %s: %s", tr("Precision"), p.date_prec.c_str(), tr("Conf"), p.date_conf, tr("Decider"),
                p.date_decider.c_str());
    ImGui::TextWrapped("%s: %s", tr("Source"), p.date_source.c_str());
    if (ev.is_object() && ev.contains("jev")) ImGui::TextDisabled("%s", ev["jev"].get<std::string>().c_str());
    // jev's part in this photo: its date question, its folder's place question, searches that judged it.
    for (auto& d : a.decisions) {
        bool mine = (d.kind == "date" && d.photo_id == p.id) || (d.kind == "place" && d.subject == util::dirname(p.src_path)) ||
                    (d.kind == "search" && d.photo_id == p.id);
        if (!mine) continue;
        ImVec4 col = d.changed ? ImVec4(0.95f, 0.72f, 0.4f, 1) : ImVec4(0.6f, 0.75f, 1, 1);
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
                ImVec4 col = winner ? ImVec4(0.45f, 0.95f, 0.55f, 1) : rejected ? ImVec4(0.55f, 0.55f, 0.55f, 1) : ImVec4(0.9f, 0.9f, 0.9f, 1);
                ImGui::TableNextColumn();
                ImGui::TextColored(col, "%s%s", winner ? "\xE2\x96\xB6 " : "", c.value("src", "").c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(col, "%s", c.value("when", "").c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", c.value("raw", "").c_str());
                if (c.contains("notes"))
                    for (auto& n : c["notes"]) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.72f, 0.4f, 1));
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
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Manual date (YYYY-MM-DD HH:MM:SS, YYYY-MM-DD or YYYY-MM)"));
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
    tip(tr("Use this date for the photo; it wins over everything else on the next Decide and Organize"));
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
    ImGui::Text("%s: %s", tr("Location"), p.location.empty() ? "-" : p.location.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s, %.2f)", p.location_source.c_str(), p.location_conf);
    if (p.has_gps) ImGui::Text("GPS: %.6f, %.6f", p.gps_lat, p.gps_lon);
    if (ImGui::CollapsingHeader(tr("Location evidence"))) {
        DirPlace d = a.ui_db.get_dir(util::dirname(p.src_path));
        ImGui::TextWrapped("%s", d.evidence.c_str());
    }
    draw_gen_info(a, p);
    if (!p.clip_model.empty() && ImGui::CollapsingHeader(tr("Tags (CLIP)"), ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!p.clip_scene.empty()) ImGui::Text("%s: %s", tr("Scene"), p.clip_scene.c_str());
        for (auto& [t, conf] : effective_tag_list(p)) {  // corrections applied; star a tag to make it a filter chip
            tag_star(a, t);
            ImGui::SameLine();
            ImGui::Text("%s  %.0f%%", t.c_str(), conf * 100);
        }
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
        if (!m.error.empty()) ImGui::TextColored(ImVec4(0.95f, 0.5f, 0.4f, 1), "%s", m.error.c_str());
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
        ImGui::SameLine();
        if (ImGui::SmallButton("<")) { if (--m < 1) { m = 12; y--; } }
        ImGui::SameLine();
        ImGui::Text("  %04d-%02d  ", y, m);
        ImGui::SameLine();
        if (ImGui::SmallButton(">")) { if (++m > 12) { m = 1; y++; } }
        ImGui::SameLine();
        if (ImGui::SmallButton(">>")) y++;
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
        st->SetInt(ky, y);
        st->SetInt(km, m);
        ImGui::EndPopup();
    }
    ImGui::PopID();
    return changed;
}

static void draw_photos(App& a) {
    double now = glfwGetTime();
    if (a.cfg.folders.empty()) {
        ImGui::Dummy(ImVec2(0, 60));
        float w = ImGui::GetContentRegionAvail().x;
        const char* msg = tr("Choose a folder of photos to begin.");
        ImGui::SetCursorPosX((w - ImGui::CalcTextSize(msg).x) / 2);
        ImGui::TextUnformatted(msg);
        const char* sub = tr("Nothing in it is changed: organized copies are made in a jev-organized folder inside it, after you approve a preview.");
        ImGui::SetCursorPosX(std::max(0.0f, (w - ImGui::CalcTextSize(sub).x) / 2));
        ImGui::TextDisabled("%s", sub);
        ImGui::Dummy(ImVec2(0, 10));
        ImGui::SetCursorPosX((w - 200) / 2);
        if (ImGui::Button(tr("Browse..."), ImVec2(200, 36))) start_dir_dialog(a, 3);
        tip(tr("Choose a folder with the file dialog"));
        return;
    }
    (void)now;
    if (a.stats.total == 0 && !a.pipe->running()) {
        if (a.view.loading()) {
            ImGui::TextDisabled("%s", tr("Loading..."));
            return;
        }
        ImGui::TextWrapped("%s", tr("No photos from this folder yet. Press Analyze."));
        return;
    }
    // Month sidebar
    ImGui::BeginChild("months", ImVec2(190, 0), ImGuiChildFlags_Borders);
    if (ImGui::Selectable(util::fmt("%s (%d)", tr("All months"), a.stats.total).c_str(), a.month.empty())) { a.month.clear(); a.rows_dirty = true; }
    ImGui::TextDisabled("%s", util::human_size(a.stats.total_bytes).c_str());
    if (a.stats.dups) ImGui::TextDisabled("%d %s · %s", a.stats.dups, tr("duplicates"), util::human_size(a.stats.dup_bytes).c_str());
    std::string year;
    for (auto& mi : a.months) {
        const std::string& m = mi.month;
        int n = mi.count;
        if (m != "undated" && m.substr(0, 4) != year) {
            year = m.substr(0, 4);
            ImGui::SeparatorText(year.c_str());
        }
        std::string label = (m == "undated" ? std::string(tr("undated")) : m) + util::fmt(" (%d)", n);
        if (ImGui::Selectable(label.c_str(), a.month == m)) { a.month = m; a.rows_dirty = true; }
        ImGui::SameLine(118);
        ImGui::TextDisabled("%s", util::human_size(mi.bytes).c_str());
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginGroup();
    // Search bar: one box, a mode, and filters tucked into a popup.
    ImGui::SetNextItemWidth(std::max(200.0f, ImGui::GetContentRegionAvail().x * 0.45f));
    // Ask asks the LLM, so it runs on Enter; the other modes search as you type.
    if (ImGui::InputTextWithHint("##search", a.search_mode == SM_ASK ? tr("Ask: \"my dog at the beach last summer\" (Enter)")
                                                                   : tr("Search: beach (sunset | sea) -night, tag:dog, is:fav, \"New York\", 海边"),
                                 a.search, sizeof a.search, a.search_mode == SM_ASK ? ImGuiInputTextFlags_EnterReturnsTrue : 0)) {
        a.rows_dirty = true;
        a.search_edit_at = glfwGetTime();  // other-language equivalents are looked up once typing pauses
        a.search_translate = false;
    }
    if (ImGui::IsItemHovered() && a.search_mode != SM_ASK)
        ImGui::SetTooltip("%s", tr("beach sunset      both\nbeach | sea, beach OR sea      either\n-night, NOT night, !night      without\n(beach | sea) -night      grouping\n\"new york\"      phrase\n~webiste      close spellings too\ntag:dog  name:IMG  desc:..  prompt:..  exif:..  place:paris      one field\ncamera:canon  model:sdxl  lora:ink      camera / generator\ndate:2019  date:2019-05..2019-08  date:>2020  year:2018..2020\nsize:>2mb  w:>3000  h:<1000  mp:>12  ext:png  ext:raw\nis:fav  is:ai  is:untagged  is:uncertain  is:portrait  is:landscape\nhas:gps  has:prompt  has:place  has:desc  has:keywords\nlike:#123  like:IMG_0042      looks like that photo\nWords in Chinese, Japanese or Korean also find English tags (and the other way round)."));
    ImGui::SameLine();
    const char* modes[] = {tr("Smart"), tr("Words"), tr("Meaning"), tr("Regex"), tr("Ask (AI)")};
    ImGui::SetNextItemWidth(110);
    if (ImGui::Combo("##mode", &a.search_mode, modes, SM_COUNT)) a.rows_dirty = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tr("Smart: words and meaning together\nWords: every word must appear (\"quoted phrase\", -exclude)\nMeaning: what the picture really shows (CLIP)\nRegex: regular expression\nAsk (AI): a sentence such as \"my dog at the beach last summer\"; press Enter"));
    ImGui::SameLine();
    int nfilters = (!a.in_name || !a.in_exif || !a.in_desc || !a.in_prompt) + (!a.date_from.empty() || !a.date_to.empty()) + a.review_only;
    if (ImGui::Button(nfilters ? util::fmt("%s (%d)", tr("Filters"), nfilters).c_str() : tr("Filters"))) ImGui::OpenPopup("filters");
    tip(tr("Choose the fields to search, a date range, and other filters"));
    if (ImGui::BeginPopup("filters")) {
        ImGui::TextDisabled("%s", tr("Look for words in"));
        if (ImGui::Checkbox(tr("File and folder names"), &a.in_name)) a.rows_dirty = true;
        tip(tr("Search in file and folder names"));
        if (ImGui::Checkbox(tr("Descriptions and tags"), &a.in_desc)) a.rows_dirty = true;
        tip(tr("Search in tags, scene, place and descriptions"));
        if (ImGui::Checkbox(tr("EXIF / metadata"), &a.in_exif)) a.rows_dirty = true;
        tip(tr("Search in the camera's and the file's own metadata"));
        if (ImGui::Checkbox(tr("AI generation prompts"), &a.in_prompt)) a.rows_dirty = true;
        tip(tr("Search in the prompts AI images were generated from (and their model and LoRA names)"));
        ImGui::Separator();
        ImGui::TextDisabled("%s", tr("Taken between"));
        if (date_picker("from", a.date_from, tr("any start"))) a.rows_dirty = true;
        ImGui::SameLine();
        ImGui::TextUnformatted(tr("and"));
        ImGui::SameLine();
        if (date_picker("to", a.date_to, tr("any end"))) a.rows_dirty = true;
        if (!a.date_from.empty() && !a.date_to.empty() && a.date_from > a.date_to) std::swap(a.date_from, a.date_to);
        int this_year = atoi(util::now_iso().substr(0, 4).c_str());
        if (ImGui::SmallButton(tr("This year"))) { a.date_from = util::fmt("%d-01-01", this_year); a.date_to = util::fmt("%d-12-31", this_year); a.rows_dirty = true; }
        ImGui::SameLine();
        if (ImGui::SmallButton(tr("Last year"))) { a.date_from = util::fmt("%d-01-01", this_year - 1); a.date_to = util::fmt("%d-12-31", this_year - 1); a.rows_dirty = true; }
        ImGui::Separator();
        if (ImGui::Checkbox(tr("Only photos with uncertain dates"), &a.review_only)) a.rows_dirty = true;
        tip(tr("Only photos whose date is a guess"));
        if (ImGui::Button(tr("Clear filters"))) {
            a.in_name = a.in_exif = a.in_desc = a.in_prompt = true;
            a.date_from.clear();
            a.date_to.clear();
            a.review_only = false;
            a.rows_dirty = true;
        }
        tip(tr("Search all fields, any date"));
        ImGui::EndPopup();
    }
    if (a.search[0] || nfilters) {
        ImGui::SameLine();
        if (ImGui::Button("x")) {
            a.search[0] = 0;
            a.in_name = a.in_exif = a.in_desc = a.in_prompt = true;
            a.date_from.clear();
            a.date_to.clear();
            a.review_only = false;
            a.rows_dirty = true;
        }
    }
    // Favorites are a filter of the photos; the grid is another way to look at them.
    ImGui::SameLine(0, 14);
    {
        bool on = a.fav_only;
        if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1, 0.75f, 0.2f, 0.55f));
        if (ImGui::Button(util::fmt("\xE2\x98\x85 %s", tr("Favorites")).c_str())) { a.fav_only = !a.fav_only; a.rows_dirty = true; }
        if (on) ImGui::PopStyleColor();
        tip(tr("Show only starred photos. Star a photo with f, or with the star in the list."));
        ImGui::SameLine();
        if (ImGui::Button(a.photo_grid ? tr("List") : tr("Grid"))) a.photo_grid = !a.photo_grid;
        tip(tr("Switch between the table and a grid of thumbnails (the same photos and keys)"));
    }
    ImGui::SameLine();
    if (a.view.loading()) ImGui::TextDisabled("%s", tr("searching..."));
    else if (a.searching) ImGui::TextDisabled("%zu %s", a.rows.size(), tr("matches"));
    else ImGui::TextDisabled("%zu / %d", a.rows.size(), a.stats.total);
    draw_fav_tag_chips(a);
    if (!a.search_error.empty()) ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", a.search_error.c_str());
    else if (!a.search_note.empty()) ImGui::TextDisabled("%s", a.search_note.c_str());

    float detail_w = std::max(360.0f, ImGui::GetContentRegionAvail().x * 0.36f);
    ImGui::BeginChild("table", ImVec2(ImGui::GetContentRegionAvail().x - detail_w - 8, 0));
    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable |
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit;
    bool srch = a.searching;
    if (a.photo_grid) draw_photo_grid(a);
    else
    tf |= ImGuiTableFlags_Sortable;
    // Separate table ids: browsing sorts by date by default, a search by relevance.
    if (ImGui::BeginTable(srch ? "photos_s" : "photos", 6, tf)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("\xE2\x98\x85", ImGuiTableColumnFlags_PreferSortDescending, 22);
        ImGui::TableSetupColumn(tr("Date"), srch ? 0 : ImGuiTableColumnFlags_DefaultSort, 130);
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
            if (ms[size_t(mi)] != a.month) { a.month = ms[size_t(mi)]; a.rows_dirty = true; a.selected = 0; }
        } else if ((nk == NK_ENTER || nk == NK_SPACE) && sel_idx >= 0) {
            std::vector<int64_t> ids;
            for (auto& r : a.rows) ids.push_back(r.id);
            open_viewer(a, ids, sel_idx);
        } else if (nk == NK_FAV && sel_idx >= 0) {
            set_favorite(a, a.rows[size_t(sel_idx)].id, !a.rows[size_t(sel_idx)].favorite);
        }
        ImGuiListClipper clip;
        clip.Begin(int(a.rows.size()));
        if (a.scroll_sel && sel_idx >= 0) clip.IncludeItemByIndex(sel_idx);
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                const PhotoRow& r = a.rows[size_t(i)];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID(int(r.id));
                ImGui::PushStyleColor(ImGuiCol_Text, r.favorite ? ImVec4(1, 0.8f, 0.25f, 1) : ImVec4(0.4f, 0.4f, 0.45f, 1));
                if (ImGui::SmallButton(r.favorite ? "\xE2\x98\x85" : "\xE2\x98\x86")) set_favorite(a, r.id, !r.favorite);
                ImGui::PopStyleColor();
                ImGui::TableNextColumn();
                std::string d = r.date_value.empty() ? tr("undated") : util::parse_db_datetime(r.date_value + "|" + r.date_prec).pretty();
                // Uncertain dates are shown in amber; details in the side panel.
                ImGui::PushStyleColor(ImGuiCol_Text, r.needs_review ? ImVec4(0.95f, 0.72f, 0.4f, 1) : ImGui::GetStyleColorVec4(ImGuiCol_Text));
                if (ImGui::Selectable(d.c_str(), a.selected == r.id, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
                    a.selected = r.id;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        std::vector<int64_t> ids;
                        for (auto& x : a.rows) ids.push_back(x.id);
                        open_viewer(a, ids, i);
                    }
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
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("detail", ImVec2(0, 0), ImGuiChildFlags_Borders);
    draw_detail(a);
    ImGui::EndChild();
    ImGui::EndGroup();
}

// Example of the chosen naming for the Review header and the Tags & EXIF tab.
static std::string example_name(const Config& c, const std::string& src_stem, const std::string& ext, const std::string& day = "20190512") {
    std::string prefix = c.name_style == NAME_KEEP_PREFIX ? name_prefix(src_stem) : "";
    return (prefix.empty() ? "" : prefix + c.name_sep) + day + c.name_sep + util::fmt("%0*d", c.sn_digits, 1) + ext;
}

// What the Review list does: one row of choices. Changing any of them rebuilds the list.
static void draw_plan_controls(App& a) {
    Config& c = a.cfg;
    bool busy = a.pipe->running();
    bool changed = false;
    ImGui::BeginDisabled(busy);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(tr("Do"));
    ImGui::SameLine();
    const char* ops[] = {tr("Organize copies"), tr("Move into month folders"), tr("Rename in place"), tr("Add metadata only")};
    const char* tips[] = {tr("Copy each photo into a month folder with a date name; your originals are untouched."),
                          tr("Move each photo into a month folder with a date name; no second copy uses disk space."),
                          tr("Give each photo a date name in the folder it is already in."),
                          tr("Do not move or rename anything: only fill in missing dates, tags, descriptions and places in the files.")};
    for (int i = 0; i < OP_COUNT; i++) {
        if (i) ImGui::SameLine();
        bool changes_originals = i == OP_MOVE || i == OP_RENAME;  // shown in the warning colour
        if (changes_originals) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.55f, 1));
        if (ImGui::RadioButton(ops[i], c.file_op == i)) { c.file_op = i; changed = true; }
        if (changes_originals) ImGui::PopStyleColor();
        tip(tips[i]);
    }
    ImGui::SameLine(0, 24);
    ImGui::BeginDisabled(c.file_op == OP_METADATA);
    ImGui::TextUnformatted(tr("Names"));
    ImGui::SameLine();
    std::string n0 = example_name(Config(c), "", ".jpg");
    Config k = c;
    k.name_style = NAME_KEEP_PREFIX;
    std::string n1 = example_name(k, "IMG_20190512_123456", ".jpg") + "  (" + tr("keeps the original name") + ")";
    const char* names[] = {n0.c_str(), n1.c_str()};
    ImGui::SetNextItemWidth(330);
    if (ImGui::Combo("##names", &c.name_style, names, NAME_COUNT)) changed = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tr("Keep the original name: its date, time, serial number and copy markers are removed, then the date and serial are added.\nIMG_20190512_123456.jpg -> IMG_20190512_00001.jpg\nParis trip 2019-05-12 (3).jpg -> Paris trip_20190512_00001.jpg"));
    ImGui::EndDisabled();
    bool meta = c.write_mode != WRITE_DB_ONLY;
    if (ImGui::Checkbox(tr("Write metadata into the files"), &meta)) {
        c.write_mode = meta ? WRITE_EMBED : WRITE_DB_ONLY;
        changed = true;
    }
    tip(tr("Dates, tags, descriptions and places are added where the file has none. Existing values are never changed."));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Dates, tags, descriptions and places are added where the file has none. Existing values are never changed."));
    if (meta) {
        ImGui::SameLine();
        const char* wm[] = {tr("inside the file"), tr("in a .xmp sidecar")};
        int w = c.write_mode == WRITE_SIDECAR ? 1 : 0;
        ImGui::SetNextItemWidth(160);
        if (ImGui::Combo("##wm", &w, wm, 2)) { c.write_mode = w ? WRITE_SIDECAR : WRITE_EMBED; changed = true; }
        ImGui::SameLine();
        if (ImGui::Checkbox(tr("Replace tags written earlier"), &c.rewrite_tags)) changed = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Keywords jev-photos added on an earlier run are updated to the current tags. Keywords the file had on its own are kept."));
    }
    ImGui::EndDisabled();
    if (changed) {
        save_config(c, a.config_file);
        if (!busy && !a.cfg.folders.empty()) start_run(a, ST_ORGANIZE, true);  // show the new list right away
    }
}

static void draw_plan(App& a) {
    bool busy = a.pipe->running();
    if (!busy && glfwGetTime() - a.plan_at > 1.0) {  // pick up include/exclude and apply results
        a.plan = a.pipe->plan();
        a.plan_at = glfwGetTime();
    }
    draw_plan_controls(a);
    ImGui::Separator();
    if (a.plan.empty()) {
        if (a.pipe->progress.finished_runs > 0 && a.stats.organized > 0 && a.last_stages == ST_ORGANIZE)
            ImGui::TextWrapped("%s", util::fmt(tr("Nothing to do: all %d photos are already organized and their metadata is up to date. To add missing fields to the files, choose \"Add metadata only\"."), a.stats.organized).c_str());
        else
            ImGui::TextWrapped("%s", tr("Nothing to review yet. Press Analyze: every copy, rename and metadata addition is listed here first, and nothing changes until you press Apply."));
        ImGui::BeginDisabled(busy || a.cfg.folders.empty());
        if (ImGui::Button(tr("Build the list"))) start_run(a, ST_ORGANIZE, true);
        tip(tr("List every change that would be made, to review before anything happens"));
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
    const char* opname = a.cfg.file_op == OP_MOVE ? "Move and rename" : a.cfg.file_op == OP_RENAME ? "Rename in place"
                       : a.cfg.file_op == OP_METADATA ? "Add missing metadata" : "Copy and rename";
    ImGui::Text("%s:  %d %s  ·  %s  ·  %s %s", tr(opname), n_todo, tr("files"), util::human_size(bytes).c_str(), tr("about"), fmt_duration(secs).c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("   (%d %s · %d %s · %d %s · %d %s)", n_refile, tr("to re-file"), n_meta, tr("metadata updates"), n_dup,
                        tr("duplicates skipped"), n_rev, tr("need review"));
    if (n_ok || n_err) {
        ImGui::SameLine();
        ImGui::TextColored(n_err ? ImVec4(1, 0.5f, 0.4f, 1) : ImVec4(0.5f, 0.9f, 0.6f, 1), "  ·  %d %s, %d %s", n_ok, tr("applied"), n_err, tr("failed"));
    }
    if (a.cfg.file_op == OP_METADATA)
        ImGui::TextDisabled("%s", tr("Nothing is moved or renamed. Missing fields are added to each photo where it is now (its organized copy if it has one)."));
    else if (a.cfg.file_op != OP_COPY)
        ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s", tr("Your original files will be renamed/moved (no copies are kept)."));
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
    if (ImGui::Button(tr("Refresh preview"))) start_run(a, ST_ORGANIZE, true);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Rebuild the plan; excluded items are left out and the serial numbers close up."));
    ImGui::SameLine();
    std::string apply_label = util::fmt("%s (%d)", tr("Apply all"), n_inc);
    if (n_inc == 0 && n_ok > 0) {  // everything applied
        ImGui::BeginDisabled();
        ImGui::Button(tr("All applied"));
        ImGui::EndDisabled();
        tip(tr("Every change in this list has been made. Analyze again, or change the choices above, for a new list."));
    } else if (apply_button(a.cfg, apply_label.c_str()) && n_inc > 0) {
        apply_plan(a);
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 20);
    ImGui::Checkbox(tr("Needs review only"), &a.plan_review_only);
    tip(tr("Show only rows whose date is a guess"));
    if (n_decide) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.72f, 0.4f, 1));
        ImGui::Checkbox(util::fmt("%d %s", n_decide, tr("descriptions to decide")).c_str(), &a.plan_decide_only);
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("These photos already have a description. The LLM and jev did not agree on keeping, appending or replacing it, so you choose (until then it is left as it is)."));
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220);
    ImGui::InputTextWithHint("##pf", tr("Search"), a.plan_filter, sizeof a.plan_filter);

    std::vector<int> vis;
    std::string f = util::lower(util::trim(a.plan_filter));
    for (int i = 0; i < int(a.plan.size()); i++) {
        auto& it = a.plan[size_t(i)];
        if (a.plan_review_only && !it.needs_review) continue;
        if (a.plan_decide_only && it.desc_action != "decide") continue;
        if (!f.empty() && util::lower(it.src + " " + it.dest + " " + it.date + " " + it.location).find(f) == std::string::npos) continue;
        vis.push_back(i);
    }
    float detail_h = 230;
    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_Resizable |
                         ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingFixedFit;
    auto rel = [&](const std::string& p) {
        return rel_path(a.cfg, p);
    };
    if (ImGui::BeginTable("plan", 8, tf, ImVec2(0, ImGui::GetContentRegionAvail().y - detail_h))) {
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
                ImGui::EndDisabled();
                ImGui::TableNextColumn();
                ImVec4 ac = dup ? ImVec4(0.55f, 0.55f, 0.55f, 1) : it.action == "refile" ? ImVec4(0.95f, 0.75f, 0.35f, 1)
                          : it.action == "move" ? ImVec4(0.95f, 0.55f, 0.45f, 1) : it.action == "metadata" ? ImVec4(0.6f, 0.75f, 1, 1)
                                                                                                             : ImVec4(0.55f, 0.9f, 0.6f, 1);
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
                ImGui::TextColored(it.needs_review ? ImVec4(0.95f, 0.72f, 0.4f, 1) : ImGui::GetStyleColorVec4(ImGuiCol_Text), "%s", it.date.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(util::human_size(it.size).c_str());
                ImGui::TableNextColumn();
                if (!it.meta_cmds.empty()) ImGui::Text("+%d%s", it.meta_added, it.meta_removed.empty() ? "" : util::fmt(" -%zu", it.meta_removed.size()).c_str());
                if (it.desc_action == "decide") { ImGui::SameLine(); ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s", tr("desc?")); }
                else if (it.desc_action == "append" || it.desc_action == "replace") { ImGui::SameLine(); ImGui::TextDisabled("%s", tr(("desc " + it.desc_action).c_str())); }
                ImGui::TableNextColumn();
                if (it.result == "ok") ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.6f, 1), "%s", tr("applied"));
                else if (!it.result.empty()) ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", it.result.c_str());
                else if (!dup) ImGui::TextDisabled("%s", it.note.c_str());
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    ImGui::BeginChild("plan_detail", ImVec2(0, 0), ImGuiChildFlags_Borders);
    if (a.plan_sel >= 0 && a.plan_sel < int(a.plan.size())) {
        const PlanItem& it = a.plan[size_t(a.plan_sel)];
        ImGui::Text("%s  %s", tr(it.action.c_str()), it.src.c_str());
        if (!it.old_dest.empty()) ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1), "%s -> %s", it.old_dest.c_str(), it.dest.c_str());
        else if (it.action != "duplicate") ImGui::TextColored(ImVec4(0.55f, 0.9f, 0.6f, 1), "-> %s", it.dest.c_str());
        if (!it.desc_new.empty() && it.desc_action != "add" && !it.desc_action.empty()) {
            // An existing description: what it says, what ours says, who decided, and your override.
            ImGui::SeparatorText(tr("Description"));
            ImGui::TextDisabled("%s", tr("In the file:"));
            ImGui::SameLine();
            ImGui::TextWrapped("%s", it.desc_old.c_str());
            ImGui::TextDisabled("%s", tr("Ours:"));
            ImGui::SameLine();
            ImGui::TextWrapped("%s", it.desc_new.c_str());
            ImGui::TextColored(it.desc_action == "decide" ? ImVec4(0.95f, 0.72f, 0.4f, 1) : ImVec4(0.6f, 0.75f, 1, 1), "%s: %s  (%s)  %s",
                               tr("Decision"), tr(it.desc_action == "decide" ? "you decide" : it.desc_action.c_str()), it.desc_by.c_str(), it.desc_why.c_str());
            ImGui::BeginDisabled(busy || it.desc_action == "update" || it.desc_by == "same");
            for (const char* act : {"keep", "append", "replace"}) {
                ImGui::SameLine();
                bool on = it.desc_action == act;
                bool repl = std::string(act) == "replace";
                if (on) ImGui::PushStyleColor(ImGuiCol_Button, repl ? ImVec4(kDanger.x, kDanger.y, kDanger.z, 0.75f) : ImVec4(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 0.6f));
                if (repl) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.62f, 0.55f, 1));
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
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Replace keeps the old text in Xmp.jev.PreviousDescription."));
        }
        if (!it.meta_target.empty()) {
            ImGui::TextDisabled("%s %s  (%s)", tr("Metadata additions to"), it.meta_target.c_str(),
                                tr("existing values are never changed; Xmp.jev.* is provenance"));
            for (auto& c : it.meta_cmds) {
                bool ours = util::starts_with(c, "set Xmp.jev.");
                ImGui::TextColored(ours ? ImVec4(0.6f, 0.6f, 0.65f, 1) : ImVec4(0.75f, 0.9f, 1, 1), "%s", c.c_str());
            }
            for (auto& k : it.meta_removed)
                ImGui::TextColored(ImVec4(0.95f, 0.6f, 0.5f, 1), "- %s  (%s)", k.c_str(), tr("keyword jev-photos wrote earlier, replaced by the current tags"));
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
    if (ImGui::Button(tr("Tag untagged photos"))) {
        RunOptions o = a.opts;
        o.stages = ST_TAG;
        o.scope = all_scope(c);
        a.last_stages = ST_TAG;
        a.pipe->start(effective(c), o);
    }
    tip(tr("Analyse new photos with CLIP and tag them; refresh tags made with an older tag list"));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Runs CLIP on photos without tags, and refreshes tags made with an older tag list."));
    ImGui::SameLine();
    if (ImGui::Button(tr("Re-tag all"))) {
        RunOptions o = a.opts;
        o.stages = ST_TAG;
        o.scope = all_scope(c);
        o.retag_all = true;
        a.last_stages = ST_TAG;
        a.pipe->start(effective(c), o);
    }
    tip(tr("Recompute every photo's tags from its stored analysis (seconds: no picture is read). Your own tags are kept."));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Recomputes every photo's tags from its stored CLIP embedding (fast; no image is read again). Your own edits are kept."));
    ImGui::SameLine();
    if (ImGui::Button(tr("Edit tag list..."))) { load_tag_vocabulary(); run_detached({"xdg-open", tag_vocabulary_path()}); }
    ImGui::SameLine(0, 30);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(c.accent[0], c.accent[1], c.accent[2], 0.70f));
    if (ImGui::Button(tr("Fill in missing metadata..."))) {
        c.file_op = OP_METADATA;
        if (c.write_mode == WRITE_DB_ONLY) c.write_mode = WRITE_EMBED;
        save_config(c, a.config_file);
        start_run(a, ST_ORGANIZE, true);
        a.show_plan_tab = true;
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Lists, for review, every date, tag, description and place that would be added to the files (nothing is overwritten). Apply it in Review."));
    ImGui::SameLine();
    if (ImGui::Button(tr("Rename..."))) {
        if (c.file_op == OP_METADATA) c.file_op = OP_RENAME;
        save_config(c, a.config_file);
        start_run(a, ST_ORGANIZE, true);
        a.show_plan_tab = true;
    }
    tip(tr("Build the rename / organize list with the naming chosen here, then review it in Organize files"));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Builds the rename / organize list with the naming chosen below."));
    ImGui::EndDisabled();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(tr("File names"));
    ImGui::SameLine();
    if (ImGui::RadioButton(tr("Date and serial"), c.name_style == NAME_DATE_SN)) { c.name_style = NAME_DATE_SN; save_config(c, a.config_file); }
    ImGui::SameLine();
    if (ImGui::RadioButton(tr("Original name, then date and serial"), c.name_style == NAME_KEEP_PREFIX)) { c.name_style = NAME_KEEP_PREFIX; save_config(c, a.config_file); }

    // Filter
    const char* filters[] = {tr("All photos"), tr("Without tags"), tr("Weak tags only"), tr("Older tag list"), tr("Edited by me"),
                             tr("No date in the file"), tr("No keywords in the file"), tr("No description in the file"), tr("No place in the file"), tr("AI-generated (prompt found)")};
    ImGui::SetNextItemWidth(230);
    ImGui::Combo("##tagfilter", &a.tags_filter, filters, IM_ARRAYSIZE(filters));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(260);
    ImGui::InputTextWithHint("##tagsearch", tr("Filter by file or tag"), a.tags_search, sizeof a.tags_search);
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

    float detail_w = std::max(380.0f, ImGui::GetContentRegionAvail().x * 0.34f);
    ImGui::BeginChild("tagtable", ImVec2(ImGui::GetContentRegionAvail().x - detail_w - 8, 0));
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
                    ImGui::TextColored(ImVec4(0.55f, 0.8f, 0.95f, 1), "%s", r.prompt_tags[q].c_str());
                }
                if (r.tags.empty() && r.prompt_tags.empty()) ImGui::TextDisabled("%s", r.tagged ? tr("(nothing clear)") : tr("(not tagged)"));
                if (r.stale && !r.user_edited) { ImGui::SameLine(); ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "*"); }
                ImGui::TableNextColumn();
                if (r.user_edited) ImGui::TextDisabled("%s", tr("mine"));
                else if (!r.tags.empty()) ImGui::Text("%.0f%%", r.best * 100);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.date_value.substr(0, 10).c_str());
                ImGui::TableNextColumn();
                auto flag = [](bool on, const char* label) {
                    ImGui::SameLine(0, 6);
                    if (on) ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.6f, 1), "%s", label);
                    else ImGui::TextDisabled("%s", label);
                };
                ImGui::Dummy(ImVec2(0, 0));
                flag(r.exif_date, tr("date"));
                flag(r.keywords, tr("keywords"));
                flag(r.description, tr("desc"));
                flag(r.place, tr("place"));
                if (!r.gen_tool.empty()) { ImGui::SameLine(0, 6); ImGui::TextColored(ImVec4(0.55f, 0.8f, 0.95f, 1), "%s", r.gen_tool.c_str()); }
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.location.c_str());
            }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::SameLine();
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
            ImGui::TextWrapped("%s", util::basename(p.src_path).c_str());
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
                else ImGui::Text("%-12s %s", label, v.c_str());
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
                    for (auto& k : w.removed) ImGui::TextColored(ImVec4(0.95f, 0.6f, 0.5f, 1), "- Xmp.dc.subject %s", k.c_str());
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
    if (g->pending.count(r.id) || g->pending.size() >= 4) return nullptr;
    g->pending.insert(r.id);
    ThumbKey key{r.src_path, r.size, 0};
    struct stat st{};
    if (stat(r.src_path.c_str(), &st) == 0) key.mtime = st.st_mtim.tv_sec;
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
        if (!d.t.rgba.empty()) g->tex[d.id].upload(d.t.rgba.data(), d.t.w, d.t.h);
    }
    if (g->tex.size() > 400) {  // drop what has not been on screen for a while
        for (auto it = g->tex.begin(); it != g->tex.end();) {
            if (g->frame - g->used[it->first] > 600) { it->second.unload(); it = g->tex.erase(it); }
            else ++it;
        }
    }
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
    const float cell = 168, img = 150;
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
    ImGuiListClipper clip;
    int nrows = (int(rows.size()) + cols - 1) / cols;
    clip.Begin(nrows, cell + ImGui::GetTextLineHeightWithSpacing());
    if (a.scroll_sel && sel >= 0) clip.IncludeItemByIndex(sel / cols);
    while (clip.Step())
        for (int row = clip.DisplayStart; row < clip.DisplayEnd; row++) {
            for (int c = 0; c < cols; c++) {
                int i = row * cols + c;
                if (i >= int(rows.size())) break;
                const PhotoRow& r = rows[size_t(i)];
                if (c) ImGui::SameLine();
                ImGui::BeginGroup();
                ImGui::PushID(int(r.id));
                ImVec2 p0 = ImGui::GetCursorScreenPos();
                if (ImGui::Selectable("##cell", a.selected == r.id, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(cell - 8, img))) {
                    a.selected = r.id;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, ids(), i);
                }
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem(r.favorite ? tr("Remove from favorites") : tr("Add to favorites"))) set_favorite(a, r.id, !r.favorite);
                    if (ImGui::MenuItem(tr("Find similar"))) {
                        snprintf(a.search, sizeof a.search, "like:#%lld", (long long)r.id);
                        a.rows_dirty = true;
                    }
                    if (ImGui::MenuItem(tr("Open folder"))) run_detached({"xdg-open", util::dirname(r.dest_path.empty() ? r.src_path : r.dest_path)});
                    ImGui::EndPopup();
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s\n%s", rel_path(a.cfg, r.src_path).c_str(), r.tags.c_str());
                if (a.scroll_sel && a.selected == r.id) { ImGui::SetScrollHereY(0.5f); a.scroll_sel = false; }
                if (const Texture* t = grid_thumb(a, r); t && t->tex) {
                    float s2 = std::min(img / t->w, img / t->h);
                    ImVec2 sz(t->w * s2, t->h * s2);
                    ImVec2 q0(p0.x + (cell - 8 - sz.x) / 2, p0.y + (img - sz.y) / 2);
                    ImGui::GetWindowDrawList()->AddImage(ImTextureRef((ImTextureID)(intptr_t)t->tex), q0, ImVec2(q0.x + sz.x, q0.y + sz.y));
                }
                if (r.favorite) ImGui::GetWindowDrawList()->AddText(ImVec2(p0.x + 4, p0.y + 2), IM_COL32(255, 205, 64, 255), "\xE2\x98\x85");
                std::string name = util::basename(r.dest_path.empty() ? r.src_path : r.dest_path);
                if (name.size() > 22) name = name.substr(0, 20) + "…";
                ImGui::TextDisabled("%s", name.c_str());
                ImGui::PopID();
                ImGui::EndGroup();
            }
        }
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
    if (ImGui::Button(tr("Check again"))) {
        RunOptions o = a.opts;
        o.stages = ST_FIX;
        o.scope = all_scope(a.cfg);  // every folder in the list, together
        o.recheck_all = true;
        a.last_stages = ST_FIX;
        a.pipe->start(effective(a.cfg), o);
    }
    tip(tr("Re-check every photo, not only new or changed ones"));
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
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Rejected proposals are not made again for the same photo."));
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
                ImGui::TableNextColumn();
                ImGui::ProgressBar(float(c.confidence), ImVec2(-1, 0), util::fmt("%.0f%%", c.confidence * 100).c_str());
                ImGui::TableNextColumn();
                std::string shown = util::basename(c.path);
                if (ImGui::Selectable(shown.c_str(), a.corr_focus == c.id, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap |
                                                                                  ImGuiSelectableFlags_AllowDoubleClick)) {
                    a.corr_focus = c.id;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) open_viewer(a, photo_ids, i);
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", c.path.c_str());
                ImGui::TableNextColumn();
                if (c.field == "name") ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1), "%s -> %s", c.current.c_str(), c.proposed.c_str());
                else if (c.action == "remove") ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.5f, 1), "- %s %s", tr("tag"), c.current.c_str());
                else ImGui::TextColored(ImVec4(0.55f, 0.9f, 0.6f, 1), "+ %s %s", tr("tag"), c.proposed.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(c.reason.c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", c.reason.c_str());
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
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tr("Scans new files (the Path row, or all sources), then compares sizes, quick hashes and, only where needed, full hashes."));
    ImGui::SameLine();
    ImGui::Checkbox(tr("Byte-verify"), &a.cfg.verify_dupes);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Compare files byte by byte before marking them identical."));
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
    ImGui::SameLine();
    if (ImGui::RadioButton(tr("Similar photos"), a.dup_show_similar)) { a.dup_show_similar = true; a.dups_dirty = true; }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", a.dup_show_similar ? tr("(report only: similar photos are never skipped automatically)")
                                                 : tr("(duplicates are not copied into the library; nothing is deleted)"));

    std::map<int64_t, const DupRow*> by_id;
    for (auto& r : a.dup_rows) by_id[r.id] = &r;
    float pane_w = 360;
    ImGui::BeginChild("duptable", ImVec2(ImGui::GetContentRegionAvail().x - pane_w - 8, 0));
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
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImVec4(1, 1, 1, 0.06f)));
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
                }
                ImGui::TableNextColumn();
                bool in_lib = !r.dest_path.empty();
                if (l.keeper) ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.6f, 1), "%s%s", l.similar ? tr("largest") : tr("kept"), in_lib ? " · lib" : "");
                else if (l.similar) ImGui::TextColored(ImVec4(0.7f, 0.75f, 1, 1), "%s %d", tr("distance"), r.similar_dist);
                else ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s", tr("duplicate"));
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
    ImGui::SameLine();
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
        if (ImGui::SmallButton(tr("Open folder"))) run_detached({"xdg-open", util::dirname(p.src_path)});
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
            ImVec4 c = l.level == 2 ? ImVec4(1, 0.45f, 0.4f, 1) : l.level == 1 ? ImVec4(0.95f, 0.8f, 0.4f, 1) : ImVec4(0.85f, 0.85f, 0.85f, 1);
            ImGui::TextColored(c, "%s  %s", l.time.c_str(), l.text.c_str());
        }
    static size_t last = 0;
    if (a.log.size() != last) {
        ImGui::SetScrollHereY(1.0f);
        last = a.log.size();
    }
    ImGui::EndChild();
}

static void draw_settings(App& a) {
    if (!a.settings_loaded) load_settings_buffers(a);
    Config& c = a.cfg;
    bool busy = a.pipe->running();
    ImGui::BeginChild("settings");
    ImGui::BeginDisabled(busy);

    if (ImGui::CollapsingHeader(tr("Organized copies"), ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox(tr("Preview before changes (recommended)"), &c.preview_first);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Analyze ends with a plan to review; nothing is copied or written until you press Apply."));
    ImGui::SetNextItemWidth(520);
    std::string hint = a.cfg.folder.empty() ? std::string(tr("default: <photo folder>/jev-organized")) : a.cfg.folder + "/jev-organized";
    ImGui::InputTextWithHint(tr("Organized copies folder"), hint.c_str(), a.lib_buf, sizeof a.lib_buf);
    ImGui::SameLine();
    if (ImGui::Button((std::string(tr("Browse...")) + "##out").c_str())) start_dir_dialog(a, 1);
    tip(tr("Choose a folder with the file dialog"));
    ImGui::TextDisabled("%s", tr("Leave empty to keep the organized copies next to your photos."));
    const char* layouts[] = {"2019/2019-05", "2019-05", "2019/05"};
    ImGui::SetNextItemWidth(200);
    ImGui::Combo(tr("Folder layout"), &c.layout, layouts, LAYOUT_COUNT);
    tip(tr("How month folders are named inside the organized folder"));
    const char* ops[] = {tr("Copy and rename (originals untouched)"), tr("Move and rename (no copies)"), tr("Rename in place (same folders)"),
                         tr("Add metadata only (no moving or renaming)")};
    ImGui::SetNextItemWidth(320);
    ImGui::Combo(tr("File operation"), &c.file_op, ops, OP_COUNT);
    tip(tr("What Organize does with each photo"));
    if (c.file_op == OP_COPY) ImGui::TextDisabled("%s", tr("Your photos stay as they are; renamed copies go into month folders."));
    else if (c.file_op == OP_MOVE) ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s", tr("Your photos are moved into the month folders and renamed; no second copy uses disk space."));
    else if (c.file_op == OP_RENAME) ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.4f, 1), "%s", tr("Your photos keep their folders and only get date names (20190512_00001.jpg)."));
    else ImGui::TextDisabled("%s", tr("Missing dates, tags, descriptions and places are added to the files where they are."));
    const char* ns[] = {tr("Date and serial (20190512_00001)"), tr("Original name, then date and serial (IMG_20190512_00001)")};
    ImGui::SetNextItemWidth(420);
    ImGui::Combo(tr("File names"), &c.name_style, ns, NAME_COUNT);
    tip(tr("How new file names are built"));
    ImGui::Checkbox(tr("Replace keywords jev-photos wrote earlier"), &c.rewrite_tags);
    tip(tr("When tags change, keywords jev-photos wrote before are updated; the file's own keywords stay"));
    ImGui::Checkbox(tr("Write a description (the AI prompt, plus \"Shows: <tags>\")"), &c.write_description);
    tip(tr("Write the AI prompt and 'Shows: <tags>' as the file's description (an existing one is only changed as you decide)"));
    ImGui::Checkbox(tr("Keywords from AI generation prompts"), &c.gen_keywords);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Tag-style prompts (\"1girl, hat, beach\") give their words directly; prose prompts are summarised by the LLM, once per distinct prompt."));
    const char* dp[] = {tr("Only fill empty descriptions (never change existing text)"), tr("Rules + LLM, jev cross-checks; disagreements go to you (recommended)"),
                        tr("Rules + LLM"), tr("Always ask me")};
    ImGui::SetNextItemWidth(460);
    ImGui::Combo(tr("Existing descriptions"), &c.desc_policy, dp, DESC_COUNT);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Placeholders (\"OLYMPUS DIGITAL CAMERA\", file names) are replaced by the rules; text that already covers ours is kept. For the rest, keep / append / replace is decided as chosen here. Replaced text is kept in Xmp.jev.PreviousDescription."));
    const char* wm[] = {tr("Embed into the organized copy"), tr("XMP sidecar"), tr("Database only")};
    ImGui::SetNextItemWidth(260);
    ImGui::Combo(tr("Metadata write"), &c.write_mode, wm, WRITE_COUNT);
    tip(tr("Where new metadata goes: into the file, into a .xmp file next to it, or only the catalog"));
    const char* seps[] = {"_", "-", " ", ""};
    int sep_idx = c.name_sep == "-" ? 1 : c.name_sep == " " ? 2 : c.name_sep.empty() ? 3 : 0;
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo(tr("Name separator"), &sep_idx, "_ (20190512_00001)\0- (20190512-00001)\0space\0none\0")) c.name_sep = seps[sep_idx];
    tip(tr("The character between the date and the serial number"));
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt(tr("Serial digits"), &c.sn_digits, 3, 8);
    tip(tr("How many digits the serial number has"));

    }

    if (ImGui::CollapsingHeader(tr("Tagging and search (CLIP)"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox((std::string(tr("Tag pictures with CLIP")) + "##clip").c_str(), &c.clip_enabled);
        tip(tr("Analyse each picture with CLIP for tags and meaning search"));
        ImGui::Checkbox(tr("Ask how much to analyse when pressing Analyze"), &c.analyze_ask);
        tip(tr("Analyse each picture with CLIP for tags and meaning search"));
        const char* cm[] = {tr("Auto (ViT-L/14 when downloaded, else ViT-B/32)"), "ViT-B/32 (fast)", "ViT-L/14 (much better, ~0.3 s per photo on the CPU)"};
        int ci = c.clip_model == "b32" ? 1 : c.clip_model == "l14" ? 2 : 0;
        ImGui::SetNextItemWidth(420);
        if (ImGui::Combo(tr("Model"), &ci, cm, 3)) c.clip_model = ci == 1 ? "b32" : ci == 2 ? "l14" : "auto";
        tip(tr("Which model to use"));
        const char* cd[] = {tr("Auto"), tr("CPU"), tr("GPU")};
        ImGui::SetNextItemWidth(160);
        ImGui::Combo(tr("Run on"), &c.clip_device, cd, 3);
        tip(tr("CPU or GPU (the GPU needs an ONNX Runtime build with CUDA)"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt(tr("CPU threads"), &c.clip_threads, 1, 16);
        tip(tr("More threads analyse faster but leave less of the machine for other work"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt(tr("Tags per photo"), &c.clip_max_tags, 1, 15);
        tip(tr("At most this many tags per photo"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt(tr("Photo thumbnail side"), &c.thumb_side, 256, 1024);
        tip(tr("Size of the cached thumbnails the analysis and the lists use"));
        if (ImGui::Button(tr("Edit tag list..."))) { load_tag_vocabulary(); run_detached({"xdg-open", tag_vocabulary_path()}); }
        ImGui::SameLine();
        ImGui::TextDisabled("%s", tag_vocabulary_path().c_str());
        std::string cdet;
        {
            std::lock_guard<std::mutex> l(a.health_mu);
            cdet = a.clip_detail;
        }
        ImGui::TextDisabled("%s", cdet.c_str());
    }

    if (ImGui::CollapsingHeader(tr("Ask search (AI)"))) {
        ImGui::TextDisabled("%s", tr("Ask turns a sentence into keywords, places and dates, searches locally, then lets a model pick what you meant."));
        ImGui::Checkbox((std::string(tr("Enabled")) + "##llm").c_str(), &c.llm_enabled);
        tip(tr("Use this service"));
        ImGui::SetNextItemWidth(420);
        ImGui::InputText("URL (OpenAI-compatible)##llm", a.llm_url, sizeof a.llm_url);
        ImGui::SetNextItemWidth(320);
        ImGui::InputText((std::string(tr("Model")) + "##llm").c_str(), a.llm_model, sizeof a.llm_model);
        ImGui::SetNextItemWidth(220);
        ImGui::InputText((std::string(tr("API key")) + "##llm").c_str(), a.llm_key, sizeof a.llm_key, ImGuiInputTextFlags_Password);
        ImGui::Checkbox(tr("Disable thinking (much faster for reasoning models)"), &c.llm_no_think);
        tip(tr("Much faster answers from reasoning models (Qwen3 style)"));
        const char* dec[] = {tr("LLM"), tr("jev only"), tr("None: local ranking only"), tr("LLM, jev settles its close calls (recommended)")};
        ImGui::SetNextItemWidth(340);
        ImGui::Combo(tr("Who picks the results"), &c.ask_decider, dec, JUDGE_COUNT);
        tip(tr("Who rates the candidates of an Ask search"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt(tr("Keep results rated at least (%)"), &c.ask_keep, 0, 100);
        tip(tr("Ask search drops candidates rated below this"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt(tr("Candidates per decision request"), &c.ask_batch, 5, 60);
        tip(tr("How many candidates go into one LLM request"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderInt(tr("Most candidates to check"), &c.ask_max_candidates, 20, 500);
        tip(tr("Ask search rates at most this many candidates"));
        ImGui::SetNextItemWidth(160);
        ImGui::SliderFloat(tr("Meaning search strictness"), &c.search_min_match, 0.1f, 0.8f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("How clearly a picture must show what you typed, scored like a tag against the other tags of its kind. Higher = fewer, surer results."));
        ImGui::SetNextItemWidth(120);
        ImGui::InputInt((std::string(tr("Timeout (s)")) + "##llm").c_str(), &c.llm_timeout);
        tip(tr("Give up on a request after this many seconds"));
    }

    if (ImGui::CollapsingHeader(tr("Dates and places (jev decision API)"))) {
    ImGui::Checkbox((std::string(tr("Enabled")) + "##jev").c_str(), &c.jev_enabled);
    tip(tr("Use this service"));
    ImGui::SetNextItemWidth(420);
    ImGui::InputText("URL##jev", a.jev_url, sizeof a.jev_url);
    ImGui::SetNextItemWidth(220);
    ImGui::InputText((std::string(tr("Model")) + "##jev").c_str(), a.jev_model, sizeof a.jev_model);
    ImGui::SetNextItemWidth(220);
    ImGui::InputText((std::string(tr("API key")) + "##jev").c_str(), a.jev_key, sizeof a.jev_key, ImGuiInputTextFlags_Password);
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt((std::string(tr("Timeout (s)")) + "##jev").c_str(), &c.jev_timeout);
    tip(tr("Give up on a request after this many seconds"));

    }

    if (ImGui::CollapsingHeader(tr("Vision-language model (advanced, off by default)"))) {
    ImGui::TextDisabled("%s", tr("Writes captions and reads text in screenshots. Needs a model server and a lot of memory."));
    ImGui::Checkbox((std::string(tr("Enabled")) + "##vl").c_str(), &c.vl_enabled);
    tip(tr("Use this service"));
    ImGui::SetNextItemWidth(420);
    ImGui::InputText("URL (OpenAI-compatible, e.g. http://127.0.0.1:11434/v1)##vl", a.vl_url, sizeof a.vl_url);
    ImGui::SetNextItemWidth(520);
    ImGui::InputText((std::string(tr("Model")) + "##vl").c_str(), a.vl_model, sizeof a.vl_model);
    ImGui::SetNextItemWidth(220);
    ImGui::InputText((std::string(tr("API key")) + "##vl").c_str(), a.vl_key, sizeof a.vl_key, ImGuiInputTextFlags_Password);
    ImGui::SetNextItemWidth(160);
    ImGui::InputText(tr("Tag language"), a.vl_lang, sizeof a.vl_lang);
    ImGui::SetNextItemWidth(160);
    ImGui::SliderInt(tr("Screenshot thumbnail side"), &c.vl_max_side, 384, 1024);
    tip(tr("Size of the images sent to the vision model for screenshots and text"));
    const char* devs[] = {tr("Auto (GPU when it fits)"), tr("CPU only")};
    ImGui::SetNextItemWidth(220);
    ImGui::Combo(tr("Run the model on (Ollama)"), &c.vl_device, devs, 2);
    tip(tr("Where Ollama runs the vision model"));
    ImGui::SetNextItemWidth(160);
    ImGui::InputInt(tr("Context (Ollama num_ctx)"), &c.vl_num_ctx, 1024);
    tip(tr("The vision model's context length"));
    c.vl_num_ctx = std::clamp(c.vl_num_ctx, 2048, 65536);
    ImGui::SetNextItemWidth(160);
    ImGui::SliderInt(tr("Concurrency"), &c.vl_concurrency, 1, 8);
    tip(tr("How many photos the vision model handles at once"));
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt((std::string(tr("Timeout (s)")) + "##vl").c_str(), &c.vl_timeout);
    tip(tr("Give up on a request after this many seconds"));
    ImGui::Checkbox(tr("JSON mode"), &c.vl_json_mode);
    tip(tr("Ask the vision model for strict JSON answers"));

    }

    if (ImGui::CollapsingHeader(tr("Decision rules (advanced)"))) {
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat(tr("Review below confidence"), &c.review_below, 0, 1, "%.2f");
    tip(tr("Dates less sure than this are marked uncertain"));
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat(tr("Ask jev when margin below"), &c.jev_margin, 0.5f, 1, "%.2f");
    tip(tr("jev is asked when the best two date answers are this close"));
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat(tr("Jev weight (dates)"), &c.jev_date_weight, 0, 1, "%.2f");
    tip(tr("How much jev's answer counts in a close date decision"));
    ImGui::TextUnformatted(tr("Location vote weights"));
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat(tr("Rules"), &c.loc_w_rules, 0, 1, "%.2f");
    tip(tr("Weight of the folder-name rules in the place vote"));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("VL", &c.loc_w_vl, 0, 1, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("jev", &c.loc_w_jev, 0, 1, "%.2f");
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat(tr("Accept location at"), &c.loc_accept, 0, 1, "%.2f");
    tip(tr("A place needs at least this score to be used"));
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt(tr("Minimum year"), &c.min_year);
    tip(tr("Dates before this year are treated as wrong"));
    ImGui::SetNextItemWidth(160);
    ImGui::SliderInt(tr("Scan threads"), &c.scan_threads, 1, 16);
    tip(tr("Files read in parallel while scanning"));
    }
    ImGui::EndDisabled();

    if (ImGui::CollapsingHeader(tr("Appearance"))) {
    int lang = get_lang();
    ImGui::SetNextItemWidth(160);
    if (ImGui::Combo(tr("Language"), &lang, kLangNames, L_COUNT)) {
        set_lang(Lang(lang));
        c.lang = kLangCodes[lang];
    }
    tip(tr("Interface language"));
    ImGui::SetNextItemWidth(160);
    ImGui::SliderFloat(tr("Font size"), &c.font_size, 12, 26, "%.0f");
    tip(tr("Interface text size"));
    if (ImGui::ColorEdit3(tr("Accent"), c.accent, ImGuiColorEditFlags_NoInputs)) apply_style(c);
    const char* rend[] = {tr("Auto (GPU, software if the driver crashes)"), tr("GPU only"), tr("Software")};
    ImGui::SetNextItemWidth(320);
    ImGui::Combo(tr("Renderer"), &c.renderer, rend, 3);
    tip(tr("Graphics: GPU, or software rendering when the GPU driver cannot open a window"));
    ImGui::SameLine();
    ImGui::TextDisabled("%s (%s)", a.gl_renderer.c_str(), tr("applies after restart"));
    }
    ImGui::Spacing();
    ImGui::BeginDisabled(busy);
    if (primary_button(tr("Save settings"), tr("Keep these settings (they are also used by the command line)"), ImVec2(160, 0))) {
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
        save_config(c, a.config_file);
        if (!a.db.is_open()) open_db(a);
        check_health(a);
        a.toast = tr("Saved");
        a.toast_until = glfwGetTime() + 2;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr("Test connections"), ImVec2(160, 0))) check_health(a);
    tip(tr("Check the jev, LLM and CLIP status now"));
    ImGui::EndChild();
}

// Analyze: first ask how much image analysis and tagging to do (it is the slow part), unless turned off.
static void request_analyze(App& a) {
    if (a.cfg.folders.empty()) { start_dir_dialog(a, 3); return; }
    if (!a.cfg.clip_enabled || !a.cfg.analyze_ask) {
        a.opts.clip_mode = a.opts.tag_mode = 0;
        start_run(a, analyze_stages(a.cfg));
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
        start_run(a, analyze_stages(c));
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    tip(tr("Close without analyzing (Esc)"));
    ImGui::EndPopup();
}

// One line that says what each kind of data is and where it is handled.
static void draw_concepts(App& a) {
    (void)a;
    struct C { const char* name; const char* tip; };
    const C cs[] = {
        {"File name", "The name on disk. Organize files gives it a date name (20190512_00001.jpg), optionally keeping the original name in front."},
        {"Tags", "Words for what the picture shows. CLIP (image analysis) proposes them with a confidence; your own tags replace them. Searchable."},
        {"Keywords", "Tags written into the file (XMP dc:subject), so other apps see them. Only added; the file's own keywords stay."},
        {"Description", "A sentence written into the file (XMP dc:description): the AI prompt for generated images, plus \"Shows: tags\". An existing description is only changed as you decide."},
        {"EXIF / XMP", "The metadata inside the file. Camera, lens, dates, GPS and AI prompts are the file's own and are never changed."},
        {"Corrections", "Fixes of generated tags and names, proposed with a confidence, applied only when you tick them."},
    };
    ImGui::TextDisabled("%s", tr("What is what:"));
    for (auto& c : cs) {
        ImGui::SameLine(0, 14);
        ImGui::TextColored(ImVec4(g_accent.x, g_accent.y, g_accent.z, 0.95f), "%s", tr(c.name));
        tip(tr(c.tip));
    }
}

static void draw_ui(App& a) {
    g_accent = ImVec4(a.cfg.accent[0], a.cfg.accent[1], a.cfg.accent[2], 1);
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("jev-photos", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
    draw_toolbar(a);
    g_prof.mark("toolbar");
    if (ImGui::BeginTabBar("tabs")) {
        auto sel = [&](const char* name) {
            bool on = a.start_tab == name;
            if (on) a.start_tab.clear();
            return on ? ImGuiTabItemFlags_SetSelected : 0;
        };
        a.on_overview = false;
        ImGuiTabItemFlags of = a.show_overview || a.start_tab == "overview" ? ImGuiTabItemFlags_SetSelected : 0;
        if (a.start_tab == "overview") a.start_tab.clear();
        a.show_overview = false;
        if (ImGui::BeginTabItem(tr("Overview"), nullptr, of)) { draw_overview(a); g_prof.mark("tab:Overview"); ImGui::EndTabItem(); }
        tip(tr("What was found and what to do next"));
        if (a.start_tab == "favorites") { a.fav_only = true; a.rows_dirty = true; a.start_tab = "photos"; }
        if (ImGui::BeginTabItem(tr("Photos"), nullptr, sel("photos"))) { draw_photos(a); g_prof.mark("tab:Photos"); ImGui::EndTabItem(); }
        tip(tr("Browse and search your photos; star favorites"));
        // Actions: everything that changes files or the catalog, one sub-tab per kind of change.
        static const std::set<std::string> action_tabs = {"preview", "corrections", "dupes", "tags"};
        ImGuiTabItemFlags pf = a.show_plan_tab || action_tabs.count(a.start_tab) ? ImGuiTabItemFlags_SetSelected : 0;
        // The sub-tab to open is kept until the Actions tab is really drawn (the parent switches a frame later).
        if (a.show_plan_tab) a.action_sub = "preview";
        if (action_tabs.count(a.start_tab)) { a.action_sub = a.start_tab; a.start_tab.clear(); }
        a.show_plan_tab = false;
        int pending = count_plan(a.plan, false);  // only what is still to apply: applied items no longer count
        std::string plan_label = std::string(tr("Actions")) + (pending ? util::fmt(" (%d)", pending) : "") + "###plan";
        bool was_tags = a.on_tags;
        a.on_tags = false;
        if (ImGui::BeginTabItem(plan_label.c_str(), nullptr, pf)) {
            draw_concepts(a);
            if (ImGui::BeginTabBar("actiontabs")) {
                std::string sub = a.action_sub;
                a.action_sub.clear();
                auto subsel = [&](const char* k) { return sub == k ? ImGuiTabItemFlags_SetSelected : 0; };
                std::string l1 = std::string(tr("Organize files")) + (pending ? util::fmt(" (%d)", pending) : "") + "###act_files";
                if (ImGui::BeginTabItem(l1.c_str(), nullptr, subsel("preview"))) { draw_plan(a); ImGui::EndTabItem(); }
                tip(tr("Copy, move or rename files, and write dates, keywords and descriptions into them. Everything is listed before it happens."));
                if (ImGui::BeginTabItem(tr("Tags & metadata"), nullptr, subsel("tags"))) { draw_tags(a); ImGui::EndTabItem(); }
                tip(tr("Each photo's tags (what CLIP sees, or your own), and what its file already carries: date, keywords, description, place."));
                std::string cl = std::string(tr("Corrections")) + (a.corrections.empty() ? "" : util::fmt(" (%zu)", a.corrections.size())) + "###corr";
                if (ImGui::BeginTabItem(cl.c_str(), nullptr, subsel("corrections"))) { draw_corrections(a); ImGui::EndTabItem(); }
                tip(tr("Proposed fixes of generated tags and names, each with how sure jev-photos is. Nothing changes until you apply them."));
                int nd = 0;
                for (auto& r : a.dup_rows) nd += r.dup_of != 0;
                std::string dl = std::string(tr("Duplicates")) + (nd ? util::fmt(" (%d)", nd) : "") + "###dupes";
                if (ImGui::BeginTabItem(dl.c_str(), nullptr, subsel("dupes"))) { draw_dupes(a); ImGui::EndTabItem(); }
                tip(tr("Identical files (and optionally similar photos): which copy is kept, and moving the extra copies to the Trash."));
                ImGui::EndTabBar();
            }
            g_prof.mark("tab:Actions");
            ImGui::EndTabItem();
        }
        tip(tr("Everything that changes files: organizing, tags and metadata, corrections and duplicates"));
        if (a.on_tags && !was_tags) a.rows_dirty = true;  // entering Tags & metadata: fetch its rows
        if (a.start_tab == "log") { a.show_log = true; a.start_tab.clear(); }
        if (ImGui::BeginTabItem(tr("Settings"), nullptr, sel("settings"))) { draw_settings(a); g_prof.mark("tab:Settings"); ImGui::EndTabItem(); }
        tip(tr("Models, servers, naming and decision rules"));
        ImGui::EndTabBar();
    }
    draw_trash_modal(a);
    draw_analyze_dialog(a);
    ImGui::End();
    draw_viewer(a);
    ImGui::Begin("jev-photos");
    if (glfwGetTime() < a.toast_until) {
        ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 20, vp->WorkPos.y + vp->WorkSize.y - 20), 0, ImVec2(1, 1));
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(a.toast.c_str());
        ImGui::EndTooltip();
    }
    ImGui::End();
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
    std::string path = fs::weakly_canonical(file).string();
    std::string root = root_in.empty() ? util::dirname(path) : fs::weakly_canonical(root_in).string();
    MetaMap m = read_meta(path);
    struct stat st{};
    stat(path.c_str(), &st);
    DateInputs in;
    in.filename = util::basename(path);
    in.dirs = rel_dirs(root, path);
    in.meta = &m;
    in.mtime = st.st_mtim.tv_sec;
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
    a.opts.scope = a.opts.scope.empty() ? all_scope(a.cfg) : fs::weakly_canonical(a.opts.scope).string();
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
    signal(SIGPIPE, SIG_IGN);
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
        else if (!a.empty() && a[0] != '-') srcs.push_back(a);  // positional: the photo folder
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    if (!srcs.empty()) {
        // Folders on the command line: the CLI works on exactly those (together); the GUI adds them to its list.
        if (cli) app.cfg.folders.clear();
        for (auto& f : srcs) add_folder(app.cfg, fs::weakly_canonical(f).string());
        app.cfg.folder = fs::weakly_canonical(srcs[0]).string();
        app.cfg.view_all = cli && srcs.size() > 1;
    }
    if (!lib.empty()) app.cfg.output = fs::weakly_canonical(lib).string();
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

    // ---------------- GUI
    if (!getenv("JEV_PHOTOS_CHILD") && !getenv("JEV_PHOTOS_NO_SUPERVISOR")) return supervise(argc, argv, app.cfg.renderer);
    if (getenv("JEV_PHOTOS_CHILD")) prctl(PR_SET_PDEATHSIG, SIGTERM);  // the window never outlives its supervisor
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
    io.IniFilename = nullptr;
    apply_style(app.cfg);
    build_fonts();
    ImGui_ImplGlfw_InitForOpenGL(app.win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    if (!app.cfg.folder.empty() && util::dir_exists(app.cfg.folder)) remember_folder(app.cfg, app.cfg.folder);
    snprintf(app.folder_buf, sizeof app.folder_buf, "%s", app.cfg.folder.c_str());
    open_db(app);
    check_health(app);
    if (!startup_preview.empty()) {
        std::string p = fs::weakly_canonical(startup_preview).string();
        set_folder(app, p);
        start_run(app, ST_SCAN | ST_DUPES | ST_DECIDE | ST_ORGANIZE, true);
    }
    app.log.add(0, exiv2_available() ? "exiv2 found" : "exiv2 NOT found: install it to read/write metadata");
    if (view_id) open_viewer(app, {view_id}, 0);
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
            vr.folder = view_scope(app.cfg);
            vr.review_only = app.review_only;
            vr.dup_similar = app.dup_show_similar;
            vr.sq.text = util::trim(app.search);
            vr.sq.mode = app.search_mode;
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
            app.toast = util::fmt("%d %s", app.trash_done.load(), tr("extra copies moved"));
            app.toast_until = glfwGetTime() + 3;
        }
        was_trashing = app.trash_busy;
        if (glfwGetTime() - app.health_at > 20) {
            app.health_at = glfwGetTime();
            check_health(app);
        }
        static int seen_runs = 0;
        if (app.pipe->progress.finished_runs != seen_runs && !app.pipe->running()) {
            seen_runs = app.pipe->progress.finished_runs;
            app.rows_dirty = true;
            app.sel_loaded = false;
            app.dups_dirty = true;
            app.catalog_gen++;
            app.plan = app.pipe->plan();
            if (app.last_run_planned && !app.plan.empty()) app.show_overview = true;  // show what was found and what to do next
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
                            {"j", ImGuiKey_J}, {"k", ImGuiKey_K}, {"h", ImGuiKey_H}, {"l", ImGuiKey_L}, {"q", ImGuiKey_Q}, {"f", ImGuiKey_F},
                            {"Up", ImGuiKey_UpArrow}, {"Down", ImGuiKey_DownArrow}, {"Left", ImGuiKey_LeftArrow},
                            {"Right", ImGuiKey_RightArrow}, {"Return", ImGuiKey_Enter}, {"Escape", ImGuiKey_Escape}, {"space", ImGuiKey_Space}};
                        auto k = keys.find(t);
                        if (k != keys.end()) { sio.AddKeyEvent(k->second, true); release = k->second; }
                        else if (util::starts_with(t, "tab:")) app.start_tab = t.substr(4);
                        else if (t == "trash-dupes") trash_duplicates(app);  // same as confirming the dialog
                        else if (t == "trash-dialog") app.trash_confirm = true;
                        else if (util::starts_with(t, "ask:")) {  // Ask search for the given sentence
                            snprintf(app.search, sizeof app.search, "%s", t.substr(4).c_str());
                            app.search_mode = SM_ASK;
                            app.rows_dirty = true;
                        }
                        else if (t == "plan") start_run(app, ST_ORGANIZE, true);
                        else if (t == "analyze") request_analyze(app);
                        else if (t == "apply") apply_plan(app);
                    }
                } else if (script_shot.empty()) {
                    glfwSetWindowShouldClose(app.win, GLFW_TRUE);
                }
            }
        }
        ImGui::NewFrame();
        draw_ui(app);
        ImGui::Render();
        g_prof.mark("imgui");
        int fw, fh;
        glfwGetFramebufferSize(app.win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.07f, 0.08f, 0.10f, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        g_prof.mark("gl");
        if (!script_shot.empty()) {
            save_screenshot(script_shot, fw, fh);
            script_shot.clear();
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
