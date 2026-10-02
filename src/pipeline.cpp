#include "pipeline.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <map>
#include <regex>
#include <set>
#include <unordered_map>

#include "dates.h"
#include "dupes.h"
#include "genmeta.h"
#include "llm.h"
#include "jev.h"
#include "location.h"
#include "meta.h"
#include "nlohmann/json.hpp"
#include "stb_image.h"
#include "util.h"
#include "clip.h"
#include "thumbs.h"
#include "vision.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

const char* stage_name(int st) {
    switch (st) {
        case ST_SCAN: return "scan";
        case ST_DECIDE: return "decide";
        case ST_VISION: return "vision";
        case ST_ORGANIZE: return "organize";
        case ST_DUPES: return "duplicates";
        case ST_TAG: return "tag";
        case ST_FIX: return "check";
        default: return "idle";
    }
}

// ---------------------------------------------------------------------------
// Log

void Log::add(int level, const std::string& text) {
    std::lock_guard<std::mutex> l(mu_);
    LogLine ln{level, util::now_iso().substr(11), text};
    if (echo) fprintf(stderr, "%s %s%s\n", ln.time.c_str(), level == 2 ? "ERROR " : level == 1 ? "warn " : "", text.c_str());
    lines_.push_back(std::move(ln));
    while (lines_.size() > 5000) lines_.pop_front();
}

std::vector<LogLine> Log::tail(size_t n) {
    std::lock_guard<std::mutex> l(mu_);
    size_t start = lines_.size() > n ? lines_.size() - n : 0;
    return std::vector<LogLine>(lines_.begin() + long(start), lines_.end());
}

size_t Log::size() {
    std::lock_guard<std::mutex> l(mu_);
    return lines_.size();
}

// ---------------------------------------------------------------------------
// helpers

namespace {

const std::set<std::string>& image_exts() {
    static const std::set<std::string> s = {"jpg", "jpeg", "jpe", "png", "gif", "bmp", "webp", "tif", "tiff", "heic", "heif", "avif",
                                            "dng", "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "orf", "rw2", "raf",
                                            "pef", "srw", "x3f", "3fr", "erf", "kdc", "mrw"};
    return s;
}

int64_t birth_time(const std::string& path) {
#ifdef _WIN32
    struct stat st{};  // Windows keeps the creation time in st_ctime
    return stat(path.c_str(), &st) == 0 ? (int64_t)st.st_ctime : 0;
#else
    struct statx sx{};
    if (statx(AT_FDCWD, path.c_str(), 0, STATX_BTIME, &sx) == 0 && (sx.stx_mask & STATX_BTIME)) return sx.stx_btime.tv_sec;
    return 0;
#endif
}

// File-name words (rules; long names are refined by the LLM later). Confidence 0.5 = rules, 1 = decided.
// The original-name prefix, or for a long name its key words joined with '-'.
std::string short_prefix(const Photo& p) {
    std::string prefix = name_prefix(util::stem(p.src_path));
    if (!long_name(prefix)) return prefix;
    std::string out;
    for (auto& [w, conf] : parse_tag_list(p.name_keywords)) {
        if (out.size() > 28) break;
        out += (out.empty() ? "" : "-") + util::replace_all(w, " ", "-");
    }
    return out.empty() ? prefix.substr(0, 30) : out;
}

void set_name_words(Photo& p) {
    json w = json::array();
    for (auto& x : name_words(util::stem(p.src_path))) w.push_back({x, 0.5});
    p.name_keywords = w.dump();
}

// Generation metadata into the photo row; tag-style prompts give their keywords right away.
void set_gen(Photo& p, const GenInfo& g) {
    p.gen_json = g.found() ? g.to_json() : "";
    p.gen_tool = g.tool;
    json kw = json::array();
    for (auto& k : prompt_keywords(g.prompt)) kw.push_back(k);
    p.gen_keywords = kw.empty() ? "" : kw.dump();
}

std::string norm_ext(const std::string& e) { return e == "jpeg" || e == "jpe" ? "jpg" : e == "tif" ? "tiff" : e; }

bool image_dims(const std::string& path, const MetaMap& m, int& w, int& h) {
    int n;
    if (stbi_info(path.c_str(), &w, &h, &n)) return true;
    w = atoi(m.get("Exif.Photo.PixelXDimension").c_str());
    h = atoi(m.get("Exif.Photo.PixelYDimension").c_str());
    return w > 0 && h > 0;
}

std::vector<std::string> json_list(const std::string& s) {
    std::vector<std::string> out;
    json a = json::parse(s.empty() ? "[]" : s, nullptr, false);
    if (a.is_array())
        for (auto& x : a)
            if (x.is_string()) out.push_back(x.get<std::string>());
    return out;
}

std::string meta_location(const MetaMap& m) {
    for (const char* k : {"Xmp.iptc.Location", "Iptc.Application2.SubLocation", "Xmp.photoshop.City", "Iptc.Application2.City"})
        if (m.has(k)) return util::trim(m.get(k));
    return "";
}

}  // namespace

std::vector<std::string> rel_dirs(const std::string& src_root, const std::string& file) {
    std::vector<std::string> out;
    fs::path root = fs::u8path(src_root).lexically_normal();
    fs::path p = fs::u8path(file).parent_path().lexically_normal();
    while (!p.empty()) {
        out.push_back(p.filename().u8string());
        if (p == root || p == p.root_path()) break;
        p = p.parent_path();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pipeline

bool Pipeline::start(const Config& cfg, const RunOptions& opt) {
    if (progress.running) return false;
    join();
    stop_ = false;
    progress.running = true;
    th_ = std::thread([this, cfg, opt] { run(cfg, opt); });
    return true;
}

void Pipeline::run_blocking(const Config& cfg, const RunOptions& opt) {
    stop_ = false;
    progress.running = true;
    run(cfg, opt);
}

void Pipeline::begin(int stage, int total) {
    progress.stage = stage;
    progress.done = 0;
    progress.total = total;
    progress.errors = 0;
    progress.skipped = 0;
    progress.stage_start = std::chrono::steady_clock::now();
    progress.set_current("");
}

template <class F>
void Pipeline::parallel(int n, int threads, F fn) {
    std::atomic<int> next{0};
    auto worker = [&] {
        for (int i; !stop_ && (i = next++) < n;) {
            fn(i);
            progress.done++;
        }
    };
    threads = std::max(1, std::min(threads, n));
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; t++) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

void Pipeline::run(Config cfg, RunOptions opt) {
    jev_down_ = false;
    if (!exiv2_available()) log_.add(1, "exiv2 not found: metadata cannot be read or written (install exiv2)");
    if (!db_.is_open()) {
        log_.add(2, "the catalog database could not be opened");
    } else {
        if (!opt.scope.empty()) {
            std::string canon;
            for (auto& sc : util::split(opt.scope, '\n')) {  // one folder per line
                std::error_code ec;
                if (!util::trim(sc).empty()) canon += (canon.empty() ? "" : "\n") + util::canonical(util::trim(sc));
            }
            opt.scope = canon;
            log_.add(0, "scope: " + util::replace_all(opt.scope, "\n", ", "));
        }
        if (!opt.undo_run.empty()) {
            undo(cfg, opt.undo_run);
            opt.stages = 0;
        }
        if ((opt.stages & ST_SCAN) && !stop_) scan(cfg, opt);
        if ((opt.stages & ST_DUPES) && !stop_) dedupe(cfg, opt);
        if ((opt.stages & ST_DECIDE) && !stop_) decide(cfg, opt);
        if ((opt.stages & ST_TAG) && !stop_) {
            if (cfg.clip_enabled) tag(cfg, opt);
            else log_.add(0, "tag stage skipped (CLIP disabled in settings)");
            if (cfg.gen_keywords && cfg.llm_enabled && !stop_) prompt_keywords_llm(cfg, opt);
        }
        if ((opt.stages & ST_VISION) && !stop_) {
            if (cfg.vl_enabled) vision(cfg, opt);
            else log_.add(0, "vision stage skipped (VL model disabled in settings)");
        }
        if ((opt.stages & ST_FIX) && !stop_) check_corrections(cfg, opt);
        if ((opt.stages & ST_ORGANIZE) && !stop_) organize(cfg, opt);
    }
    if (stop_) log_.add(1, "stopped");
    DbStats s = db_.is_open() ? db_.stats(opt.scope) : DbStats{};
    std::lock_guard<std::mutex> l(progress.mu);
    progress.summary = util::fmt("%d photos (%s)  ·  %d dated, %d uncertain, %d undated  ·  %d tagged  ·  %d with a place  ·  %d organized  ·  %d duplicates (%s)",
                                 s.total, util::human_size(s.total_bytes).c_str(), s.resolved - s.undated - s.review, s.review, s.undated,
                                 s.tagged + (s.tagged ? 0 : s.vision_done), s.located, s.organized, s.dups, util::human_size(s.dup_bytes).c_str());
    log_.add(0, "done: " + progress.summary);
    progress.stage = 0;
    progress.finished_runs++;
    progress.running = false;
}

// ---- stage 1: scan

void Pipeline::scan(const Config& c, const RunOptions& o) {
    begin(ST_SCAN, 0);
    std::vector<std::pair<std::string, std::string>> files;  // (root, path)
    // Organized-copy folders are never scanned: the shared output folder, and each folder's own jev-organized.
    std::set<std::string> libs;
    if (!c.library.empty()) libs.insert(util::canonical(c.library));
    for (auto& s : c.sources) {
        std::error_code ec;
        libs.insert(library_for(c, util::canonical(s)));
    }
    std::string lib = c.library.empty() ? "" : util::canonical(c.library);
    // (root used for folder-name evidence, folder actually walked)
    std::vector<std::pair<std::string, std::string>> walks;
    if (!o.scope.empty()) {
        // Each chosen path keeps the configured source it lives in as its root, so folder names above it still count.
        for (auto& sc0 : util::split(o.scope, '\n')) {
            std::string sc = util::trim(sc0);
            if (sc.empty()) continue;
            std::string root = sc;
            for (auto& s : c.sources) {
                std::error_code ec;
                std::string cs = util::canonical(s);
                if (path_under(sc, cs) && cs.size() < root.size()) root = cs;
            }
            walks.push_back({root, sc});
        }
    } else {
        for (auto& s : c.sources) {
            std::error_code ec;
            std::string cs = util::canonical(s);
            walks.push_back({cs, cs});
        }
    }
    if (!lib.empty() && !o.scope.empty() && path_under(o.scope, lib)) {
        log_.add(2, "the chosen path is inside the library folder");
        return;
    }
    for (auto& [root, walk] : walks) {
        std::error_code ec;
        if (!util::dir_exists(walk)) {
            log_.add(2, "source folder not found: " + walk);
            continue;
        }
        progress.set_current("listing " + walk);
        auto it = fs::recursive_directory_iterator(fs::u8path(walk), fs::directory_options::skip_permission_denied, ec);
        for (; !ec && it != fs::recursive_directory_iterator() && !stop_; it.increment(ec)) {
            const std::string ps = util::norm_path(it->path().u8string());  // UTF-8, forward slashes on every system
            std::string name = it->path().filename().u8string();
            if (it->is_directory(ec)) {
                // Skip hidden folders, the library itself and macOS/Synology thumbnail stores.
                if ((!name.empty() && name[0] == '.') || name == "@eaDir" || libs.count(ps))
                    it.disable_recursion_pending();
                continue;
            }
            if (!name.empty() && name[0] == '.') continue;
            if (it->is_regular_file(ec) && image_exts().count(util::ext_lower(name))) files.push_back({root, ps});
        }
    }
    progress.total = int(files.size());
    int64_t bytes = 0;
    for (auto& f : files) {
        struct stat st{};
        if (stat(f.second.c_str(), &st) == 0) bytes += st.st_size;
    }
    log_.add(0, util::fmt("scan: %zu image files, %s", files.size(), util::human_size(bytes).c_str()));
    parallel(int(files.size()), c.scan_threads, [&](int i) {
        const auto& [root, path] = files[size_t(i)];
        struct stat st{};
        if (stat(path.c_str(), &st) != 0) { progress.errors++; return; }
        if (db_.known_unchanged(path, st.st_size, ST_MTIME(st))) { progress.skipped++; return; }
        progress.set_current(path);
        Photo p;
        p.src_path = path;
        p.src_root = root;
        p.size = st.st_size;
        p.mtime = ST_MTIME(st);
        p.btime = birth_time(path);
        p.ext = util::ext_lower(path);
        p.quick_hash = util::quick_hash(path, p.size);  // full hashes are computed later, only for size+quick collisions
        MetaMap m = read_meta(path);
        image_dims(path, m, p.width, p.height);
        // Thumbnail once, here, on the CPU: vision, the similar-photo check and the UI all reuse it.
        std::string terr;
        if (ensure_thumb({path, p.size, p.mtime}, meta_orientation(m), thumb_side_for(path, c.thumb_side, c.vl_max_side), terr).empty())
            log_.add(1, "no thumbnail (" + terr + "): " + path);
        p.meta_json = m.to_json();
        p.meta_error = m.error;
        set_gen(p, read_gen_info(path, m));
        set_name_words(p);
        if (!m.error.empty()) log_.add(1, "metadata unreadable (" + m.error + "): " + path);
        db_.upsert_scan(p);
    });
    log_.add(0, util::fmt("scan: %d new/changed, %d unchanged, %d errors", progress.done - progress.skipped - progress.errors,
                          progress.skipped.load(), progress.errors.load()));
    // Photos scanned before generation metadata was read: look once (PNG text chunks are cheap to find).
    std::vector<int64_t> unchecked = scoped("COALESCE(gen_checked,0)=0", o);
    int found = 0;
    for (int64_t id : unchecked) {
        if (stop_) break;
        Photo p;
        if (!db_.load(id, p)) continue;
        set_gen(p, read_gen_info(p.src_path, MetaMap::from_json(p.meta_json)));
        db_.save_gen(id, p.gen_json, p.gen_tool, p.gen_keywords);
        found += !p.gen_tool.empty();
    }
    if (found) log_.add(0, util::fmt("scan: AI generation metadata (prompt, model, seed) in %d earlier photos", found));
    for (int64_t id : scoped("name_keywords IS NULL", o)) {
        Photo p;
        if (!db_.load(id, p)) continue;
        set_name_words(p);
        db_.save_name_keywords(id, p.name_keywords);
    }
}

// ---- stage 1b: duplicates (size -> quick hash -> full hash -> byte compare), then optional near duplicates

void Pipeline::dedupe(const Config& c, const RunOptions& o) {
    begin(ST_DUPES, 0);
    progress.set_current("grouping by size");
    // Duplicates are looked for inside the chosen folder (the whole catalog when no folder is given).
    std::vector<DupRow> rows;
    for (auto& r : db_.dup_rows(true))
        if (o.scope.empty() || path_under(r.src_path, o.scope)) rows.push_back(std::move(r));
    std::vector<int64_t> scope_ids;
    for (auto& r : rows) scope_ids.push_back(r.id);
    std::map<int64_t, std::vector<size_t>> by_size;
    for (size_t i = 0; i < rows.size(); i++)
        if (rows[i].size > 0) by_size[rows[i].size].push_back(i);

    // Tier 2: quick hash (only for same-size files; old libraries may lack it).
    std::vector<size_t> need_quick, need_full;
    for (auto& [sz, v] : by_size)
        if (v.size() > 1)
            for (size_t i : v)
                if (rows[i].quick_hash.empty()) need_quick.push_back(i);
    for (size_t i : need_quick) rows[i].quick_hash = util::quick_hash(rows[i].src_path, rows[i].size);
    std::map<std::pair<int64_t, std::string>, std::vector<size_t>> by_quick;
    int64_t same_size = 0;
    for (auto& [sz, v] : by_size)
        if (v.size() > 1)
            for (size_t i : v) {
                same_size++;
                if (!rows[i].quick_hash.empty()) by_quick[{sz, rows[i].quick_hash}].push_back(i);
            }
    // Tier 3: full XXH3-128, only where size and quick hash still collide.
    for (auto& [k, v] : by_quick)
        if (v.size() > 1)
            for (size_t i : v)
                if (rows[i].content_hash.empty()) need_full.push_back(i);
    int64_t full_bytes = 0;
    for (size_t i : need_full) full_bytes += rows[i].size;
    progress.total = int(need_full.size());
    log_.add(0, util::fmt("duplicates: %zu files, %lld share a size, %zu need a full hash (%s to read)", rows.size(),
                          (long long)same_size, need_full.size(), util::human_size(full_bytes).c_str()));
    parallel(int(need_full.size()), c.scan_threads, [&](int k) {
        DupRow& r = rows[need_full[size_t(k)]];
        progress.set_current(r.src_path);
        r.content_hash = util::content_hash(r.src_path);
        if (r.content_hash.empty()) { progress.errors++; return; }
        db_.set_content_hash(r.id, r.content_hash);
    });
    if (stop_) return;

    std::map<std::string, std::vector<size_t>> by_full;
    for (auto& [k, v] : by_quick)
        if (v.size() > 1)
            for (size_t i : v)
                if (!rows[i].content_hash.empty()) by_full[util::fmt("%lld:", (long long)rows[i].size) + rows[i].content_hash].push_back(i);
    std::vector<std::pair<int64_t, int64_t>> links;
    int groups = 0;
    int64_t reclaim = 0;
    for (auto& [h, v] : by_full) {
        if (v.size() < 2) continue;
        std::vector<const DupRow*> g;
        for (size_t i : v) g.push_back(&rows[i]);
        int64_t keep = choose_keeper(g);
        const DupRow* kr = *std::find_if(g.begin(), g.end(), [&](const DupRow* r) { return r->id == keep; });
        int members = 0;
        for (auto* r : g) {
            if (r->id == keep) continue;
            // Hash equality is overwhelming evidence; a byte comparison makes it proof (when both files are here).
            if (c.verify_dupes && util::file_exists(r->src_path) && util::file_exists(kr->src_path) && !util::files_equal(r->src_path, kr->src_path)) {
                log_.add(1, "hash collision, files differ: " + r->src_path + " vs " + kr->src_path);
                continue;
            }
            links.push_back({r->id, keep});
            reclaim += r->size;
            members++;
        }
        groups += members > 0;
    }
    db_.apply_dups(scope_ids, links);
    log_.add(0, util::fmt("duplicates: %d groups, %zu extra copies, %s reclaimable%s", groups, links.size(),
                          util::human_size(reclaim).c_str(), c.verify_dupes ? " (byte-verified)" : ""));

    if (!c.similar_check || stop_) return;
    // Near duplicates: dHash every unique image once (cached), then cluster.
    std::set<int64_t> dup_ids;
    for (auto& [id, of] : links) dup_ids.insert(id);
    std::vector<size_t> need_dhash;
    for (size_t i = 0; i < rows.size(); i++)
        if (!dup_ids.count(rows[i].id) && !rows[i].dhash_done) need_dhash.push_back(i);
    begin(ST_DUPES, int(need_dhash.size()));
    log_.add(0, util::fmt("similar: computing perceptual hashes for %zu images", need_dhash.size()));
    parallel(int(need_dhash.size()), c.scan_threads, [&](int k) {
        DupRow& r = rows[need_dhash[size_t(k)]];
        progress.set_current(r.src_path);
        uint64_t h = 0;
        // The thumbnail is already upright and small: decoding it is ~50x cheaper than the original.
        std::string terr, tp = ensure_thumb({r.src_path, r.size, r.mtime}, meta_orientation(MetaMap::from_json(r.meta_json)),
                                            thumb_side_for(r.src_path, c.thumb_side, c.vl_max_side), terr);
        bool ok = !tp.empty() ? image_dhash(tp, 1, h) : image_dhash(r.src_path, meta_orientation(MetaMap::from_json(r.meta_json)), h);
        r.dhash = h;
        r.dhash_done = true;
        if (!ok) { r.dhash = 0; progress.skipped++; }
        db_.set_dhash(r.id, h, ok);
    });
    if (stop_) return;
    std::vector<const DupRow*> cands;
    for (auto& r : rows)
        if (!dup_ids.count(r.id) && r.dhash_done && r.dhash) cands.push_back(&r);
    std::vector<SimilarLink> sl = similar_links(cands, c.similar_threshold);
    std::vector<std::tuple<int64_t, int64_t, int>> sim;
    std::set<int64_t> reps;
    for (auto& l : sl) {
        sim.push_back({l.id, l.to, l.dist});
        reps.insert(l.to);
    }
    db_.apply_similar(scope_ids, sim);
    log_.add(0, util::fmt("similar: %zu groups, %zu near-duplicate photos (dHash distance <= %d); review them in the Duplicates tab",
                          reps.size(), sim.size(), c.similar_threshold));
}

// ---- stage 2: decide date + location

void Pipeline::decide(const Config& c, const RunOptions& o) {
    std::vector<int64_t> all = db_.ids("1");
    std::vector<int64_t> todo = scoped(o.redecide_all ? "1" : "resolved=0", o);
    begin(ST_DECIDE, int(todo.size()));
    if (todo.empty()) return;
    if (o.redecide_all && o.scope.empty()) db_.clear_dirs();
    int64_t now = util::now_epoch();

    // Pass 1: detect "bulk copy" mtimes. Files in one folder that share an mtime minute while their own
    // (embedded / name) evidence spans several days were clearly copied at that minute.
    progress.set_current("checking file times");
    struct Key {
        std::string dir;
        int64_t minute;
        bool operator<(const Key& k) const { return dir != k.dir ? dir < k.dir : minute < k.minute; }
    };
    std::map<Key, std::pair<int, std::set<int64_t>>> groups;  // -> (count, evidence days)
    std::unordered_map<int64_t, Key> key_of;
    for (int64_t id : all) {
        Photo p;
        if (!db_.load(id, p)) continue;
        Key k{util::dirname(p.src_path), p.mtime / 60};
        key_of[id] = k;
        auto& g = groups[k];
        g.first++;
        MetaMap m = MetaMap::from_json(p.meta_json);
        DateInputs in;
        in.filename = util::basename(p.src_path);
        in.meta = &m;
        in.now = now;
        in.min_year = c.min_year;
        for (auto& cand : collect_date_candidates(in))
            if (!cand.rejected() && cand.when.prec >= util::P_DAY && cand.src != DS_FS_MTIME && cand.src != DS_FS_BIRTH)
                g.second.insert(util::days_from_civil(cand.when.y, cand.when.mo, cand.when.d));
    }
    auto bulk = [&](int64_t id) {
        auto it = key_of.find(id);
        if (it == key_of.end()) return false;
        auto& g = groups[it->second];
        return (g.first >= 5 && g.second.size() >= 2) || g.first >= 20;
    };

    // Date decision for one photo; `extra` carries evidence found after the first pass (neighbours).
    auto decide_date_for = [&](Photo& p, const std::vector<DateCandidate>& extra) {
        MetaMap m = MetaMap::from_json(p.meta_json);
        DateInputs in;
        in.filename = util::basename(p.src_path);
        in.dirs = rel_dirs(p.src_root, p.src_path);
        in.meta = &m;
        in.mtime = p.mtime;
        in.btime = p.btime;
        in.bulk_mtime = bulk(p.id);
        in.now = now;
        in.min_year = c.min_year;
        std::vector<DateCandidate> cands = collect_date_candidates(in);
        cands.insert(cands.end(), extra.begin(), extra.end());
        if (!p.manual_date.empty()) {
            DateCandidate mc;
            mc.src = DS_MANUAL;
            mc.when = util::parse_db_datetime(p.manual_date);
            mc.base = mc.weight = mc.when.valid() ? 1.0 : 0.0;
            mc.raw = p.manual_date;
            cands.push_back(mc);
        }
        DateDecision d = decide_from_candidates(cands, c.review_below);

        // Close call between distinct dates -> ask jev.
        if (c.jev_enabled && !jev_down_ && d.decider == "rules" && d.alternatives.size() >= 2 && d.margin < c.jev_margin) {
            std::vector<std::pair<std::string, std::string>> opts;
            for (size_t k = 0; k < d.alternatives.size(); k++) opts.push_back({"d" + std::to_string(k), date_option_text(d, d.alternatives[k])});
            json state = {{"file", in.filename}, {"folder", in.dirs.empty() ? "" : in.dirs[0]}};
            JevChoice jc = jev_choice(c, "Which is the real date this photo was taken?", opts, state.dump());
            if (jc.ok) {
                std::string before = d.when.valid() ? d.when.pretty() : "undated";
                apply_jev_date(d, jc.probs, c.jev_date_weight, c.review_below);
                // Keep a record: what the rules said, what jev preferred, what was decided.
                Decision rec;
                rec.kind = "date";
                rec.photo_id = p.id;
                rec.subject = p.src_path;
                rec.question = "Which is the real date this photo was taken?";
                json o = json::array();
                size_t best = 0;
                for (size_t k = 0; k < opts.size() && k < jc.probs.size(); k++) {
                    o.push_back({opts[k].second, std::round(jc.probs[k] * 1000) / 1000});
                    if (jc.probs[k] > jc.probs[best]) best = k;
                }
                rec.options = o.dump();
                rec.rules_pick = before;
                rec.jev_pick = best < opts.size() ? opts[best].second : "";
                rec.final_pick = d.when.valid() ? d.when.pretty() : "undated";
                rec.changed = rec.final_pick != before;
                db_.add_decision(rec);
            } else {
                jev_down_ = true;
                log_.add(1, "jev decision API unavailable, continuing with rules only: " + jc.error);
            }
        }
        p.date_value = d.when.valid() ? d.when.str() : "";
        p.date_prec = util::precision_name(d.when.prec);
        p.date_source = d.source;
        p.date_conf = d.confidence;
        p.date_decider = d.decider;
        p.date_evidence = d.to_json();
        p.needs_review = d.needs_review;
        p.has_gps = meta_gps(m, p.gps_lat, p.gps_lon);
        db_.save_date(p);
    };

    for (size_t i = 0; i < todo.size() && !stop_; i++, progress.done++) {
        Photo p;
        if (!db_.load(todo[i], p)) continue;
        progress.set_current(p.src_path);
        decide_date_for(p, {});
        MetaMap m = MetaMap::from_json(p.meta_json);
        DateInputs in;
        in.dirs = rel_dirs(p.src_root, p.src_path);

        // Location: metadata wins; otherwise the folder decision (cached per folder), then the file name.
        std::string loc = meta_location(m), src = loc.empty() ? "" : "metadata";
        double conf = loc.empty() ? 0 : 1;
        if (loc.empty()) {
            std::string dir = util::dirname(p.src_path);
            DirPlace dp = db_.get_dir(dir);
            if (!dp.found) {
                std::vector<PlaceCandidate> pc = place_candidates(in.dirs, "");
                PlaceDecision pd;
                if (!pc.empty()) {
                    std::vector<PlaceVote> votes{rules_vote(pc, c.loc_w_rules)};
                    if (c.vl_enabled) {
                        std::vector<std::string> parts;
                        for (auto& x : pc) parts.push_back(x.text);
                        VlPlace vp = vl_place_from_names(c, parts);
                        if (vp.ok) {
                            PlaceVote v{"vl", c.loc_w_vl, std::vector<double>(pc.size(), 0.0), "answer: " + vp.place};
                            int idx = match_candidate(pc, vp.place);
                            if (idx >= 0) v.scores[size_t(idx)] = vp.confidence;
                            votes.push_back(v);
                        } else {
                            log_.add(1, "VL place check failed: " + vp.error);
                        }
                    }
                    if (c.jev_enabled && !jev_down_) {
                        std::vector<std::pair<std::string, std::string>> opts;
                        for (size_t k = 0; k < pc.size(); k++)
                            opts.push_back({"p" + std::to_string(k), "\"" + pc[k].text + "\" is the name of a city, country, region, park or landmark"});
                        opts.push_back({"none", "None of these folder names is a geographic place"});
                        JevChoice jc = jev_choice(c, "Which of these folder names is the place where the photos were taken?", opts, "{}");
                        if (jc.ok) {
                            PlaceDecision without = decide_place(pc, votes, c.loc_accept);
                            PlaceVote v{"jev", c.loc_w_jev, std::vector<double>(jc.probs.begin(), jc.probs.end() - 1), "p(none)=" + util::fmt("%.2f", jc.probs.back())};
                            votes.push_back(v);
                            PlaceDecision with = decide_place(pc, votes, c.loc_accept);
                            Decision rec;
                            rec.kind = "place";
                            rec.subject = dir;
                            rec.question = "Which of these folder names is the place where the photos were taken?";
                            json o = json::array();
                            size_t best = 0;
                            for (size_t k = 0; k < opts.size() && k < jc.probs.size(); k++) {
                                o.push_back({opts[k].second, std::round(jc.probs[k] * 1000) / 1000});
                                if (jc.probs[k] > jc.probs[best]) best = k;
                            }
                            rec.options = o.dump();
                            rec.rules_pick = without.text.empty() ? "(no place)" : without.text;
                            rec.jev_pick = best < pc.size() ? pc[best].text : "(no place)";
                            rec.final_pick = with.text.empty() ? "(no place)" : with.text;
                            rec.changed = rec.final_pick != rec.rules_pick;
                            db_.add_decision(rec);
                        } else {
                            jev_down_ = true;
                            log_.add(1, "jev decision API unavailable, continuing with rules only: " + jc.error);
                        }
                    }
                    pd = decide_place(pc, votes, c.loc_accept);
                }
                dp.found = true;
                dp.location = pd.text;
                dp.conf = pd.confidence;
                dp.source = pd.source;
                dp.evidence = pd.evidence_json;
                db_.put_dir(dir, dp);
                if (!pd.text.empty()) log_.add(0, "location \"" + pd.text + "\" for folder " + dir);
            }
            loc = dp.location;
            src = dp.source;
            conf = dp.conf;
        }
        if (loc.empty()) {
            // A place in the file name itself ("Paris_0012.jpg"): rules only, and only with a trip cue.
            std::vector<PlaceCandidate> pc = place_candidates({}, util::stem(p.src_path));
            for (auto& x : pc)
                if (x.travel_cue && x.prior >= c.loc_accept) { loc = x.text; src = "file-name"; conf = x.prior; break; }
        }
        if (loc.empty() && p.has_gps) src = "gps";
        db_.save_location(p.id, loc, src, conf);
    }

    // Pass 3: a weak photo inside a numbered camera sequence takes the day of its confident neighbours
    // (IMG_0002 on 05-13, IMG_0003 with a reset clock, IMG_0004 on 05-14).
    if (stop_) return;
    struct Seq {
        int64_t id;
        std::string prefix;
        long num;
        util::Civil when;
        bool strong;
    };
    static const std::regex seq_re("^(.*?)([0-9]{3,6})$");
    std::map<std::string, std::vector<Seq>> by_dir;
    for (int64_t id : db_.ids("resolved=1")) {
        Photo p;
        if (!db_.load(id, p)) continue;
        std::smatch mm;
        std::string st = util::stem(p.src_path);
        if (!std::regex_match(st, mm, seq_re)) continue;
        by_dir[util::dirname(p.src_path)].push_back({id, util::lower(mm[1].str()), std::stol(mm[2].str()),
                                                     util::parse_db_datetime(p.date_value + "|" + p.date_prec),
                                                     !p.needs_review && p.date_conf >= 0.8 && p.date_prec != "month" && p.date_prec != "year"});
    }
    std::set<int64_t> todo_set(todo.begin(), todo.end());
    int fixed = 0;
    for (auto& [dir, list] : by_dir) {
        for (auto& s : list) {
            if (s.strong || !todo_set.count(s.id)) continue;
            const Seq *prev = nullptr, *next = nullptr;
            for (auto& o : list) {
                if (!o.strong || o.prefix != s.prefix || std::labs(o.num - s.num) > 20) continue;
                if (o.num < s.num && (!prev || o.num > prev->num)) prev = &o;
                if (o.num > s.num && (!next || o.num < next->num)) next = &o;
            }
            if (!prev && !next) continue;
            DateCandidate nc;
            nc.src = DS_NEIGHBOR;
            const Seq* base = prev ? prev : next;
            nc.when = util::make_civil(base->when.y, base->when.mo, base->when.d, 0, 0, 0, util::P_DAY);
            if (prev && next) {
                int64_t gap = std::llabs(util::days_from_civil(prev->when.y, prev->when.mo, prev->when.d) -
                                         util::days_from_civil(next->when.y, next->when.mo, next->when.d));
                if (gap > 3) continue;  // the sequence jumps: no safe guess
                nc.base = nc.weight = 0.60;
                nc.raw = util::fmt("between #%ld (%s) and #%ld (%s)", prev->num, prev->when.pretty().c_str(), next->num, next->when.pretty().c_str());
            } else {
                nc.base = nc.weight = 0.40;
                nc.raw = util::fmt("next to #%ld (%s)", base->num, base->when.pretty().c_str());
            }
            Photo p;
            if (!db_.load(s.id, p)) continue;
            decide_date_for(p, {nc});
            fixed++;
        }
    }
    if (fixed) log_.add(0, util::fmt("decide: %d weak photos re-dated from their camera sequence", fixed));
}

// ---- stage 3a: tag (CLIP on the thumbnails; also the semantic search index)

void Pipeline::tag(const Config& c_in, const RunOptions& o) {
    Config c = c_in;
    if (o.clip_mode == 2) c.clip_model = "b32";  // quick: the small model for this run
    ClipInfo info = clip_choose(c);
    std::vector<int64_t> todo;
    if (o.clip_mode == 1) todo = scoped("(dup_of IS NULL OR dup_of=0)", o);  // every photo again
    else if (o.clip_mode != 3) todo = scoped("(dup_of IS NULL OR dup_of=0) AND (clip_model IS NULL OR clip_model<>'" + info.id + "')", o);
    std::vector<int64_t> tagged = scoped("(dup_of IS NULL OR dup_of=0) AND clip_model='" + info.id + "'", o);
    if (o.clip_mode == 1) tagged.clear();  // they are all re-encoded (and re-tagged) anyway
    bool chosen = !o.reanalyse_ids.empty() || !o.retag_ids.empty();  // the CLIP tab's smart update
    if (chosen) {
        std::set<int64_t> re(o.reanalyse_ids.begin(), o.reanalyse_ids.end()), rt(o.retag_ids.begin(), o.retag_ids.end());
        todo = scoped("(dup_of IS NULL OR dup_of=0)", o);
        todo.erase(std::remove_if(todo.begin(), todo.end(), [&](int64_t id) { return !re.count(id); }), todo.end());
        tagged.erase(std::remove_if(tagged.begin(), tagged.end(), [&](int64_t id) { return !rt.count(id) || re.count(id); }), tagged.end());
    }
    ClipModel model;
    std::string err;
    if (!model.load(info, c.clip_device, c.clip_threads, !todo.empty(), true, err)) {
        begin(ST_TAG, 0);
        log_.add(2, "CLIP not available (" + err + "); tag stage skipped");
        return;
    }
    TagIndex idx;
    if (!build_tag_index(model, idx, err)) {
        begin(ST_TAG, 0);
        log_.add(2, "CLIP tag vocabulary failed: " + err);
        return;
    }
    // Photos already embedded with this model only need their tags recomputed when the vocabulary (or the way tags
    // are picked) changed: that is a dot product per tag, no image decoding.
    std::vector<int64_t> retag;
    for (int64_t id : tagged) {
        Photo p;
        if (!db_.load(id, p)) continue;
        bool again = chosen || o.retag_all || o.tag_mode == 1 || p.clip_vocab != idx.vocab_hash;
        if (!again && o.tag_mode == 2) {  // few confident tags: worth another look with the current list
            int sure = 0;
            for (auto& [t, conf] : parse_tag_list(p.clip_tags)) sure += conf >= kTagWordMin;
            again = sure < 2;
        }
        if (again) retag.push_back(id);
    }
    begin(ST_TAG, int(todo.size() + retag.size()));
    auto save_tags = [&](int64_t id, const Vec& v, bool with_vec) {
        ImageTags t = pick_tags(idx, v, c.clip_max_tags);
        json tags = json::array();
        for (auto& [name, p] : t.tags) tags.push_back({name, std::round(p * 1000) / 1000});
        if (with_vec) db_.save_clip(id, info.id, v, tags.dump(), t.scene, idx.vocab_hash);
        else db_.save_tags(id, tags.dump(), t.scene, idx.vocab_hash);
    };
    if (!retag.empty()) {
        log_.add(0, util::fmt("tag: re-tagging %zu photos with the current vocabulary (%zu tags)", retag.size(), idx.tags.size()));
        for (int64_t id : retag) {
            if (stop_) return;
            Vec v;
            if (db_.clip_vec(id, v)) save_tags(id, v, false);
            else progress.skipped++;
            progress.done++;
        }
    }
    if (todo.empty()) return;
    log_.add(0, util::fmt("tag: %zu photos with CLIP %s on %s, %zu-tag vocabulary", todo.size(), info.label.c_str(),
                          model.device_used().c_str(), idx.tags.size()));
    // Batches of 16: thumbnails are prepared in parallel, then one model call per batch.
    const size_t kBatch = 16;
    for (size_t at = 0; at < todo.size() && !stop_; at += kBatch) {
        size_t n = std::min(kBatch, todo.size() - at);
        std::vector<Photo> ps(n);
        std::vector<std::vector<float>> px(n);
        std::vector<bool> ok(n, false);
        std::vector<std::thread> th;
        for (size_t k = 0; k < n; k++)
            th.emplace_back([&, k] {
                if (!db_.load(todo[at + k], ps[k])) return;
                MetaMap m = MetaMap::from_json(ps[k].meta_json);
                std::string e, tp = ensure_thumb({ps[k].src_path, ps[k].size, ps[k].mtime}, meta_orientation(m),
                                                 thumb_side_for(ps[k].src_path, c.thumb_side, c.vl_max_side), e);
                ok[k] = !tp.empty() && clip_preprocess(tp, px[k], e);
            });
        for (auto& t : th) t.join();
        std::vector<std::vector<float>> batch;
        std::vector<size_t> which;
        for (size_t k = 0; k < n; k++)
            if (ok[k]) { batch.push_back(std::move(px[k])); which.push_back(k); }
            else { progress.skipped++; progress.done++; }
        if (batch.empty()) continue;
        progress.set_current(ps[which[0]].src_path);
        std::vector<Vec> emb;
        if (!model.encode_images(batch, emb, err)) {
            log_.add(2, "CLIP failed: " + err);
            progress.errors += int(batch.size());
            progress.done += int(batch.size());
            continue;
        }
        for (size_t j = 0; j < which.size() && j < emb.size(); j++) {
            save_tags(ps[which[j]].id, emb[j], true);
            progress.done++;
        }
    }
}

// ---- stage 3b: keywords from prose prompts (tag-style prompts were split at scan time). One LLM call per distinct
// prompt: a batch of renders usually shares one.

void Pipeline::prompt_keywords_llm(const Config& c, const RunOptions& o) {
    // Long file names ("a_beautiful_sunset_over_the_mountains_with_a_lake_4k_wallpaper"): the LLM picks the key words,
    // which also become the short name prefix. Decided once (confidence 1.0 marks a decided list).
    int named = 0;
    for (int64_t id : scoped("name_keywords IS NOT NULL AND name_keywords NOT LIKE '%1.0]%' AND name_keywords<>'[]'", o)) {
        if (stop_) return;
        Photo p;
        if (!db_.load(id, p)) continue;
        std::string prefix = name_prefix(util::stem(p.src_path));
        if (!long_name(prefix)) continue;
        std::string err, reply = llm_ask(c, "A photo file is named: \"" + prefix +
                                                "\". Pick the 2 to 4 words of this name that best say what the picture is about (keep their "
                                                "language; leave out filler, quality and size words). Reply JSON only: {\"keywords\": [\"...\"]}",
                                         80, true, err);
        json j = json::parse(extract_json_object(reply), nullptr, false);
        json w = json::array();
        if (j.is_object() && j.contains("keywords") && j["keywords"].is_array())
            for (auto& k : j["keywords"])
                if (k.is_string() && !util::trim(k.get<std::string>()).empty() && w.size() < 4) w.push_back({util::lower(util::trim(k.get<std::string>())), 1.0});
        if (w.empty()) {
            if (!err.empty()) break;  // LLM down: the rules' words stay
            continue;
        }
        db_.save_name_keywords(id, w.dump());
        named++;
    }
    if (named) log_.add(0, util::fmt("name keywords: %d long file names reduced to their key words", named));
    std::vector<int64_t> todo = scoped("COALESCE(gen_json,'')<>'' AND COALESCE(gen_keywords,'')=''", o);
    if (todo.empty()) return;
    begin(ST_TAG, int(todo.size()));
    std::map<std::string, std::string> done;  // prompt -> keywords JSON
    int calls = 0, failed = 0, declined = 0;
    for (int64_t id : todo) {
        if (stop_) return;
        progress.done++;
        Photo p;
        if (!db_.load(id, p)) continue;
        GenInfo g = GenInfo::from_json(p.gen_json);
        std::string pr = clean_prompt(g.prompt);
        if (pr.empty()) continue;
        if (!done.count(pr)) {
            progress.set_current("prompt keywords: " + util::basename(p.src_path));
            std::string err, prompt =
                "This is the prompt an image generator was given:\n\"\"\"\n" + pr.substr(0, 3000) +
                "\n\"\"\"\nList up to 10 short English keywords (1-3 words each) for what the resulting picture shows: subjects, "
                "clothing, objects, setting, art style. Lower case. Reply JSON only: {\"keywords\": [\"...\"]}";
            std::string reply = llm_ask(c, prompt, 200, true, err);
            json j = json::parse(extract_json_object(reply), nullptr, false);
            json kw = json::array();
            if (j.is_object() && j.contains("keywords") && j["keywords"].is_array())
                for (auto& k : j["keywords"])
                    if (k.is_string() && !util::trim(k.get<std::string>()).empty() && kw.size() < 10) kw.push_back(util::lower(util::trim(k.get<std::string>())));
            calls++;
            if (kw.empty() && err.empty()) {
                // The LLM answered but gave nothing (it declines some prompts): fall back to the prompt's own words.
                for (auto& k : local_keywords(g.prompt)) kw.push_back(k);
                if (!kw.empty()) declined++;
            }
            if (kw.empty()) {
                failed++;
                if (!err.empty() && failed == 1) log_.add(1, "prompt keywords: LLM unavailable (" + err + ")");
                if (failed >= 3 && calls == failed) {  // the LLM is down: next run
                    log_.add(1, util::fmt("prompt keywords: the LLM is unavailable%s; tried again next run", err.empty() ? "" : (" (" + err + ")").c_str()));
                    return;
                }
                continue;
            }
            done[pr] = kw.dump();
        }
        db_.save_gen_keywords(id, done[pr]);
    }
    log_.add(0, util::fmt("prompt keywords: %zu photos, %d distinct prompts read by the LLM", todo.size(), calls) +
                    (declined ? util::fmt(", %d declined by the LLM (keywords taken from the prompt's own words instead)", declined) : "") +
                    (failed ? util::fmt(", %d gave no keywords (tried again next run; the prompt text itself is still searchable)", failed) : ""));
}

// ---- stage 3c: corrections. Re-check what jev-photos generated itself (CLIP tags, the names it gave) against the
// rest of the evidence, and propose specific fixes with a confidence. Nothing is changed here: the proposals go to
// Review. The file's own metadata (prompt, camera, lens, dates, descriptions others wrote) is evidence, never a target.

namespace {
const std::set<std::string> kArtMedia = {"screenshot", "scanned document", "illustration", "anime", "cartoon", "digital art", "3d render",
                                         "painting", "drawing", "pencil sketch", "meme"};
const std::set<std::string> kScreenTags = {"screenshot", "website", "app screen", "chat conversation", "code", "presentation slide",
                                           "text document", "qr code"};
bool screenshot_name(const std::string& name) {
    std::string n = util::lower(name);
    for (const char* k : {"screenshot", "screen shot", "screencap", "\xe6\x88\xaa\xe5\xb1\x8f", "\xe6\x88\xaa\xe5\x9b\xbe",
                          "\xe3\x82\xb9\xe3\x82\xaf\xe3\x83\xaa\xe3\x83\xbc\xe3\x83\xb3\xe3\x82\xb7\xe3\x83\xa7\xe3\x83\x83\xe3\x83\x88",
                          "\xec\x8a\xa4\xed\x81\xac\xeb\xa6\xb0\xec\x83\xb7"})  // 截屏 截图 スクリーンショット 스크린샷
        if (n.find(k) != std::string::npos) return true;
    return false;
}
}  // namespace

void Pipeline::check_corrections(const Config& c, const RunOptions& o) {
    std::vector<int64_t> ids = scoped("(dup_of IS NULL OR dup_of=0) AND COALESCE(clip_model,'')<>''", o);
    begin(ST_FIX, int(ids.size()));
    if (ids.empty()) return;
    // CLIP text side, to verify proposed tags against each photo's stored embedding.
    ClipInfo info = clip_choose(c);
    ClipModel model;
    TagIndex idx;
    std::string err;
    bool clip_ok = c.clip_enabled && model.load(info, c.clip_device, std::max(1, c.clip_threads / 2), false, true, err) && build_tag_index(model, idx, err);
    if (!clip_ok) log_.add(1, "corrections: CLIP unavailable (" + err + "), checking without it");
    bool llm_ok = c.llm_enabled;
    std::map<std::string, json> llm_cache;  // prompt + tags -> the LLM's verdict
    int checked = 0, proposed = 0;
    for (int64_t id : ids) {
        if (stop_) break;
        progress.done++;
        Photo p;
        if (!db_.load(id, p)) continue;
        Vec vec;
        if (clip_ok && p.clip_model == info.id) db_.clip_vec(id, vec);
        auto tags = effective_tag_list(p);
        GenInfo gen = GenInfo::from_json(p.gen_json);
        MetaMap m = MetaMap::from_json(p.meta_json);
        // Same inputs as last time: nothing new to say.
        std::string key = p.clip_tags + "|" + p.tag_fix + "|" + p.user_tags + "|" + p.gen_keywords + "|" + p.name_keywords + "|" + p.dest_path + "|" +
                          std::to_string(c.name_style);
        char hk[32];
        snprintf(hk, sizeof hk, "%016llx", (unsigned long long)std::hash<std::string>{}(key));
        if (!o.recheck_all && p.fix_checked == hk) continue;
        checked++;
        progress.set_current("checking " + util::basename(p.src_path));
        std::vector<Correction> out;
        std::set<std::string> have;
        for (auto& [t, conf] : tags) have.insert(util::lower(t));
        auto propose = [&](const std::string& field, const std::string& action, const std::string& cur, const std::string& to, double conf,
                           const std::string& why, const std::string& src) {
            if (conf < 0.3) return;  // not worth a look
            if (field == "tag" && action == "add" && tag_blocked(to)) return;  // you deleted that tag
            for (auto& x : out)
                if (x.field == field && x.current == cur && x.proposed == to) return;
            if (db_.correction_rejected(id, field, cur, to)) return;  // you said no (or it is done)
            Correction k;
            k.photo_id = id; k.field = field; k.action = action; k.current = cur; k.proposed = to;
            k.confidence = std::round(conf * 100) / 100; k.reason = why; k.source = src;
            out.push_back(k);
        };
        auto clip_conf = [&](const std::string& t) {
            std::string e;
            return vec.empty() ? -1.0 : phrase_confidence(model, idx, t, vec, e);
        };
        if (p.user_tags.empty()) {  // your own tags are never second-guessed
            // 1. Camera photos are photographs: art / screen tags on them are wrong.
            std::string make = util::trim(m.get("Exif.Image.Make")), cam = util::trim(m.get("Exif.Image.Model"));
            bool camera = !gen.found() && !make.empty() &&
                          (m.has("Exif.Photo.ExposureTime") || m.has("Exif.Photo.FNumber") || m.has("Exif.Photo.LensModel") || m.has("Exif.Photo.ISOSpeedRatings"));
            if (camera)
                for (auto& [t, conf] : tags)
                    if (kArtMedia.count(util::lower(t)) || kScreenTags.count(util::lower(t)))
                        propose("tag", "remove", t, "", 0.9 - 0.3 * conf, "taken with a camera (" + make + " " + cam + "), so it is a photograph",
                                "camera EXIF");
            // 2. Named as a screenshot (any language) and no camera: it is one.
            if (!camera && screenshot_name(util::basename(p.src_path)) && !have.count("screenshot"))
                propose("tag", "add", "", "screenshot", 0.85, "the file name says it is a screenshot", "file name");
            // 3. AI images: the prompt says what the picture is. The LLM checks the tags against it (one call per distinct
            //    prompt + tag set). Its words are already keywords, so only wrong tags are proposed here.
            // Edits (an input image) are skipped: the prompt only says what changed, and the source decides the rest.
            if (gen.found() && !gen.prompt.empty() && !tags.empty() && gen.sources.empty()) {
                std::string tag_list;
                for (auto& [t, conf] : tags) tag_list += (tag_list.empty() ? "" : ", ") + t + util::fmt(" (%.0f%%)", conf * 100);
                std::string ckey = clean_prompt(gen.prompt).substr(0, 2000) + "\x1f" + tag_list;
                json j;
                if (llm_cache.count(ckey)) j = llm_cache[ckey];
                else if (llm_ok) {
                    std::string prompt =
                        "An image was generated from this prompt:\n\"\"\"\n" + clean_prompt(gen.prompt).substr(0, 2000) +
                        "\n\"\"\"\nAn image classifier tagged the result: " + tag_list +
                        ".\nSome tags say what KIND of picture it is (photo, screenshot, anime, digital art, 3d render, painting, drawing, "
                        "scanned document, meme...). Others say what it shows.\nWhich classifier tags are clearly wrong for an image made from "
                        "this prompt? A kind-of-picture tag is wrong when the prompt asks for a different kind (for example \"screenshot\" when "
                        "the prompt asks for a photo or a portrait). A content tag is wrong only when the prompt contradicts it. A tag is NOT "
                        "wrong just because the prompt does not mention it: generators add details, backgrounds and objects of their own.\n" +
                        "Reply JSON only: {\"wrong\": [{\"tag\": \"...\", \"confidence\": 0-100, \"why\": \"short reason\"}]}";
                    std::string e, reply = llm_ask(c, prompt, 400, true, e);
                    j = json::parse(extract_json_object(reply), nullptr, false);
                    if (j.is_object()) llm_cache[ckey] = j;
                    else if (!e.empty()) {
                        llm_ok = false;
                        log_.add(1, "corrections: LLM unavailable (" + e + "), using rules only");
                    }
                }
                if (j.is_object() && j.contains("wrong") && j["wrong"].is_array()) {
                    for (auto& w : j["wrong"]) {
                        if (!w.is_object() || !w.contains("tag") || !w["tag"].is_string()) continue;
                        std::string t = w["tag"].get<std::string>();
                        float tc = -1;
                        for (auto& [tt, cc] : tags)
                            if (util::lower(tt) == util::lower(t)) tc = cc, t = tt;
                        if (tc < 0) continue;  // not one of its tags
                        double lc = std::clamp(w.value("confidence", 0.0) / 100.0, 0.0, 1.0);
                        bool kind = kArtMedia.count(util::lower(t)) || util::lower(t) == "photo" || util::lower(t) == "black and white photo" ||
                                    util::lower(t) == "old photo";
                        if (util::lower(gen.prompt).find(util::lower(t)) != std::string::npos) continue;  // the prompt asks for it
                        // What CLIP sees clearly needs a stronger case than what it barely saw.
                        lc *= 1.0 - (kind ? 0.3 : 0.5) * tc;
                        std::string why = w.value("why", std::string("does not fit"));
                        if (why.size() > 160) why = why.substr(0, why.find_last_of(" ,.", 157)) + "…";
                        propose("tag", "remove", t, "", std::min(0.95, lc), "prompt: " + why, "prompt + LLM");
                    }
                } else {
                    // No LLM answer (down, or it declines the prompt): a screen tag on a generated picture whose prompt
                    // is not about screens is wrong.
                    std::string pl = util::lower(gen.prompt);
                    bool about_screens = pl.find("screen") != std::string::npos || pl.find("website") != std::string::npos ||
                                         pl.find(" ui") != std::string::npos || pl.find("interface") != std::string::npos;
                    for (auto& [t, conf] : tags)
                        if (kScreenTags.count(util::lower(t)) && !about_screens)
                            propose("tag", "remove", t, "", 0.75 - 0.2 * conf, "a generated picture whose prompt is not about screens", "prompt");
                }
            }
            // 4. Words of the file name that the picture really shows (checked by CLIP).
            if (!gen.found() && !vec.empty())
                for (auto& [w, conf] : parse_tag_list(p.name_keywords)) {
                    if (have.count(w) || util::utf8_len(w) < 3) continue;
                    double cc = clip_conf(w);
                    if (cc >= 0.4) propose("tag", "add", "", w, 0.3 + 0.6 * cc, util::fmt("in the file name, and CLIP sees it (%.0f%%)", cc * 100),
                                           "file name + CLIP");
                }
        }
        // 5. Names jev-photos gave that kept a long original name: shorten to its key words.
        if (c.name_style == NAME_KEEP_PREFIX && !p.dest_path.empty() && util::file_exists(p.dest_path)) {
            std::string stem = util::stem(p.dest_path), day;
            int sn = 0;
            if (parse_serial_name(stem, c.name_sep, day, sn)) {
                std::string tail = day + c.name_sep + util::fmt("%0*d", c.sn_digits, sn);
                std::string cur_prefix = stem.size() > tail.size() + c.name_sep.size() ? stem.substr(0, stem.size() - tail.size() - c.name_sep.size()) : "";
                std::string want = short_prefix(p);
                if (long_name(cur_prefix) && !want.empty() && want != cur_prefix) {
                    std::string to = want + c.name_sep + tail + "." + util::ext_lower(p.dest_path);
                    bool decided = p.name_keywords.find("1.0]") != std::string::npos;
                    propose("name", "rename", util::basename(p.dest_path), to, decided ? 0.8 : 0.6, "a long name shortened to its key words",
                            decided ? "file name + LLM" : "file name");
                }
            }
        }
        db_.replace_corrections({id}, out);
        db_.set_fix_checked(id, hk);
        proposed += int(out.size());
    }
    log_.add(0, util::fmt("corrections: %d photos checked, %d proposals to review", checked, proposed));
}

// ---- stage 3: vision

void Pipeline::vision(const Config& c, const RunOptions& o) {
    std::vector<int64_t> todo = scoped(std::string("(dup_of IS NULL OR dup_of=0) AND (vision_status IS NULL OR vision_status=''") +
                                           (o.retry_failed_vision ? " OR vision_status='failed')" : ")"), o);
    begin(ST_VISION, int(todo.size()));
    if (todo.empty()) return;
    std::string detail;
    if (!vl_health(c, detail)) {
        log_.add(2, "VL model API not reachable at " + c.vl_url + " (" + detail + "); vision stage skipped");
        return;
    }
    if (detail != "ok") log_.add(1, "VL: " + detail + " (" + c.vl_model + ")");
    log_.add(0, util::fmt("vision: %zu photos with %s", todo.size(), c.vl_model.c_str()));
    std::atomic<int> consecutive_fail{0};
    parallel(int(todo.size()), c.vl_concurrency, [&](int i) {
        if (consecutive_fail >= 5) { stop_ = true; return; }
        Photo p;
        if (!db_.load(todo[size_t(i)], p)) return;
        progress.set_current(p.src_path);
        MetaMap m = MetaMap::from_json(p.meta_json);
        std::string jpeg, err;
        p.vision_model = c.vl_model;
        std::string tp = ensure_thumb({p.src_path, p.size, p.mtime}, meta_orientation(m),
                                      thumb_side_for(p.src_path, c.thumb_side, c.vl_max_side), err);
        if (tp.empty() || !util::read_file(tp, jpeg)) {
            p.vision_status = "unsupported";
            p.vision_error = err;
            progress.skipped++;
            db_.save_vision(p);
            return;
        }
        VisionResult r = vision_describe(c, jpeg);
        if (!r.ok) {
            p.vision_status = "failed";
            p.vision_error = r.error;
            progress.errors++;
            log_.add(1, "vision failed (" + r.error + "): " + p.src_path);
            if (++consecutive_fail >= 5) log_.add(2, "vision: 5 failures in a row, stopping");
        } else {
            consecutive_fail = 0;
            p.vision_status = "done";
            p.vision_error.clear();
            p.caption = r.caption;
            p.scene = r.scene;
            p.objects = json(r.objects).dump();
            p.tags = json(r.tags).dump();
            p.landmark = r.landmark;
            p.vision_text = r.text;
            p.people = r.people;
        }
        db_.save_vision(p);
    });
}

// ---- stage 4: organize = plan (pure) + apply

bool path_under(const std::string& path, const std::string& dir) {
    if (dir.empty()) return true;
    if (dir.find('\n') != std::string::npos) {  // several folders, one per line
        for (auto& d : util::split(dir, '\n'))
            if (!util::trim(d).empty() && path_under(path, util::trim(d))) return true;
        return false;
    }
    std::string d = dir;
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return path == d || (path.size() > d.size() && path.compare(0, d.size(), d) == 0 && path[d.size()] == '/');
}

std::vector<int64_t> Pipeline::scoped(const std::string& where, const RunOptions& o) {
    std::vector<int64_t> ids = db_.ids(where);
    if (o.scope.empty()) return ids;
    std::vector<int64_t> out;
    for (int64_t id : ids) {
        Photo p;
        if (db_.load(id, p) && path_under(p.src_path, o.scope)) out.push_back(id);
    }
    return out;
}

std::vector<PlanItem> Pipeline::plan() {
    std::lock_guard<std::mutex> l(plan_mu_);
    return plan_;
}

bool Pipeline::has_plan() {
    std::lock_guard<std::mutex> l(plan_mu_);
    return !plan_.empty();
}

void Pipeline::set_include(int64_t id, bool include) {
    std::lock_guard<std::mutex> l(plan_mu_);
    for (auto& it : plan_)
        if (it.id == id && it.action != "duplicate") it.include = include;
}

void Pipeline::set_include_where(bool (*pred)(const PlanItem&), bool include) {
    std::lock_guard<std::mutex> l(plan_mu_);
    for (auto& it : plan_)
        if (it.action != "duplicate" && pred(it)) it.include = include;
}

namespace {

MetaPlan meta_plan_for(const Photo& p, const util::Civil& when, bool rewrite = false, const MetaMap* target_now = nullptr,
                       const std::string& target_path = "", const Config* c = nullptr, const std::string& desc_action = "") {
    MetaPlan plan;
    if (c && c->write_description) plan.description = our_description(*c, p);
    plan.desc_action = desc_action;
    plan.date = when;
    plan.date_conf = p.date_conf;
    plan.date_source = p.date_source;
    plan.date_trusted = !p.needs_review;
    plan.caption = p.caption;
    plan.scene = p.scene.empty() ? p.clip_scene : p.scene;
    plan.landmark = p.landmark;
    plan.location = p.location;
    plan.location_source = p.location_source;
    for (auto& k : effective_tags(p))
        if (util::lower(k) != "photo") plan.keywords.push_back(k);  // "photo" says nothing about a photo
    for (auto& k : json_list(p.objects)) plan.keywords.push_back(k);
    if (!c || c->gen_keywords)  // what the picture was generated from
        for (auto& [k, conf] : parse_tag_list(p.gen_keywords))
            if (plan.keywords.size() < 40) plan.keywords.push_back(k);
    // A vision-model scene is a keyword; the CLIP scene already is one of the tags when it is confident.
    if (!p.scene.empty() && p.user_tags.empty()) plan.keywords.push_back(p.scene);
    if (!p.location.empty()) plan.keywords.push_back(p.location);
    if (!p.landmark.empty()) plan.keywords.push_back(p.landmark);
    plan.keywords.erase(std::remove_if(plan.keywords.begin(), plan.keywords.end(), [](const std::string& k) { return tag_blocked(k); }),
                        plan.keywords.end());
    plan.rewrite_keywords = rewrite;
    // A copy we made carries the original's keywords plus ours: anything beyond the original's list is ours (files
    // written before Xmp.jev.Keywords existed have no other record).
    if (rewrite && target_now && !target_path.empty() && target_path != p.src_path) {
        std::set<std::string> orig;
        for (auto& k : MetaMap::from_json(p.meta_json).list("Xmp.dc.subject")) orig.insert(util::lower(k));
        for (auto& k : target_now->list("Xmp.dc.subject"))
            if (!orig.count(util::lower(k))) plan.previous_keywords.push_back(k);
    }
    return plan;
}

// Metadata the target currently has. A planned copy is byte-identical to the source, so its snapshot stands in.
MetaMap existing_meta(const Photo& p, const std::string& dest, MetaTarget target) {
    // The original's scan snapshot is current (it is refreshed after every write), so only copies are re-read.
    bool placed = dest != p.src_path && util::file_exists(dest);
    MetaMap img = placed && target == MetaTarget::Embed ? read_meta(dest) : MetaMap::from_json(p.meta_json);
    if (target == MetaTarget::Embed) return img;
    MetaMap side = util::file_exists(sidecar_path(dest)) ? read_meta(sidecar_path(dest)) : MetaMap{};
    for (auto& [k, v] : img.kv)  // the sidecar must not contradict what the image already carries
        if (!side.kv.count(k)) { side.kv[k] = v; side.type[k] = img.type[k]; }
    return side;
}

}  // namespace

std::string our_description(const Config& c, const Photo& p) {
    (void)c;
    std::vector<std::string> parts;
    GenInfo g = GenInfo::from_json(p.gen_json);
    if (g.found()) {
        std::string pr = clean_prompt(g.prompt);
        for (auto& ch : pr)
            if (ch == '\n') ch = ' ';
        if (pr.size() > 1500) {
            size_t cut = 1500;
            while (cut > 0 && (static_cast<unsigned char>(pr[cut]) & 0xC0) == 0x80) cut--;
            pr = pr.substr(0, cut) + "…";
        }
        if (!pr.empty()) parts.push_back(pr);
    }
    if (!p.caption.empty()) parts.push_back(p.caption);
    std::string shows;
    for (auto& t : effective_tags(p))
        if (util::lower(t) != "photo") shows += (shows.empty() ? "" : ", ") + t;
    if (!shows.empty()) parts.push_back("Shows: " + shows + ".");
    std::string out;
    for (auto& x : parts) out += (out.empty() ? "" : " \xE2\x80\x94 ") + x;
    return out;
}

bool placeholder_description(const std::string& text, const std::string& file_stem) {
    std::string t = util::lower(util::trim(text));
    if (t.size() < 3 || t == util::lower(file_stem)) return true;
    static const std::regex junk(
        R"(^(olympus digital camera|sony dsc|digital camera|minolta digital camera|kodak.*|samsung.*|lg electronics.*|default|untitled.*|image|photo|picture|screenshot|dcim.*|created with .*|made with .*|ilce-?\w*|nikon.*|canon.*|(img|dsc|dscn|dscf|pxl|p|mvimg|vid)[_-]?\d+.*|\d+|[-_. ]+)$)");
    return std::regex_match(t, junk);
}

bool description_covers(const std::string& have, const std::string& ours) {
    std::string h = util::lower(have);
    std::string o = util::lower(ours);
    if (h.find(o) != std::string::npos) return true;
    // every content word of ours already appears in theirs
    static const std::set<std::string> stop = {"shows", "a", "an", "the", "of", "and", "with", "in", "on", "at", "photo"};
    int words = 0, found = 0;
    std::string w;
    auto flush = [&] {
        if (w.size() > 1 && !stop.count(w)) { words++; found += h.find(w) != std::string::npos; }
        w.clear();
    };
    for (unsigned char ch : o) {
        if (isalnum(ch) || ch >= 0x80) w += char(ch);
        else flush();
    }
    flush();
    return words > 0 && found == words;
}

// What to do with a description the file already has (only called when it has one and ours differs).
// rules first (placeholders are replaced, nothing new = keep); then the LLM decides and jev cross-checks: when they
// agree the action applies, otherwise you decide in Review. Decisions are kept per (their text, our text) pair.
void Pipeline::decide_description(const Config& c, Photo& p, PlanItem& it) {
    const std::string& E = it.desc_old;
    const std::string& O = it.desc_new;
    char key[32];
    std::string pair = E + "\x1f" + O;
    snprintf(key, sizeof key, "%016llx", (unsigned long long)std::hash<std::string>{}(pair));
    json prev = json::parse(p.desc_choice.empty() ? "{}" : p.desc_choice, nullptr, false);
    if (prev.is_object() && prev.value("key", "") == key && prev.value("action", "") != "decide") {
        it.desc_action = prev.value("action", "keep");
        it.desc_by = prev.value("by", "");
        it.desc_why = prev.value("why", "");
        return;
    }
    auto settle = [&](const std::string& action, const std::string& by, const std::string& why) {
        it.desc_action = action;
        it.desc_by = by;
        it.desc_why = why;
        json j = {{"key", key}, {"action", action}, {"by", by}, {"why", why}};
        p.desc_choice = j.dump();
        db_.save_desc_choice(p.id, p.desc_choice);
    };
    if (c.desc_policy == DESC_FILL_ONLY) return settle("keep", "settings", "existing descriptions are never changed");
    if (placeholder_description(E, util::stem(p.src_path))) return settle("replace", "rules", "camera default or placeholder text");
    if (description_covers(E, O)) return settle("keep", "rules", "it already says everything ours says");
    if (c.desc_policy == DESC_ASK) return settle("decide", "you", "");
    // LLM
    std::string action, err;
    int conf = 0;
    if (c.llm_enabled) {
        std::string prompt = "A photo's existing description is: \"" + E.substr(0, 1500) + "\"\nA new automatic description is: \"" + O.substr(0, 1500) +
                             "\"\nDecide what to do with the description field. Options:\n"
                             "- keep: the existing text already says what the new one says (or more); leave it.\n"
                             "- append: the existing text is a real description, and the new one adds useful information; put the new text after it.\n"
                             "- replace: the existing text is not a real description (a camera/software default, file name, placeholder or junk); "
                             "replace it (the old text is backed up).\nReply JSON only: {\"action\": \"keep|append|replace\", \"confidence\": 0-100}";
        json j = json::parse(extract_json_object(llm_ask(c, prompt, 60, true, err)), nullptr, false);
        if (j.is_object()) {
            action = j.value("action", "");
            conf = j.contains("confidence") && j["confidence"].is_number() ? j["confidence"].get<int>() : 0;
        }
        if (action != "keep" && action != "append" && action != "replace") action.clear();
    }
    if (action.empty()) return settle("decide", "you", "the LLM could not decide" + (err.empty() ? std::string() : " (" + err + ")"));
    std::string llm_why = util::fmt("LLM: %s (%d%%)", action.c_str(), conf);
    if (c.desc_policy == DESC_LLM || !c.jev_enabled || jev_down_)
        return conf >= 70 ? settle(action, "LLM", llm_why) : settle("decide", "you", llm_why + ", not sure");
    // jev cross-check: two yes/no statements (the form Julia answers), combined into its own action.
    auto yes = [&](const std::string& y, const std::string& n, double& py) {
        JevChoice jc = jev_choice(c, "Which statement is true?", {{"yes", y}, {"no", n}}, "{}");
        if (!jc.ok) { jev_down_ = true; log_.add(1, "jev decision API unavailable: " + jc.error); return false; }
        py = jc.probs[0];
        return true;
    };
    std::string e = E.substr(0, 400), o = O.substr(0, 400);
    double p_junk = 0, p_adds = 0;
    if (!yes("The text \"" + e + "\" is a camera or software default, a file name or a placeholder, not a description of a picture.",
             "The text \"" + e + "\" describes what is in a picture.", p_junk) ||
        !yes("Adding \"" + o + "\" to the photo description \"" + e + "\" adds new information.",
             "Adding \"" + o + "\" to the photo description \"" + e + "\" only repeats what it already says.", p_adds))
        return conf >= 70 ? settle(action, "LLM", llm_why + " (jev unavailable)") : settle("decide", "you", llm_why);
    std::string jev_action = p_junk > 0.5 ? "replace" : p_adds > 0.5 ? "append" : "keep";
    std::string why = llm_why + util::fmt(" · jev: %s (placeholder %.0f%%, adds %.0f%%)", jev_action.c_str(), p_junk * 100, p_adds * 100);
    Decision rec;
    rec.kind = "description";
    rec.photo_id = p.id;
    rec.subject = p.src_path;
    rec.question = "What to do with the existing description \"" + E.substr(0, 120) + "\"?";
    rec.options = json::array({json::array({"placeholder: replace", std::round(p_junk * 1000) / 1000}),
                               json::array({"adds information: append", std::round(p_adds * 1000) / 1000})}).dump();
    rec.rules_pick = "LLM: " + action;
    rec.jev_pick = jev_action;
    bool agree = jev_action == action;
    rec.final_pick = agree ? action : "you decide";
    rec.changed = !agree;
    db_.add_decision(rec);
    if (agree) settle(action, "LLM + jev", why);
    else settle("decide", "you", why + " — they disagree");
}

void Pipeline::set_desc_action(int64_t id, const std::string& action) {
    std::string choice;
    {
        std::lock_guard<std::mutex> l(plan_mu_);
        for (auto& it : plan_)
            if (it.id == id) {
                it.desc_action = action;
                it.desc_by = "you";
                if (action == "append" || action == "replace") it.include = true;  // you chose a change: it is applied
                if (util::starts_with(it.note, "description: your choice")) it.note.clear();
                char key[32];
                snprintf(key, sizeof key, "%016llx", (unsigned long long)std::hash<std::string>{}(it.desc_old + "\x1f" + it.desc_new));
                choice = json({{"key", key}, {"action", action}, {"by", "you"}, {"why", "your choice"}}).dump();
            }
    }
    if (!choice.empty()) db_.save_desc_choice(id, choice);
}

WriteResult preview_metadata(const Config& c, const Photo& p, std::string& file) {
    file = !p.dest_path.empty() && util::file_exists(p.dest_path) ? p.dest_path : p.src_path;
    MetaTarget target = c.write_mode == WRITE_SIDECAR ? MetaTarget::Sidecar : MetaTarget::Embed;
    MetaMap existing = existing_meta(p, file, target);
    util::Civil when = util::parse_db_datetime(p.date_value + "|" + p.date_prec);
    json dc = json::parse(p.desc_choice.empty() ? "{}" : p.desc_choice, nullptr, false);
    std::string action = dc.is_object() ? dc.value("action", "") : "";
    return write_meta(file, existing, meta_plan_for(p, when, c.rewrite_tags, &existing, file, &c, action), target, true);
}

std::vector<PlanItem> Pipeline::build_plan(const Config& c, const RunOptions& o) {
    std::vector<PlanItem> out;
    std::set<int64_t> excluded;
    {
        std::lock_guard<std::mutex> l(plan_mu_);
        for (auto& it : plan_)  // a refreshed preview keeps the user's exclusions and renumbers without them
            if (!it.include) excluded.insert(it.id);
    }
    std::vector<Photo> todo, dups;
    for (int64_t id : scoped("resolved=1", o)) {
        Photo p;
        if (!db_.load(id, p)) continue;
        (p.dup_of ? dups : todo).push_back(std::move(p));
    }
    std::stable_sort(todo.begin(), todo.end(), [](const Photo& a, const Photo& b) {
        bool ua = a.date_value.empty(), ub = b.date_value.empty();
        if (ua != ub) return ub;
        return a.date_value != b.date_value ? a.date_value < b.date_value : a.src_path < b.src_path;
    });
    begin(ST_ORGANIZE, int(todo.size()));
    std::map<std::string, int> next_sn;
    std::set<std::string> planned;
    MetaTarget target = c.write_mode == WRITE_SIDECAR ? MetaTarget::Sidecar : MetaTarget::Embed;
    bool meta_only = c.file_op == OP_METADATA;
    for (auto& p : todo) {
        if (stop_) break;
        progress.done++;
        util::Civil when = util::parse_db_datetime(p.date_value + "|" + p.date_prec);
        std::string day_key, folder;
        if (!when.valid()) {
            day_key = "00000000";
            folder = "undated";
        } else {
            int mo = when.prec >= util::P_MONTH ? when.mo : 0, d = when.prec >= util::P_DAY ? when.d : 0;
            day_key = util::fmt("%04d%02d%02d", when.y, mo, d);
            std::string ym = util::fmt("%04d-%02d", when.y, mo);
            folder = c.layout == LAYOUT_YM ? ym : c.layout == LAYOUT_Y_M ? util::fmt("%04d/%02d", when.y, mo) : util::fmt("%04d/", when.y) + ym;
        }
        bool vision_ready = (p.vision_status == "done" || !p.clip_model.empty());
        bool refile = !meta_only && !p.dest_path.empty() && p.day_key != day_key;
        bool meta_behind = !p.dest_path.empty() && c.write_mode != WRITE_DB_ONLY && vision_ready && p.meta_state != "full";
        if (!meta_only && !p.dest_path.empty() && !refile && !meta_behind) { progress.skipped++; continue; }
        if (meta_only && c.write_mode == WRITE_DB_ONLY) { progress.skipped++; continue; }

        PlanItem it;
        it.id = p.id;
        it.src = p.src_path;
        it.date = when.valid() ? when.pretty() : "undated";
        it.conf = p.date_conf;
        it.needs_review = p.needs_review;
        it.location = p.location;
        it.include = !excluded.count(p.id);
        it.dest = p.dest_path;
        it.size = p.size;
        if (meta_only) {
            // Only fill in what is missing, on the file where the photo lives now (its organized copy if there is one).
            it.action = "metadata";
            it.dest = !p.dest_path.empty() && util::file_exists(p.dest_path) ? p.dest_path : p.src_path;
            it.note = it.dest == p.src_path ? "adds missing fields to the original" : "adds missing fields to the organized copy";
        } else if (p.dest_path.empty() || refile) {
            it.action = refile ? "refile" : c.file_op == OP_MOVE ? "move" : c.file_op == OP_RENAME ? "rename" : "copy";
            it.old_dest = refile ? p.dest_path : "";
            if (it.include) {
                // Rename in place keeps each photo in its own folder; the others file into month folders.
                bool in_place = c.file_op == OP_RENAME;
                std::string library = library_for(c, p.src_root);
                std::string dir = in_place ? util::dirname(refile ? p.dest_path : p.src_path) : library + "/" + folder;
                std::string sn_key = in_place ? dir + "|" + day_key : day_key;
                std::string prefix = c.name_style == NAME_KEEP_PREFIX ? short_prefix(p) : "";
                if (!in_place) sn_key = library + "|" + day_key;  // serials are per output folder
                int sn = next_sn.count(sn_key) ? next_sn[sn_key] : db_.next_sn(day_key, in_place ? dir : library);
                for (;; sn++) {
                    it.dest = dir + "/" + (prefix.empty() ? "" : prefix + c.name_sep) + day_key + c.name_sep + util::fmt("%0*d", c.sn_digits, sn) +
                              "." + norm_ext(p.ext);
                    if (!planned.count(it.dest) && !util::file_exists(it.dest) && !util::file_exists(sidecar_path(it.dest)) &&
                        !db_.dest_taken(it.dest))
                        break;
                }
                next_sn[sn_key] = sn + 1;
                planned.insert(it.dest);
            } else {
                it.dest = "(excluded)";
            }
        } else {
            it.action = "metadata";
            it.note = "tags changed after organizing";
        }
        if (c.write_mode != WRITE_DB_ONLY && it.include) {
            std::string now_at = it.action == "refile" ? it.old_dest : it.action == "metadata" ? it.dest : it.dest;
            MetaMap existing = existing_meta(p, now_at, target);
            // An existing description that differs from ours: decide what to do (rules, LLM, jev, or you).
            if (c.write_description) {
                it.desc_old = meta_description(existing);
                it.desc_new = our_description(c, p);
                if (it.desc_new.empty()) it.desc_action.clear();
                else if (it.desc_old.empty()) it.desc_action = "add";
                else {
                    std::string prev = existing.get("Xmp.jev.Description");
                    if (!prev.empty() && it.desc_old.find(prev) != std::string::npos) it.desc_action = "update", it.desc_by = "ours";
                    else if (it.desc_old.find(it.desc_new) != std::string::npos) it.desc_action = "keep", it.desc_by = "same";
                    else decide_description(c, p, it);
                }
            }
            WriteResult w = write_meta(it.dest, existing, meta_plan_for(p, when, c.rewrite_tags, &existing, now_at, &c, it.desc_action), target, true);
            it.meta_target = w.target;
            for (auto& cmd : w.commands)
                if (!util::starts_with(cmd, "reg ")) it.meta_cmds.push_back(cmd);
            it.meta_added = w.added;
            it.meta_removed = w.removed;
            // Metadata-only items with nothing new to add (only our provenance fields) are left out.
            if (it.action == "metadata" && w.added <= 0 && w.removed.empty() && it.desc_action != "decide") {
                progress.skipped++;
                continue;
            }
            // Only waiting for your choice about an existing description: listed (so you can choose) but not ticked and
            // not counted as an update, so the list does not show the same "updates" on every run.
            if (it.action == "metadata" && w.added <= 0 && w.removed.empty() && it.desc_action == "decide") {
                it.include = false;
                it.note = "description: your choice (keep / append / replace)";
            }
        }
        if (p.needs_review) it.note = it.note.empty() ? "date needs review" : it.note + "; date needs review";
        out.push_back(std::move(it));
    }
    for (auto& p : dups) {
        PlanItem it;
        it.id = p.id;
        it.action = "duplicate";
        it.src = p.src_path;
        it.date = p.date_value;
        it.include = false;
        it.note = util::fmt("same content as #%lld, not copied (%s)", (long long)p.dup_of, util::human_size(p.size).c_str());
        out.push_back(std::move(it));
    }
    return out;
}

void Pipeline::apply_item(const Config& c, PlanItem& it) {
    Photo p;
    if (!db_.load(it.id, p)) { it.result = "photo no longer in the database"; return; }
    util::Civil when = util::parse_db_datetime(p.date_value + "|" + p.date_prec);
    if (it.action == "copy" || it.action == "move" || it.action == "rename" || it.action == "refile") {
        // The preview may be old: re-check everything it assumed.
        if (util::file_exists(it.dest) || db_.dest_taken(it.dest)) { it.result = "destination taken since the preview; refresh it"; return; }
        struct stat st{};
        std::string from = it.action == "refile" ? it.old_dest : it.src;
        if (stat(from.c_str(), &st) != 0) { it.result = "source missing: " + from; return; }
        if (it.action != "refile" && (st.st_size != p.size || ST_MTIME(st) != p.mtime)) { it.result = "source changed since scan; rescan"; return; }
        std::string dir = util::dirname(it.dest), err;
        if (!util::mkdirs(dir)) { it.result = "cannot create " + dir; return; }
        bool ok;
        if (it.action == "refile") {
            ok = rename(it.old_dest.c_str(), it.dest.c_str()) == 0;
            if (!ok) err = strerror(errno);
            else if (util::file_exists(sidecar_path(it.old_dest))) rename(sidecar_path(it.old_dest).c_str(), sidecar_path(it.dest).c_str());
        } else if (it.action == "move" || it.action == "rename") {
            ok = rename(it.src.c_str(), it.dest.c_str()) == 0;
            if (!ok && errno == EXDEV) {  // other filesystem: copy, verify, then remove the source
                ok = util::copy_file_preserve(it.src, it.dest, err) && util::files_equal(it.src, it.dest);
                if (ok) unlink(it.src.c_str());
                else if (err.empty()) err = "verification failed";
            } else if (!ok) {
                err = strerror(errno);
            }
        } else {
            ok = util::copy_file_preserve(it.src, it.dest, err);
        }
        if (!ok) { it.result = err; return; }
        journal(it.action, it, from, it.dest, p.dest_path);
        p.dest_path = it.dest;
        if (!parse_serial_name(util::stem(it.dest), c.name_sep, p.day_key, p.sn)) { p.day_key.clear(); p.sn = 0; }
        p.organized_at = util::now_iso();
        p.meta_state.clear();
        db_.save_organize(p);
        if (it.action == "move" || it.action == "rename") {  // the original itself now lives at the new path
            db_.move_src(p.id, it.dest);
            p.src_path = it.dest;
        }
    }
    bool vision_ready = (p.vision_status == "done" || !p.clip_model.empty());
    // Where the metadata goes: the placed file, or (metadata-only) wherever the photo lives now.
    std::string file = it.action == "metadata" && !it.dest.empty() ? it.dest : p.dest_path;
    bool is_original = file == p.src_path;
    if (c.write_mode == WRITE_DB_ONLY) {
        p.meta_state = "db-only";
    } else {
        MetaTarget target = c.write_mode == WRITE_SIDECAR ? MetaTarget::Sidecar : MetaTarget::Embed;
        MetaMap existing = target == MetaTarget::Embed ? read_meta(file) : existing_meta(p, file, target);  // fresh, not the preview's
        MetaPlan plan = meta_plan_for(p, when, c.rewrite_tags, &existing, file, &c, it.desc_action);  // "decide" = left as it is
        backup_before_write(c, p.id, file);
        backup_before_write(c, p.id, sidecar_path(file));
        WriteResult w = write_meta(file, existing, plan, target, false);
        if (!w.ok && target == MetaTarget::Embed) {
            // Format exiv2 cannot write (e.g. some RAW/HEIC): fall back to a sidecar rather than giving up.
            log_.add(1, "embedded write failed (" + w.error + "), using sidecar: " + file);
            w = write_meta(file, existing_meta(p, file, MetaTarget::Sidecar), plan, MetaTarget::Sidecar, false);
        }
        p.write_error = w.ok ? "" : w.error;
        if (!w.ok) log_.add(1, "metadata not written (" + w.error + "): " + file);
        if (w.ok && is_original && target == MetaTarget::Embed) {
            // The original changed (size, hash, metadata): record that, so the next scan does not treat it as a
            // new version and re-decide everything.
            struct stat st{};
            if (stat(file.c_str(), &st) == 0)
                db_.refresh_file(p.id, st.st_size, ST_MTIME(st), util::quick_hash(file, st.st_size), read_meta(file).to_json());
        }
        if (p.dest_path.empty()) {  // not organized: nothing else to record
            it.result = w.ok ? "ok" : "metadata failed: " + w.error;
            return;
        }
        p.meta_state = w.ok ? (vision_ready ? "full" : "base") : "failed";
    }
    db_.save_organize(p);
    it.result = p.meta_state == "failed" ? "placed; metadata failed: " + p.write_error : "ok";
}

void Pipeline::organize(const Config& c, const RunOptions& o) {
    if (c.sources.empty()) {
        log_.add(2, "no output folder (choose a photo folder first)");
        return;
    }
    if (!o.apply_plan) {
        std::vector<PlanItem> fresh = build_plan(c, o);
        {
            std::lock_guard<std::mutex> l(plan_mu_);
            plan_ = fresh;
        }
        int n[5] = {0};
        for (auto& it : fresh) {
            n[it.action == "copy" || it.action == "move" || it.action == "rename" ? 0 : it.action == "refile" ? 1 : it.action == "metadata" ? 2 : 3]++;
            if (it.needs_review && it.action != "duplicate") n[4]++;
        }
        log_.add(0, util::fmt("preview: %d to place, %d to re-file, %d metadata updates, %d duplicates skipped, %d need review",
                              n[0], n[1], n[2], n[3], n[4]));
        if (o.plan_only || c.dry_run) {
            log_.add(0, "preview only: nothing was copied or written");
            return;
        }
    }
    // Execute the stored plan (what the preview showed), included items only, in order. Every change is journaled
    // under this run's id so it can be undone.
    run_id_ = util::now_iso();
    std::vector<PlanItem> work = plan();
    int total = 0;
    for (auto& it : work) total += it.include && it.action != "duplicate";
    begin(ST_ORGANIZE, total);
    int ok = 0;
    for (auto& it : work) {
        if (stop_) break;
        if (!it.include || it.action == "duplicate") continue;
        progress.set_current(it.src);
        apply_item(c, it);
        progress.done++;
        if (it.result == "ok") ok++;
        else {
            progress.errors++;
            log_.add(2, it.action + " " + it.src + ": " + it.result);
        }
    }
    {
        std::lock_guard<std::mutex> l(plan_mu_);
        plan_ = work;  // keep results visible in the preview table
    }
    log_.add(0, util::fmt("applied %d of %d planned changes", ok, total));
    if (ok) last_applied_run = run_id_;
    for (auto& old : db_.journal_prune(10)) { std::error_code ec; fs::remove_all(fs::u8path(undo_dir(c, old)), ec); }  // keep the last 10 runs undoable
    run_id_.clear();
}

std::string undo_dir(const Config& c, const std::string& run) {
    std::string r;
    for (char ch : run) r += isalnum((unsigned char)ch) ? ch : '-';
    return util::dirname(db_path(c)) + "/undo/" + r;
}

void Pipeline::journal(const std::string& kind, const PlanItem& it, const std::string& src, const std::string& dst, const std::string& prev_dest) {
    if (run_id_.empty()) return;
    JournalEntry e;
    e.run = run_id_;
    e.kind = kind;
    e.photo_id = it.id;
    e.src = src;
    e.dst = dst;
    e.prev_dest = prev_dest;
    db_.journal_add(e);
}

// Before metadata is written into a file in place, keep a copy of it (or note that it did not exist yet).
void Pipeline::backup_before_write(const Config& c, int64_t id, const std::string& file) {
    if (run_id_.empty()) return;
    JournalEntry e;
    e.run = run_id_;
    e.kind = "meta";
    e.photo_id = id;
    e.dst = file;
    if (util::file_exists(file)) {
        std::string dir = undo_dir(c, run_id_), err;
        util::mkdirs(dir);
        e.backup = dir + "/" + std::to_string(id) + "_" + util::basename(file);
        for (int k = 2; util::file_exists(e.backup); k++) e.backup = dir + "/" + std::to_string(id) + "_" + std::to_string(k) + "_" + util::basename(file);
        if (!util::copy_file_preserve(file, e.backup, err)) {
            log_.add(1, "no undo backup of " + file + ": " + err);
            return;
        }
    }
    db_.journal_add(e);
}

// Put things back as they were before an applied run, newest change first. Files are only moved back when
// nothing else took their place; an organized copy the run made goes to the Trash.
void Pipeline::undo(const Config& c, const std::string& run) {
    std::vector<JournalEntry> es = db_.journal_entries(run);
    std::reverse(es.begin(), es.end());
    {
        std::lock_guard<std::mutex> l(plan_mu_);
        plan_.clear();  // the list it was applied from no longer describes the files
    }
    begin(ST_ORGANIZE, int(es.size()));
    int ok = 0, failed = 0;
    for (auto& e : es) {
        if (stop_) break;
        progress.done++;
        if (e.undone) continue;
        progress.set_current(e.dst);
        std::string err;
        bool done = false;
        if (e.kind == "meta") {
            if (e.backup.empty()) {  // the write created this file (a sidecar): remove it again
                done = !util::file_exists(e.dst) || unlink(e.dst.c_str()) == 0;
            } else if (util::file_exists(e.backup)) {
                std::string tmp = e.dst + ".jev-undo";  // copy beside it, then swap in (the copy never overwrites)
                unlink(tmp.c_str());
                done = util::copy_file_preserve(e.backup, tmp, err) && rename(tmp.c_str(), e.dst.c_str()) == 0;
                if (!done) unlink(tmp.c_str());
                if (done) {
                    struct stat st{};
                    Photo p;
                    if (db_.load(e.photo_id, p) && p.src_path == e.dst && stat(e.dst.c_str(), &st) == 0)
                        db_.refresh_file(p.id, st.st_size, ST_MTIME(st), util::quick_hash(e.dst, st.st_size), read_meta(e.dst).to_json());
                }
            } else {
                err = "backup missing";
            }
        } else if (e.kind == "copy") {
            done = !util::file_exists(e.dst) || util::trash(e.dst);
            if (!done) err = "could not move the copy to the Trash";
            else {
                if (util::file_exists(sidecar_path(e.dst))) util::trash(sidecar_path(e.dst));
                db_.clear_dest(e.photo_id, e.prev_dest);
            }
        } else {  // move | rename | refile: rename back
            if (util::file_exists(e.src)) err = "something else is at " + e.src + " now";
            else if (!util::file_exists(e.dst)) err = "not found: " + e.dst;
            else {
                util::mkdirs(util::dirname(e.src));
                done = rename(e.dst.c_str(), e.src.c_str()) == 0;
                if (!done && errno == EXDEV) {
                    done = util::copy_file_preserve(e.dst, e.src, err) && util::files_equal(e.dst, e.src);
                    if (done) unlink(e.dst.c_str());
                } else if (!done) {
                    err = strerror(errno);
                }
                if (done && util::file_exists(sidecar_path(e.dst))) rename(sidecar_path(e.dst).c_str(), sidecar_path(e.src).c_str());
                if (done) {
                    if (e.kind != "refile") db_.move_src(e.photo_id, e.src);
                    db_.clear_dest(e.photo_id, e.prev_dest);
                }
            }
        }
        if (done) {
            db_.journal_mark_undone(e.id);
            ok++;
        } else {
            failed++;
            progress.errors++;
            log_.add(2, "undo " + e.kind + " " + e.dst + ": " + err);
        }
    }
    log_.add(failed ? 1 : 0, util::fmt("undo: %d changes put back, %d could not be", ok, failed));
}

std::string name_prefix(const std::string& stem) {
    static const std::regex copy_marker(R"((\s*\(\d+\)|[ _-]copy( \d+)?|\s*副本.*)$)", std::regex::icase);
    // dates with an optional time: 20190512, 2019-05-12, 2019_05_12 123456, 2019-05-12T12.34.56, 20190512-123456789
    static const std::regex date_time(
        R"((^|[^0-9])((19|20)\d\d[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01]))([ _T.-]?([01]\d|2[0-3])[-_.:h]?[0-5]\d([-_.:m]?[0-5]\d)?s?(\d{3})?)?(?=[^0-9]|$))");
    static const std::regex time_only(R"((^|[^0-9])([01]\d|2[0-3])[.:-][0-5]\d([.:-][0-5]\d)?(?=[^0-9]|$))");
    static const std::regex unix_ts(R"((^|[^0-9])1\d{9}(\d{3})?(?=[^0-9]|$))");
    static const std::regex whatsapp(R"([ _-]?WA\d{4}$)", std::regex::icase);
    static const std::regex trailing_sn(R"([ _.-]*\d{2,}[ _.-]*$)");
    static const std::regex seps(R"(([ _.-])[ _.-]+)");
    static const std::regex trailing_word(R"([ _.-]+(at|on|um|de|from|le|の)$)", std::regex::icase);
    std::string s = stem;
    for (int i = 0; i < 3; i++) s = std::regex_replace(s, copy_marker, "");
    s = std::regex_replace(s, date_time, "$1");
    s = std::regex_replace(s, time_only, "$1");
    s = std::regex_replace(s, unix_ts, "$1");
    s = std::regex_replace(s, whatsapp, "");
    for (int i = 0; i < 3; i++) {
        s = std::regex_replace(s, trailing_sn, "");
        s = std::regex_replace(s, seps, "$1");
        s = std::regex_replace(s, trailing_word, "");
        while (!s.empty() && std::string(" _.-").find(s.back()) != std::string::npos) s.pop_back();
        while (!s.empty() && std::string(" _.-").find(s.front()) != std::string::npos) s.erase(0, 1);
    }
    // Nothing but digits and punctuation left: no prefix.
    bool letters = false;
    for (unsigned char ch : s) letters |= ch >= 0x80 || isalpha(ch);
    if (!letters) return "";
    if (s.size() > 60) {
        size_t cut = 60;
        while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) cut--;
        s = s.substr(0, cut);
    }
    return s;
}

bool parse_serial_name(const std::string& stem, const std::string& sep, std::string& day_key, int& sn) {
    size_t end = stem.size(), d = end;
    while (d > 0 && isdigit(static_cast<unsigned char>(stem[d - 1]))) d--;
    if (d == end) return false;
    std::string digits = stem.substr(d);
    size_t day_end;
    if (sep.empty()) {  // 2019051200001: the day is the first 8 of the digit run
        if (digits.size() < 9) return false;
        day_key = digits.substr(0, 8);
        sn = atoi(digits.c_str() + 8);
        return true;
    }
    if (d < sep.size() + 8 || stem.compare(d - sep.size(), sep.size(), sep) != 0) return false;
    day_end = d - sep.size();
    for (size_t k = day_end - 8; k < day_end; k++)
        if (!isdigit(static_cast<unsigned char>(stem[k]))) return false;
    if (day_end > 8 && isdigit(static_cast<unsigned char>(stem[day_end - 9]))) return false;
    day_key = stem.substr(day_end - 8, 8);
    sn = atoi(digits.c_str());
    return true;
}

bool long_name(const std::string& prefix) {
    int words = 1;
    for (char ch : prefix) words += ch == ' ' || ch == '_' || ch == '-';
    return util::utf8_len(prefix) > 30 || words > 4;
}

std::vector<std::string> name_words(const std::string& stem) {
    static const std::set<std::string> drop = {
        "img", "dsc", "dscn", "dscf", "pxl", "mvimg", "vid", "mov", "image", "img", "pic", "picture", "photo", "photos", "file", "copy",
        "final", "edit", "edited", "export", "exported", "untitled", "new", "a", "an", "the", "of", "and", "with", "in", "on", "at",
        "for", "to", "by", "from", "my", "our", "is", "are", "4k", "8k", "hd", "uhd", "fhd", "1080p", "720p", "2160p", "hq", "hdr",
        "masterpiece", "best", "quality", "high", "highres", "resolution", "jpg", "jpeg", "png", "webp", "wa", "ai", "v1", "v2", "v3"};
    std::string base = name_prefix(stem);
    std::vector<std::string> out;
    std::set<std::string> seen;
    std::string cur;
    auto flush = [&] {
        std::string w = util::lower(cur);
        cur.clear();
        if (w.empty() || drop.count(w)) return;
        bool digits = std::all_of(w.begin(), w.end(), [](unsigned char ch) { return isdigit(ch); });
        int letters = 0, nums = 0;
        for (unsigned char ch : w) { letters += isalpha(ch) != 0; nums += isdigit(ch) != 0; }
        bool cjk = false;
        for (unsigned char ch : w) cjk |= ch >= 0x80;
        if (digits || (!cjk && w.size() < 2)) return;
        if (!cjk && nums >= 2 && letters >= 2 && w.size() >= 6) return;  // hashes and ids: KQE7ZGB0M14N0G4E9J6QY3DAQ0
        if (!cjk && w.size() >= 12 && letters == int(w.size()) && w.find_first_of("aeiou") == std::string::npos) return;
        if (seen.insert(w).second) out.push_back(w);
    };
    for (size_t i = 0; i < base.size(); i++) {
        unsigned char ch = base[i];
        if (ch == ' ' || ch == '_' || ch == '-' || ch == '.' || ch == '+' || ch == '(' || ch == ')' || ch == '[' || ch == ']' || ch == ',' || ch == '#')
            flush();
        else cur += char(ch);
    }
    flush();
    return out;
}

bool apply_correction(Db& db, const Correction& c, std::string& err) {
    Photo p;
    if (!db.load(c.photo_id, p)) { err = "photo no longer in the catalog"; return false; }
    if (c.field == "tag") {
        json fix = json::parse(p.tag_fix.empty() ? "{}" : p.tag_fix, nullptr, false);
        if (!fix.is_object()) fix = json::object();
        if (!fix.contains("remove")) fix["remove"] = json::array();
        if (!fix.contains("add")) fix["add"] = json::array();
        std::string t = util::lower(c.action == "remove" ? c.current : c.proposed);
        auto drop_from = [&](json& arr) {
            json keep = json::array();
            for (auto& x : arr) {
                std::string v = x.is_string() ? x.get<std::string>() : x.is_array() && !x.empty() && x[0].is_string() ? x[0].get<std::string>() : "";
                if (util::lower(v) != t) keep.push_back(x);
            }
            arr = keep;
        };
        if (c.action == "remove") {
            drop_from(fix["add"]);
            fix["remove"].push_back(t);
        } else {
            drop_from(fix["remove"]);
            drop_from(fix["add"]);
            fix["add"].push_back({t, std::round(c.confidence * 100) / 100});
        }
        db.save_tag_fix(p.id, fix.dump());
        return true;
    }
    if (c.field == "name") {
        // Only files jev-photos named: the organized copy (or the original it renamed in place).
        if (p.dest_path.empty() || util::basename(p.dest_path) != c.current) { err = "the file is not named as expected any more"; return false; }
        std::string to = util::dirname(p.dest_path) + "/" + c.proposed;
        if (util::file_exists(to) || db.dest_taken(to)) { err = "a file named " + c.proposed + " already exists"; return false; }
        if (rename(p.dest_path.c_str(), to.c_str()) != 0) { err = strerror(errno); return false; }
        if (util::file_exists(sidecar_path(p.dest_path))) rename(sidecar_path(p.dest_path).c_str(), sidecar_path(to).c_str());
        bool in_place = p.src_path == p.dest_path;
        p.dest_path = to;
        db.save_organize(p);
        if (in_place) db.move_src(p.id, to);
        return true;
    }
    err = "unknown correction";
    return false;
}
