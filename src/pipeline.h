// The four resumable stages: scan -> decide (date + location) -> vision -> organize (+ metadata write).
#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "db.h"
#include "meta.h"

enum Stage { ST_SCAN = 1, ST_DECIDE = 2, ST_VISION = 4, ST_ORGANIZE = 8, ST_DUPES = 16, ST_TAG = 32, ST_FIX = 64, ST_ALL = 127 };
const char* stage_name(int st);

struct LogLine {
    int level;  // 0 info, 1 warn, 2 error
    std::string time, text;
};

class Log {
public:
    void add(int level, const std::string& text);
    std::vector<LogLine> tail(size_t n);
    size_t size();
    bool echo = false;  // CLI mode prints to stderr

private:
    std::mutex mu_;
    std::deque<LogLine> lines_;
};

struct Progress {
    std::atomic<int> stage{0}, done{0}, total{0}, errors{0}, skipped{0};
    std::atomic<bool> running{false};
    std::atomic<int> finished_runs{0};  // bumped when a run ends, so a UI can react even to very short runs
    std::chrono::steady_clock::time_point stage_start;
    std::mutex mu;
    std::string current, summary;
    void set_current(const std::string& s) { std::lock_guard<std::mutex> l(mu); current = s; }
    std::string get_current() { std::lock_guard<std::mutex> l(mu); return current; }
};

struct RunOptions {
    int stages = ST_ALL;
    bool redecide_all = false;  // re-run decisions for already resolved photos
    bool retry_failed_vision = false;
    std::string scope;          // only this folder (and below); empty = all configured sources
    bool plan_only = false;     // organize stage builds the preview plan and changes nothing
    bool apply_plan = false;    // organize stage executes the stored plan (included items only)
    bool retag_all = false;     // tag stage recomputes every photo's tags (e.g. after tuning the vocabulary)
    bool recheck_all = false;   // correction check looks at every photo again, not only changed ones
    // Image analysis (CLIP): 0 missing ones, 1 all photos again, 2 quick model (ViT-B/32) for the missing ones, 3 skip
    int clip_mode = 0;
    // Tags from the stored analysis: 0 missing or outdated, 1 all, 2 photos with fewer than 2 confident tags
    int tag_mode = 0;
    // The CLIP tab's smart update: exactly these photos are re-analysed / re-tagged (when either is set).
    std::vector<int64_t> reanalyse_ids, retag_ids;
    std::string undo_run;       // undo this applied run (journal) instead of running stages
};

// One planned file-system/metadata change, shown in the preview before anything is touched.
struct PlanItem {
    int64_t id = 0;
    std::string action;         // copy | move | rename | refile | metadata | duplicate
    int64_t size = 0;           // bytes (for totals and time estimates)
    std::string src, dest, old_dest, date, location, note;
    double conf = 0;
    bool needs_review = false;
    bool include = true;        // user can exclude items before applying
    std::string meta_target;    // file that receives metadata (image or .xmp), empty = none
    std::vector<std::string> meta_cmds;  // exiv2 commands that would run (additions only)
    int meta_added = 0;
    std::vector<std::string> meta_removed;  // keywords jev-photos wrote earlier that the rewrite takes out
    // Description: the file's text, ours, and what happens. action: add | keep | append | replace | decide (you choose)
    std::string desc_old, desc_new, desc_action, desc_by, desc_why;
    std::string result;         // after apply: "ok" or the error
};

class Pipeline {
public:
    Pipeline(Db& db, Log& log) : db_(db), log_(log) {}
    ~Pipeline() { join(); }
    bool start(const Config& cfg, const RunOptions& opt);  // false if already running
    void run_blocking(const Config& cfg, const RunOptions& opt);
    void stop() { stop_ = true; }
    bool running() const { return progress.running; }
    void join() { if (th_.joinable()) th_.join(); }
    Progress progress;

    std::vector<PlanItem> plan();                   // copy of the current plan
    void set_include(int64_t id, bool include);     // toggle an item before applying
    void set_include_where(bool (*pred)(const PlanItem&), bool include);
    void set_desc_action(int64_t id, const std::string& action);  // your choice for a description (kept in the DB too)
    bool has_plan();

private:
    void run(Config cfg, RunOptions opt);
    void scan(const Config& c, const RunOptions& o);
    void dedupe(const Config& c, const RunOptions& o);
    void decide(const Config& c, const RunOptions& o);
    void vision(const Config& c, const RunOptions& o);
    void tag(const Config& c, const RunOptions& o);
    void prompt_keywords_llm(const Config& c, const RunOptions& o);
    void check_corrections(const Config& c, const RunOptions& o);  // proposals only; applied from Review
    void organize(const Config& c, const RunOptions& o);
    std::vector<PlanItem> build_plan(const Config& c, const RunOptions& o);
    void decide_description(const Config& c, Photo& p, PlanItem& it);
    void apply_item(const Config& c, PlanItem& it);
    void journal(const std::string& kind, const PlanItem& it, const std::string& src, const std::string& dst, const std::string& prev_dest);
    void backup_before_write(const Config& c, int64_t id, const std::string& file);
    void undo(const Config& c, const std::string& run);
    std::vector<int64_t> scoped(const std::string& where, const RunOptions& o);
    void begin(int stage, int total);
    template <class F> void parallel(int n, int threads, F fn);

    Db& db_;
    Log& log_;
    std::thread th_;
    std::atomic<bool> stop_{false};
    bool jev_down_ = false;
    std::mutex plan_mu_;
    std::vector<PlanItem> plan_;
    std::string run_id_;  // the apply being journaled
public:
    std::string last_applied_run;  // the run id of the last apply that changed something (for Undo)
};

// Where a run's backups of files written in place are kept.
std::string undo_dir(const Config& c, const std::string& run);

// The description jev-photos writes: the generation prompt (AI images), the caption, and "Shows: <tags>".
std::string our_description(const Config& c, const Photo& p);
// Is this description text a camera/software default or placeholder rather than a description?
bool placeholder_description(const std::string& text, const std::string& file_stem);
// Does `have` already say what `ours` says (every word of ours appears in it)?
bool description_covers(const std::string& have, const std::string& ours);

// What adding metadata to this photo would do now (dry run; reads the organized copy if there is one).
WriteResult preview_metadata(const Config& c, const Photo& p, std::string& file);

// File naming. name_prefix: the original name without dates, times, serial numbers and copy markers
// ("IMG_20190512_123456" -> "IMG", "Paris trip 2019-05-02 (3)" -> "Paris trip", "20190512_00001" -> "").
std::string name_prefix(const std::string& stem);
// "…20190512_00001" -> day "20190512", sn 1 (the part of an organized name after any prefix).
bool parse_serial_name(const std::string& stem, const std::string& sep, std::string& day_key, int& sn);
// The meaningful words of a file name: no dates, serials, hashes, camera prefixes or filler words.
std::vector<std::string> name_words(const std::string& stem);
// A name prefix this long is better replaced by its key words (decided once, by the LLM when available).
bool long_name(const std::string& prefix);
// Apply one accepted correction: tag changes go into the photo's tag_fix, a rename renames the file jev-photos
// itself named (and its sidecar). err set on failure.
bool apply_correction(Db& db, const Correction& c, std::string& err);

// True when `path` is `dir` or inside it.
bool path_under(const std::string& path, const std::string& dir);

// Folder names between the source root (inclusive) and the file, nearest first.
std::vector<std::string> rel_dirs(const std::string& src_root, const std::string& file);
