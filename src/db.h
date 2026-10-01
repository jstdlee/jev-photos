// SQLite catalogue: one row per source image, per-folder location decisions, FTS5 search index.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <string>
#include <vector>

struct sqlite3;

struct Photo {
    int64_t id = 0;
    std::string src_path, src_root, quick_hash, content_hash, ext;  // XXH3-64 of size+head+tail; XXH3-128 of the file (lazy)
    int64_t size = 0, mtime = 0, btime = 0;
    int width = 0, height = 0;
    std::string meta_json, meta_error;

    std::string date_value, date_prec, date_source, date_decider, date_evidence, manual_date;
    double date_conf = 0;
    bool needs_review = true, resolved = false;

    bool has_gps = false;
    double gps_lat = 0, gps_lon = 0;
    std::string location, location_source;
    double location_conf = 0;

    std::string vision_status, caption, scene, objects, tags, landmark, vision_text, vision_model, vision_error;  // objects/tags: JSON arrays
    int people = 0;

    std::string dest_path, day_key, organized_at, meta_state, write_error;
    int sn = 0;
    int64_t dup_of = 0;       // exact duplicate of this photo id (0 = unique or the kept copy)
    int64_t similar_to = 0;   // near-duplicate (resized/re-saved) of this photo id, report only
    int similar_dist = 0;     // dHash Hamming distance to similar_to
    std::string clip_model, clip_tags, clip_scene;  // clip_tags: JSON [[tag, confidence], ...]
    std::string clip_vocab;   // tag vocabulary hash the clip_tags came from (stale when the vocabulary changes)
    std::string user_tags;    // JSON ["tag", ...] when the user edited the tags ("" = automatic)
    // AI generation metadata found in the file (genmeta.h): GenInfo JSON, the tool, and keywords from the prompt
    std::string gen_json, gen_tool, gen_keywords;  // gen_keywords: JSON ["...", ...] (tag-style prompt or LLM-extracted)
    std::string desc_choice;  // JSON {key, action, by}: how an existing description is handled (decided once per text pair)
    std::string tag_fix;      // JSON {"remove": [...], "add": [[tag, conf], ...]}: accepted corrections of the automatic tags
    std::string name_keywords;  // JSON ["...", ...]: the words of the original file name that mean something
    bool favorite = false;
    std::string fix_checked;  // what the last correction check saw (inputs hash): unchanged photos are not re-checked
};

// CLIP tags below this confidence are guesses: shown in the detail pane, but not searchable and not written.
constexpr float kTagWordMin = 0.25f;
// [[tag, conf], ...] or ["tag", ...] (conf 1) -> list
std::vector<std::pair<std::string, float>> parse_tag_list(const std::string& json);
// The tags a photo carries: the user's own list when they edited it, else vision-model tags + confident CLIP tags,
// with accepted corrections (tag_fix) applied.
std::vector<std::string> effective_tags(const Photo& p);
std::vector<std::pair<std::string, float>> effective_tag_list(const Photo& p);  // same, with confidences (user/fix = 1)

// A proposed correction of something jev-photos generated itself (never of the file's own metadata).
struct Correction {
    int64_t id = 0, photo_id = 0;
    std::string field;      // tag | scene | name
    std::string action;     // remove | add | rename
    std::string current, proposed;
    double confidence = 0;  // how sure we are the change is right and worth making, 0..1
    std::string reason, source;  // why; which evidence (prompt, camera, file name, CLIP, LLM)
    std::string status;     // pending | applied | rejected
    std::string path;       // photo path (for display)
};

// One jev decision, kept so the user can see what jev was asked, what it chose, and whether that changed anything.
struct Decision {
    int64_t id = 0, photo_id = 0;
    std::string at, kind;        // kind: date | place | search
    std::string subject;         // file, folder or search text
    std::string question;
    std::string options;         // JSON [[text, jev probability], ...]
    std::string rules_pick, jev_pick, final_pick;
    bool changed = false;        // jev's input changed the outcome
};
struct DecisionStats {
    int total = 0, changed = 0;
    std::map<std::string, std::pair<int, int>> by_kind;  // kind -> (asked, changed)
};

// One row of the Tags & EXIF tab.
struct TagRow {
    int64_t id = 0;
    std::string src_path, dest_path, date_value, date_prec, location;
    std::vector<std::pair<std::string, float>> tags;  // effective (user tags have conf 1)
    bool user_edited = false, tagged = false, stale = false;  // stale: tagged with another model/vocabulary
    float best = 0;                                           // best tag confidence
    bool exif_date = false, keywords = false, description = false, place = false;  // what the file itself carries
    bool favorite = false;
    std::vector<std::string> name_words;
    std::string gen_tool;                                                             // AI generator, "" = none found
    std::vector<std::string> prompt_tags;                                             // keywords from its prompt
    int64_t size = 0;
};

struct PhotoRow {  // what the GUI table shows
    int64_t id = 0;
    std::string date_value, date_prec, date_source, decider, location, scene, tags, dest_path, src_path, vision_status;
    double date_conf = 0;
    bool needs_review = false;
    int64_t dup_of = 0, similar_to = 0, size = 0;
    int width = 0, height = 0;
    bool favorite = false;
    double score = 0;   // search relevance (0 when not searching)
    std::string why;    // what matched
};

// Everything the search needs about one photo (kept in memory by the searcher).
struct SearchDoc {
    int64_t id = 0;
    int year = 0;                     // 0 = undated
    std::string date, prec;           // date_value / date_prec, for date-range filters
    std::string name, exif, desc;     // lower-cased search text per field
    std::string name_raw, desc_raw;   // original case, for regex and "why"
    std::vector<float> vec;           // CLIP image embedding (may be empty)
    std::string clip_model;
    std::vector<std::pair<std::string, float>> clip_tags;  // tag, confidence (all of them, also the weak ones)
    std::string prompt, prompt_raw;  // AI generation prompt (+ negative), lower-cased / original
    std::vector<std::string> tags;   // effective tags, lower-cased (for tag: filters and favorite-tag chips)
    bool favorite = false;
    // for filters: size:, width:, ext:, place:, has:gps, is:uncertain
    int64_t size = 0;
    int width = 0, height = 0;
    std::string ext, location;
    bool has_gps = false, uncertain = false;
};

struct DupRow {  // everything the duplicate scan needs, without the heavy JSON columns
    int64_t id = 0, size = 0, mtime = 0, dup_of = 0, similar_to = 0;
    std::string src_path, src_root, dest_path, quick_hash, content_hash, date_value, meta_json, ext;
    bool keep = false, dhash_done = false;
    uint64_t dhash = 0;
    int width = 0, height = 0, meta_len = 0, similar_dist = 0;
};

struct MonthInfo {
    std::string month;  // "2019-05" or "undated"
    int count = 0;
    int64_t bytes = 0;
};

struct DbStats {
    int total = 0, resolved = 0, review = 0, undated = 0, vision_done = 0, vision_failed = 0, organized = 0, dups = 0, located = 0;
    int similar = 0, tagged = 0;
    int64_t total_bytes = 0, dup_bytes = 0;
};

struct DirPlace {
    bool found = false;
    std::string location, source, evidence;
    double conf = 0;
};

class Db {
public:
    ~Db();
    bool open(const std::string& path, std::string& err);
    void close();
    bool is_open() const { return db_ != nullptr; }
    const std::string& path() const { return path_; }

    // scan
    bool known_unchanged(const std::string& src_path, int64_t size, int64_t mtime);
    int64_t upsert_scan(const Photo& p);  // resets downstream results when the file changed
    std::vector<int64_t> ids(const std::string& where = "1");
    bool load(int64_t id, Photo& p);

    // results
    void save_date(const Photo& p);
    void save_location(int64_t id, const std::string& loc, const std::string& source, double conf);
    void save_vision(const Photo& p);
    void save_organize(const Photo& p);
    void set_manual_date(int64_t id, const std::string& value_with_prec);  // "YYYY-MM-DD HH:MM:SS|second" or "" to clear
    int next_sn(const std::string& day_key, const std::string& out_root);  // serials are per output folder
    // duplicate scan
    std::vector<DupRow> dup_rows(bool with_meta = false);
    void set_content_hash(int64_t id, const std::string& h);
    void set_dhash(int64_t id, uint64_t h, bool ok);
    // Replace dup_of / similar_to for the rows in `scope_ids` (the folder that was checked).
    void apply_dups(const std::vector<int64_t>& scope_ids, const std::vector<std::pair<int64_t, int64_t>>& id_dup_of);
    void apply_similar(const std::vector<int64_t>& scope_ids, const std::vector<std::tuple<int64_t, int64_t, int>>& id_to_dist);
    void set_keeper(const std::vector<int64_t>& group, int64_t keep);
    void remove_photo(int64_t id);                           // the file is gone (e.g. moved to the Trash)
    void move_src(int64_t id, const std::string& new_path);  // the file itself was moved/renamed
    // We changed the file ourselves (metadata added in place): new size/time/hash/snapshot, results stay valid.
    void refresh_file(int64_t id, int64_t size, int64_t mtime, const std::string& quick_hash, const std::string& meta_json);
    // CLIP
    void save_clip(int64_t id, const std::string& model, const std::vector<float>& vec, const std::string& tags_json, const std::string& scene,
                   const std::string& vocab);
    void save_tags(int64_t id, const std::string& tags_json, const std::string& scene, const std::string& vocab);  // re-tag, same vector
    bool clip_vec(int64_t id, std::vector<float>& out);
    void set_user_tags(int64_t id, const std::string& tags_json);  // "" = back to automatic
    // AI generation metadata
    void save_gen(int64_t id, const std::string& gen_json, const std::string& tool, const std::string& keywords_json);
    void save_gen_keywords(int64_t id, const std::string& keywords_json);
    void save_desc_choice(int64_t id, const std::string& choice_json);
    // corrections, favorites, file-name keywords, translations
    void save_name_keywords(int64_t id, const std::string& json);
    void set_favorite(int64_t id, bool on);
    void replace_corrections(const std::vector<int64_t>& photo_ids, const std::vector<Correction>& pending);  // keeps decided ones
    std::vector<Correction> corrections(const std::string& folder, bool pending_only);
    void set_correction_status(int64_t id, const std::string& status);
    bool correction_rejected(int64_t photo_id, const std::string& field, const std::string& current, const std::string& proposed);
    void save_tag_fix(int64_t id, const std::string& json);
    void set_fix_checked(int64_t id, const std::string& key);
    std::string translation(const std::string& term);  // cached JSON list of equivalents, "" = not cached
    void save_translation(const std::string& term, const std::string& json);
    std::vector<TagRow> tag_rows(const std::string& folder, const std::string& model_id, const std::string& vocab);
    // jev decisions
    void add_decision(const Decision& d);
    std::vector<Decision> decisions(const std::string& folder, int limit);
    DecisionStats decision_stats(const std::string& folder);
    std::vector<SearchDoc> search_docs(const std::string& folder);
    bool dest_taken(const std::string& dest);

    DirPlace get_dir(const std::string& path);
    void put_dir(const std::string& path, const DirPlace& d);
    void clear_dirs();

    // `folder` limits results to photos under that folder ("" = everything in the catalog).
    std::vector<PhotoRow> query(const std::string& search, bool review_only, const std::string& month, int limit, const std::string& folder = "");
    std::vector<MonthInfo> months(const std::string& folder = "");
    DbStats stats(const std::string& folder = "");
    void reindex(int64_t id);

private:
    bool exec(const char* sql, std::string* err = nullptr);
    sqlite3* db_ = nullptr;
    std::string path_;
    std::recursive_mutex mu_;
};
