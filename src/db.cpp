#include "db.h"

#include <sqlite3.h>

#include <set>

#include "genmeta.h"
#include "nlohmann/json.hpp"
#include "util.h"

using json = nlohmann::json;
using Lock = std::lock_guard<std::recursive_mutex>;

namespace {

const char* kSchema = R"SQL(
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY,
  src_path TEXT UNIQUE NOT NULL, src_root TEXT, size INTEGER, mtime INTEGER, btime INTEGER, ext TEXT,
  width INTEGER, height INTEGER, meta_json TEXT, meta_error TEXT, scanned_at TEXT,
  date_value TEXT, date_prec TEXT, date_source TEXT, date_conf REAL, date_decider TEXT, date_evidence TEXT,
  needs_review INTEGER DEFAULT 1, resolved INTEGER DEFAULT 0, manual_date TEXT,
  has_gps INTEGER DEFAULT 0, gps_lat REAL, gps_lon REAL, location TEXT, location_source TEXT, location_conf REAL,
  vision_status TEXT, caption TEXT, scene TEXT, objects TEXT, tags TEXT, landmark TEXT, vision_text TEXT, people INTEGER,
  vision_model TEXT, vision_error TEXT,
  dest_path TEXT UNIQUE, day_key TEXT, sn INTEGER, organized_at TEXT, meta_state TEXT, write_error TEXT, dup_of INTEGER,
  search_text TEXT
);
CREATE INDEX IF NOT EXISTS photos_date ON photos(date_value);
CREATE INDEX IF NOT EXISTS photos_day ON photos(day_key, sn);
CREATE TABLE IF NOT EXISTS dirs(path TEXT PRIMARY KEY, location TEXT, location_conf REAL, location_source TEXT, evidence TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, at TEXT, kind TEXT, photo_id INTEGER, subject TEXT, question TEXT,
  options TEXT, rules_pick TEXT, jev_pick TEXT, final_pick TEXT, changed INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS decisions_subject ON decisions(subject);
CREATE TABLE IF NOT EXISTS corrections(id INTEGER PRIMARY KEY, photo_id INTEGER, field TEXT, action TEXT, current TEXT, proposed TEXT,
  confidence REAL, reason TEXT, source TEXT, status TEXT DEFAULT 'pending', at TEXT);
CREATE INDEX IF NOT EXISTS corrections_photo ON corrections(photo_id);
CREATE TABLE IF NOT EXISTS translations(term TEXT PRIMARY KEY, alts TEXT, at TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(text, tokenize='trigram');
INSERT OR IGNORE INTO kv VALUES('schema', '1');
)SQL";

struct Stmt {
    sqlite3_stmt* s = nullptr;
    int idx = 0;
    Stmt(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK) {
            fprintf(stderr, "sqlite prepare: %s\n  %s\n", sqlite3_errmsg(db), sql);
            s = nullptr;
        }
    }
    ~Stmt() { sqlite3_finalize(s); }
    Stmt& b(const std::string& v) { sqlite3_bind_text(s, ++idx, v.c_str(), int(v.size()), SQLITE_TRANSIENT); return *this; }
    Stmt& b(const char* v) { return b(std::string(v)); }
    Stmt& b(int64_t v) { sqlite3_bind_int64(s, ++idx, v); return *this; }
    Stmt& b(int v) { sqlite3_bind_int64(s, ++idx, v); return *this; }
    Stmt& b(bool v) { sqlite3_bind_int64(s, ++idx, v ? 1 : 0); return *this; }
    Stmt& b(double v) { sqlite3_bind_double(s, ++idx, v); return *this; }
    Stmt& null() { sqlite3_bind_null(s, ++idx); return *this; }
    Stmt& blob(const void* p, size_t n) { sqlite3_bind_blob(s, ++idx, p, int(n), SQLITE_TRANSIENT); return *this; }
    bool step() { return s && sqlite3_step(s) == SQLITE_ROW; }
    bool run() {
        if (!s) return false;
        int rc = sqlite3_step(s);
        return rc == SQLITE_DONE || rc == SQLITE_ROW;
    }
    std::string t(int c) { auto p = sqlite3_column_text(s, c); return p ? reinterpret_cast<const char*>(p) : ""; }
    int64_t i(int c) { return sqlite3_column_int64(s, c); }
    double d(int c) { return sqlite3_column_double(s, c); }
};

std::string join_json_list(const std::string& j) {
    json a = json::parse(j.empty() ? "[]" : j, nullptr, false);
    std::string out;
    if (a.is_array())
        for (auto& x : a)
            if (x.is_string()) out += (out.empty() ? "" : ", ") + x.get<std::string>();
    return out;
}

}  // namespace

std::vector<std::pair<std::string, float>> parse_tag_list(const std::string& j) {
    std::vector<std::pair<std::string, float>> out;
    json a = json::parse(j.empty() ? "[]" : j, nullptr, false);
    if (!a.is_array()) return out;
    for (auto& t : a) {
        if (t.is_string() && !t.get<std::string>().empty()) out.push_back({t.get<std::string>(), 1.0f});
        else if (t.is_array() && !t.empty() && t[0].is_string()) out.push_back({t[0].get<std::string>(), t.size() > 1 && t[1].is_number() ? t[1].get<float>() : 1.0f});
    }
    return out;
}

std::vector<std::pair<std::string, float>> effective_tag_list(const Photo& p) {
    std::vector<std::pair<std::string, float>> out;
    std::set<std::string> seen;
    auto add = [&](const std::string& t, float c) {
        std::string k = util::lower(util::trim(t));
        if (!k.empty() && seen.insert(k).second) out.push_back({util::trim(t), c});
    };
    if (!p.user_tags.empty()) {
        for (auto& [t, c] : parse_tag_list(p.user_tags)) add(t, 1.0f);
        return out;
    }
    std::set<std::string> removed;
    json fix = json::parse(p.tag_fix.empty() ? "{}" : p.tag_fix, nullptr, false);
    if (fix.is_object() && fix.contains("remove") && fix["remove"].is_array())
        for (auto& r : fix["remove"])
            if (r.is_string()) removed.insert(util::lower(r.get<std::string>()));
    for (auto& [t, c] : parse_tag_list(p.tags))
        if (!removed.count(util::lower(t))) add(t, c);
    for (auto& [t, c] : parse_tag_list(p.clip_tags))
        if (c >= kTagWordMin && !removed.count(util::lower(t))) add(t, c);
    if (fix.is_object() && fix.contains("add"))
        for (auto& [t, c] : parse_tag_list(fix["add"].dump())) add(t, c);
    return out;
}

std::vector<std::string> effective_tags(const Photo& p) {
    std::vector<std::string> out;
    for (auto& [t, c] : effective_tag_list(p)) out.push_back(t);
    return out;
}

Db::~Db() { close(); }

bool Db::exec(const char* sql, std::string* err) {
    char* e = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &e);
    if (rc != SQLITE_OK) {
        if (err) *err = e ? e : "sqlite error";
        else fprintf(stderr, "sqlite: %s\n", e ? e : "?");
        sqlite3_free(e);
        return false;
    }
    return true;
}

bool Db::open(const std::string& path, std::string& err) {
    Lock l(mu_);
    close();
    if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK) {
        err = db_ ? sqlite3_errmsg(db_) : "cannot open database";
        close();
        return false;
    }
    sqlite3_busy_timeout(db_, 5000);
    if (!exec(kSchema, &err)) {
        close();
        return false;
    }
    // Additive migrations: older libraries gain the new columns in place.
    {
        std::set<std::string> have;
        Stmt t(db_, "PRAGMA table_info(photos)");
        while (t.step()) have.insert(t.t(1));
        const char* cols[][2] = {{"quick_hash", "TEXT"}, {"content_hash", "TEXT"}, {"dhash", "INTEGER DEFAULT 0"},
                                 {"dhash_done", "INTEGER DEFAULT 0"}, {"similar_to", "INTEGER DEFAULT 0"},
                                 {"similar_dist", "INTEGER DEFAULT 0"}, {"dup_keep", "INTEGER DEFAULT 0"},
                                 {"clip_model", "TEXT"}, {"clip_vec", "BLOB"}, {"clip_tags", "TEXT"}, {"clip_scene", "TEXT"},
                                 {"clip_vocab", "TEXT"}, {"user_tags", "TEXT"}, {"gen_json", "TEXT"}, {"gen_tool", "TEXT"},
                                 {"gen_keywords", "TEXT"}, {"gen_checked", "INTEGER DEFAULT 0"}, {"desc_choice", "TEXT"},
                                 {"tag_fix", "TEXT"}, {"name_keywords", "TEXT"}, {"favorite", "INTEGER DEFAULT 0"},
                                 {"fix_checked", "TEXT"}};
        for (auto& c : cols)
            if (!have.count(c[0])) exec((std::string("ALTER TABLE photos ADD COLUMN ") + c[0] + " " + c[1]).c_str());
        exec("CREATE INDEX IF NOT EXISTS photos_quick ON photos(size, quick_hash)");
        exec("CREATE INDEX IF NOT EXISTS photos_content ON photos(content_hash)");
    }
    path_ = path;
    return true;
}

void Db::close() {
    Lock l(mu_);
    if (db_) sqlite3_close_v2(db_);
    db_ = nullptr;
    path_.clear();
}

bool Db::known_unchanged(const std::string& src, int64_t size, int64_t mtime) {
    Lock l(mu_);
    Stmt s(db_, "SELECT 1 FROM photos WHERE src_path=? AND size=? AND mtime=?");
    s.b(src).b(size).b(mtime);
    return s.step();
}

int64_t Db::upsert_scan(const Photo& p) {
    Lock l(mu_);
    // A changed file loses its full hash and perceptual hash (recomputed lazily by the duplicate scan).
    Stmt s(db_, R"(INSERT INTO photos(src_path, src_root, size, mtime, btime, quick_hash, ext, width, height, meta_json, meta_error, scanned_at,
                                     gen_json, gen_tool, gen_keywords, gen_checked, name_keywords)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)
                   ON CONFLICT(src_path) DO UPDATE SET src_root=excluded.src_root, size=excluded.size, mtime=excluded.mtime,
                     btime=excluded.btime, quick_hash=excluded.quick_hash, ext=excluded.ext, width=excluded.width, height=excluded.height,
                     meta_json=excluded.meta_json, meta_error=excluded.meta_error, scanned_at=excluded.scanned_at,
                     gen_json=excluded.gen_json, gen_tool=excluded.gen_tool,
                     gen_keywords=CASE WHEN COALESCE(photos.gen_json,'')=COALESCE(excluded.gen_json,'') AND excluded.gen_keywords=''
                                       THEN photos.gen_keywords ELSE excluded.gen_keywords END, gen_checked=1,
                     name_keywords=CASE WHEN photos.src_path=excluded.src_path AND photos.name_keywords LIKE '%1.0]%'
                                        THEN photos.name_keywords ELSE excluded.name_keywords END,
                     resolved=0, content_hash=NULL, dhash=0, dhash_done=0,
                     vision_status=CASE WHEN photos.quick_hash=excluded.quick_hash AND photos.size=excluded.size
                                        THEN photos.vision_status ELSE NULL END)");
    s.b(p.src_path).b(p.src_root).b(p.size).b(p.mtime).b(p.btime).b(p.quick_hash).b(p.ext).b(p.width).b(p.height)
        .b(p.meta_json).b(p.meta_error).b(util::now_iso()).b(p.gen_json).b(p.gen_tool).b(p.gen_keywords).b(p.name_keywords);
    s.run();
    Stmt q(db_, "SELECT id FROM photos WHERE src_path=?");
    q.b(p.src_path);
    return q.step() ? q.i(0) : 0;
}

std::vector<int64_t> Db::ids(const std::string& where) {
    Lock l(mu_);
    std::vector<int64_t> out;
    Stmt s(db_, ("SELECT id FROM photos WHERE " + where + " ORDER BY id").c_str());
    while (s.step()) out.push_back(s.i(0));
    return out;
}

bool Db::load(int64_t id, Photo& p) {
    Lock l(mu_);
    Stmt s(db_, R"(SELECT id, src_path, src_root, COALESCE(quick_hash,''), ext, size, mtime, btime, width, height, meta_json, meta_error,
                   date_value, date_prec, date_source, date_decider, date_evidence, manual_date, date_conf, needs_review, resolved,
                   has_gps, gps_lat, gps_lon, location, location_source, location_conf,
                   vision_status, caption, scene, objects, tags, landmark, vision_text, vision_model, vision_error, people,
                   dest_path, day_key, organized_at, meta_state, write_error, sn, dup_of, COALESCE(content_hash,''),
                   COALESCE(similar_to,0), COALESCE(similar_dist,0), COALESCE(clip_model,''), COALESCE(clip_tags,''),
                   COALESCE(clip_scene,''), COALESCE(clip_vocab,''), COALESCE(user_tags,''), COALESCE(gen_json,''), COALESCE(gen_tool,''),
                   COALESCE(gen_keywords,''), COALESCE(desc_choice,''), COALESCE(tag_fix,''), COALESCE(name_keywords,''),
                   COALESCE(favorite,0), COALESCE(fix_checked,'') FROM photos WHERE id=?)");
    s.b(id);
    if (!s.step()) return false;
    int c = 0;
    p.id = s.i(c++); p.src_path = s.t(c++); p.src_root = s.t(c++); p.quick_hash = s.t(c++); p.ext = s.t(c++);
    p.size = s.i(c++); p.mtime = s.i(c++); p.btime = s.i(c++); p.width = int(s.i(c++)); p.height = int(s.i(c++));
    p.meta_json = s.t(c++); p.meta_error = s.t(c++);
    p.date_value = s.t(c++); p.date_prec = s.t(c++); p.date_source = s.t(c++); p.date_decider = s.t(c++); p.date_evidence = s.t(c++);
    p.manual_date = s.t(c++); p.date_conf = s.d(c++); p.needs_review = s.i(c++) != 0; p.resolved = s.i(c++) != 0;
    p.has_gps = s.i(c++) != 0; p.gps_lat = s.d(c++); p.gps_lon = s.d(c++);
    p.location = s.t(c++); p.location_source = s.t(c++); p.location_conf = s.d(c++);
    p.vision_status = s.t(c++); p.caption = s.t(c++); p.scene = s.t(c++); p.objects = s.t(c++); p.tags = s.t(c++);
    p.landmark = s.t(c++); p.vision_text = s.t(c++); p.vision_model = s.t(c++); p.vision_error = s.t(c++); p.people = int(s.i(c++));
    p.dest_path = s.t(c++); p.day_key = s.t(c++); p.organized_at = s.t(c++); p.meta_state = s.t(c++); p.write_error = s.t(c++);
    p.sn = int(s.i(c++)); p.dup_of = s.i(c++);
    p.content_hash = s.t(c++); p.similar_to = s.i(c++); p.similar_dist = int(s.i(c++));
    p.clip_model = s.t(c++); p.clip_tags = s.t(c++); p.clip_scene = s.t(c++); p.clip_vocab = s.t(c++); p.user_tags = s.t(c++);
    p.gen_json = s.t(c++); p.gen_tool = s.t(c++); p.gen_keywords = s.t(c++); p.desc_choice = s.t(c++);
    p.tag_fix = s.t(c++); p.name_keywords = s.t(c++); p.favorite = s.i(c++) != 0; p.fix_checked = s.t(c++);
    return true;
}

void Db::save_date(const Photo& p) {
    Lock l(mu_);
    Stmt s(db_, R"(UPDATE photos SET date_value=?, date_prec=?, date_source=?, date_conf=?, date_decider=?, date_evidence=?,
                   needs_review=?, resolved=1, has_gps=?, gps_lat=?, gps_lon=? WHERE id=?)");
    s.b(p.date_value).b(p.date_prec).b(p.date_source).b(p.date_conf).b(p.date_decider).b(p.date_evidence).b(p.needs_review)
        .b(p.has_gps).b(p.gps_lat).b(p.gps_lon).b(p.id);
    s.run();
    reindex(p.id);
}

void Db::save_location(int64_t id, const std::string& loc, const std::string& source, double conf) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET location=?, location_source=?, location_conf=? WHERE id=?");
    s.b(loc).b(source).b(conf).b(id);
    s.run();
    reindex(id);
}

void Db::save_vision(const Photo& p) {
    Lock l(mu_);
    Stmt s(db_, R"(UPDATE photos SET vision_status=?, caption=?, scene=?, objects=?, tags=?, landmark=?, vision_text=?, people=?,
                   vision_model=?, vision_error=? WHERE id=?)");
    s.b(p.vision_status).b(p.caption).b(p.scene).b(p.objects).b(p.tags).b(p.landmark).b(p.vision_text).b(p.people)
        .b(p.vision_model).b(p.vision_error).b(p.id);
    s.run();
    reindex(p.id);
}

void Db::save_organize(const Photo& p) {
    Lock l(mu_);
    Stmt s(db_, R"(UPDATE photos SET dest_path=?, day_key=?, sn=?, organized_at=?, meta_state=?, write_error=? WHERE id=?)");
    if (p.dest_path.empty()) s.null(); else s.b(p.dest_path);
    s.b(p.day_key).b(p.sn).b(p.organized_at).b(p.meta_state).b(p.write_error).b(p.id);
    s.run();
    reindex(p.id);
}

void Db::set_manual_date(int64_t id, const std::string& v) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET manual_date=?, resolved=0 WHERE id=?");
    s.b(v).b(id);
    s.run();
}

int Db::next_sn(const std::string& day_key, const std::string& out_root) {
    Lock l(mu_);
    std::string pre = out_root + "/";
    Stmt s(db_, "SELECT COALESCE(MAX(sn),0) FROM photos WHERE day_key=? AND substr(dest_path,1,length(?))=?");
    s.b(day_key).b(pre).b(pre);
    return s.step() ? int(s.i(0)) + 1 : 1;
}

std::vector<DupRow> Db::dup_rows(bool with_meta) {
    Lock l(mu_);
    std::vector<DupRow> out;
    Stmt s(db_, R"(SELECT id, size, mtime, COALESCE(dup_of,0), COALESCE(similar_to,0), src_path, src_root, COALESCE(dest_path,''),
                   COALESCE(quick_hash,''), COALESCE(content_hash,''), COALESCE(date_value,''), length(meta_json), ext,
                   COALESCE(dup_keep,0), COALESCE(dhash_done,0), COALESCE(dhash,0), COALESCE(width,0), COALESCE(height,0),
                   COALESCE(similar_dist,0), CASE WHEN ? THEN meta_json ELSE '' END FROM photos ORDER BY id)");
    s.b(with_meta);
    while (s.step()) {
        DupRow r;
        int c = 0;
        r.id = s.i(c++); r.size = s.i(c++); r.mtime = s.i(c++); r.dup_of = s.i(c++); r.similar_to = s.i(c++);
        r.src_path = s.t(c++); r.src_root = s.t(c++); r.dest_path = s.t(c++); r.quick_hash = s.t(c++); r.content_hash = s.t(c++);
        r.date_value = s.t(c++); r.meta_len = int(s.i(c++)); r.ext = s.t(c++); r.keep = s.i(c++) != 0; r.dhash_done = s.i(c++) != 0;
        r.dhash = uint64_t(s.i(c++)); r.width = int(s.i(c++)); r.height = int(s.i(c++)); r.similar_dist = int(s.i(c++));
        r.meta_json = s.t(c++);
        out.push_back(std::move(r));
    }
    return out;
}

void Db::set_content_hash(int64_t id, const std::string& h) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET content_hash=? WHERE id=?");
    s.b(h).b(id);
    s.run();
}

void Db::set_dhash(int64_t id, uint64_t h, bool ok) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET dhash=?, dhash_done=? WHERE id=?");
    s.b(int64_t(h)).b(ok ? 1 : 2).b(id);  // 2 = could not decode, do not retry every scan
    s.run();
}

void Db::apply_dups(const std::vector<int64_t>& scope_ids, const std::vector<std::pair<int64_t, int64_t>>& v) {
    Lock l(mu_);
    exec("BEGIN");
    for (int64_t id : scope_ids) {
        Stmt r(db_, "UPDATE photos SET dup_of=0 WHERE id=? AND dup_of<>0");
        r.b(id);
        r.run();
    }
    for (auto& [id, of] : v) {
        Stmt s(db_, "UPDATE photos SET dup_of=? WHERE id=?");
        s.b(of).b(id);
        s.run();
    }
    exec("COMMIT");
}

void Db::apply_similar(const std::vector<int64_t>& scope_ids, const std::vector<std::tuple<int64_t, int64_t, int>>& v) {
    Lock l(mu_);
    exec("BEGIN");
    for (int64_t id : scope_ids) {
        Stmt r(db_, "UPDATE photos SET similar_to=0, similar_dist=0 WHERE id=? AND similar_to<>0");
        r.b(id);
        r.run();
    }
    for (auto& [id, to, dist] : v) {
        Stmt s(db_, "UPDATE photos SET similar_to=?, similar_dist=? WHERE id=?");
        s.b(to).b(dist).b(id);
        s.run();
    }
    exec("COMMIT");
}

void Db::set_keeper(const std::vector<int64_t>& group, int64_t keep) {
    Lock l(mu_);
    exec("BEGIN");
    for (int64_t id : group) {
        Stmt s(db_, "UPDATE photos SET dup_keep=? WHERE id=?");
        s.b(id == keep).b(id);
        s.run();
    }
    exec("COMMIT");
}

void Db::remove_photo(int64_t id) {
    Lock l(mu_);
    Stmt d(db_, "DELETE FROM photos WHERE id=?");
    d.b(id);
    d.run();
    Stmt f(db_, "DELETE FROM photos_fts WHERE rowid=?");
    f.b(id);
    f.run();
}

void Db::move_src(int64_t id, const std::string& path) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET src_path=? WHERE id=?");
    s.b(path).b(id);
    s.run();
}

void Db::save_gen(int64_t id, const std::string& gen_json, const std::string& tool, const std::string& keywords_json) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET gen_json=?, gen_tool=?, gen_keywords=CASE WHEN ?='' THEN gen_keywords ELSE ? END, gen_checked=1 WHERE id=?");
    s.b(gen_json).b(tool).b(keywords_json).b(keywords_json).b(id);
    s.run();
    reindex(id);
}

void Db::save_gen_keywords(int64_t id, const std::string& keywords_json) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET gen_keywords=? WHERE id=?");
    s.b(keywords_json).b(id);
    s.run();
    reindex(id);
}

void Db::save_desc_choice(int64_t id, const std::string& choice_json) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET desc_choice=? WHERE id=?");
    s.b(choice_json).b(id);
    s.run();
}

void Db::refresh_file(int64_t id, int64_t size, int64_t mtime, const std::string& quick_hash, const std::string& meta_json) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET size=?, mtime=?, quick_hash=?, meta_json=?, content_hash=NULL WHERE id=?");
    s.b(size).b(mtime).b(quick_hash).b(meta_json).b(id);
    s.run();
}

void Db::save_clip(int64_t id, const std::string& model, const std::vector<float>& vec, const std::string& tags_json, const std::string& scene,
                   const std::string& vocab) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET clip_model=?, clip_vec=?, clip_tags=?, clip_scene=?, clip_vocab=? WHERE id=?");
    s.b(model).blob(vec.data(), vec.size() * sizeof(float)).b(tags_json).b(scene).b(vocab).b(id);
    s.run();
    reindex(id);
}

void Db::save_tags(int64_t id, const std::string& tags_json, const std::string& scene, const std::string& vocab) {
    Lock l(mu_);
    // Changed tags make an organized photo's written keywords out of date ("base" = placed, metadata to refresh).
    Stmt s(db_, R"(UPDATE photos SET meta_state=CASE WHEN meta_state='full' AND COALESCE(clip_tags,'')<>? THEN 'base' ELSE meta_state END,
                   clip_tags=?, clip_scene=?, clip_vocab=? WHERE id=?)");
    s.b(tags_json).b(tags_json).b(scene).b(vocab).b(id);
    s.run();
    reindex(id);
}

bool Db::clip_vec(int64_t id, std::vector<float>& out) {
    Lock l(mu_);
    Stmt s(db_, "SELECT clip_vec FROM photos WHERE id=?");
    s.b(id);
    out.clear();
    if (!s.step()) return false;
    const void* blob = sqlite3_column_blob(s.s, 0);
    int bytes = sqlite3_column_bytes(s.s, 0);
    if (blob && bytes > 0) out.assign(static_cast<const float*>(blob), static_cast<const float*>(blob) + bytes / sizeof(float));
    return !out.empty();
}

void Db::set_user_tags(int64_t id, const std::string& tags_json) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET user_tags=?, meta_state=CASE WHEN meta_state='full' THEN 'base' ELSE meta_state END WHERE id=?");
    if (tags_json.empty()) s.null(); else s.b(tags_json);
    s.b(id);
    s.run();
    reindex(id);
}

std::vector<TagRow> Db::tag_rows(const std::string& folder, const std::string& model_id, const std::string& vocab) {
    Lock l(mu_);
    std::string sql = R"(SELECT id, src_path, COALESCE(dest_path,''), COALESCE(date_value,''), COALESCE(date_prec,''), COALESCE(location,''),
                         COALESCE(clip_tags,''), COALESCE(user_tags,''), COALESCE(tags,''), COALESCE(clip_model,''), COALESCE(clip_vocab,''),
                         COALESCE(meta_json,''), size, COALESCE(gen_tool,''), COALESCE(gen_keywords,''), COALESCE(tag_fix,''),
                         COALESCE(name_keywords,''), COALESCE(favorite,0) FROM photos WHERE (dup_of IS NULL OR dup_of=0))";
    if (!folder.empty()) sql += " AND (src_path=? OR substr(src_path,1,length(?))=?)";
    sql += " ORDER BY src_path";
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/");
    std::vector<TagRow> out;
    while (s.step()) {
        TagRow r;
        int c = 0;
        r.id = s.i(c++); r.src_path = s.t(c++); r.dest_path = s.t(c++); r.date_value = s.t(c++); r.date_prec = s.t(c++); r.location = s.t(c++);
        std::string clip = s.t(c++), user = s.t(c++), vl = s.t(c++), model = s.t(c++), voc = s.t(c++), meta = s.t(c++);
        r.size = s.i(c++);
        r.gen_tool = s.t(c++);
        std::string gen_kw = s.t(c++), fix = s.t(c++), name_kw = s.t(c++);
        r.favorite = s.i(c++) != 0;
        for (auto& [t, conf] : parse_tag_list(name_kw)) r.name_words.push_back(t);
        r.user_edited = !user.empty();
        r.tagged = !model.empty();
        r.stale = r.tagged && (model != model_id || (!vocab.empty() && voc != vocab));
        if (r.user_edited) r.tags = parse_tag_list(user);
        else {
            // Effective tags first (corrections applied), then the weak guesses for context.
            Photo tp;
            tp.tags = vl;
            tp.clip_tags = clip;
            tp.tag_fix = fix;
            r.tags = effective_tag_list(tp);
            std::set<std::string> have;
            for (auto& t : r.tags) have.insert(util::lower(t.first));
            json fj = json::parse(fix.empty() ? "{}" : fix, nullptr, false);
            std::set<std::string> removed;
            if (fj.is_object() && fj.contains("remove"))
                for (auto& x : fj["remove"]) if (x.is_string()) removed.insert(util::lower(x.get<std::string>()));
            for (auto& t : parse_tag_list(clip))
                if (!have.count(util::lower(t.first)) && !removed.count(util::lower(t.first))) r.tags.push_back(t);
        }
        for (auto& t : parse_tag_list(gen_kw)) r.prompt_tags.push_back(t.first);
        for (auto& t : r.tags) r.best = std::max(r.best, t.second);
        // What the file already carries (from the scan snapshot).
        json m = json::parse(meta.empty() ? "{}" : meta, nullptr, false);
        auto has = [&](const char* k) { return m.is_object() && m.contains(k); };
        r.exif_date = has("Exif.Photo.DateTimeOriginal") || has("Xmp.exif.DateTimeOriginal") || has("Xmp.photoshop.DateCreated");
        r.keywords = has("Xmp.dc.subject") || has("Iptc.Application2.Keywords");
        r.description = has("Xmp.dc.description") || has("Exif.Image.ImageDescription") || has("Iptc.Application2.Caption");
        r.place = has("Exif.GPSInfo.GPSLatitude") || has("Xmp.iptc.Location") || has("Xmp.photoshop.City") || has("Iptc.Application2.City");
        out.push_back(std::move(r));
    }
    return out;
}

void Db::add_decision(const Decision& d) {
    Lock l(mu_);
    Stmt s(db_, R"(INSERT INTO decisions(at, kind, photo_id, subject, question, options, rules_pick, jev_pick, final_pick, changed)
                   VALUES(?,?,?,?,?,?,?,?,?,?))");
    s.b(d.at.empty() ? util::now_iso() : d.at).b(d.kind).b(d.photo_id).b(d.subject).b(d.question).b(d.options).b(d.rules_pick).b(d.jev_pick)
        .b(d.final_pick).b(d.changed);
    s.run();
}

std::vector<Decision> Db::decisions(const std::string& folder, int limit) {
    Lock l(mu_);
    std::string sql = "SELECT id, at, kind, photo_id, subject, question, options, rules_pick, jev_pick, final_pick, changed FROM decisions";
    if (!folder.empty())
        sql += " WHERE subject=? OR substr(subject,1,length(?))=? OR (kind='search' AND photo_id IN (SELECT id FROM photos WHERE substr(src_path,1,length(?))=?))";
    sql += " ORDER BY id DESC LIMIT " + std::to_string(limit);
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/").b(folder + "/").b(folder + "/");
    std::vector<Decision> out;
    while (s.step()) {
        Decision d;
        int c = 0;
        d.id = s.i(c++); d.at = s.t(c++); d.kind = s.t(c++); d.photo_id = s.i(c++); d.subject = s.t(c++); d.question = s.t(c++);
        d.options = s.t(c++); d.rules_pick = s.t(c++); d.jev_pick = s.t(c++); d.final_pick = s.t(c++); d.changed = s.i(c++) != 0;
        out.push_back(std::move(d));
    }
    return out;
}

DecisionStats Db::decision_stats(const std::string& folder) {
    Lock l(mu_);
    std::string sql = "SELECT kind, COUNT(*), SUM(changed) FROM decisions";
    if (!folder.empty())
        sql += " WHERE subject=? OR substr(subject,1,length(?))=? OR (kind='search' AND photo_id IN (SELECT id FROM photos WHERE substr(src_path,1,length(?))=?))";
    sql += " GROUP BY kind";
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/").b(folder + "/").b(folder + "/");
    DecisionStats st;
    while (s.step()) {
        st.by_kind[s.t(0)] = {int(s.i(1)), int(s.i(2))};
        st.total += int(s.i(1));
        st.changed += int(s.i(2));
    }
    return st;
}

std::vector<SearchDoc> Db::search_docs(const std::string& folder) {
    Lock l(mu_);
    std::string sql = R"(SELECT id, COALESCE(date_value,''), COALESCE(date_prec,''), src_path, COALESCE(dest_path,''), COALESCE(meta_json,''), COALESCE(caption,''),
                         COALESCE(scene,''), COALESCE(objects,''), COALESCE(tags,''), COALESCE(landmark,''), COALESCE(location,''),
                         COALESCE(vision_text,''), COALESCE(clip_tags,''), COALESCE(clip_scene,''), clip_vec, COALESCE(clip_model,''),
                         COALESCE(user_tags,''), COALESCE(gen_json,''), COALESCE(gen_keywords,''), COALESCE(tag_fix,''),
                         COALESCE(name_keywords,''), COALESCE(favorite,0) FROM photos WHERE (dup_of IS NULL OR dup_of=0))";
    if (!folder.empty()) sql += " AND (src_path=? OR substr(src_path,1,length(?))=?)";
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/");
    std::vector<SearchDoc> out;
    while (s.step()) {
        SearchDoc d;
        int c = 0;
        d.id = s.i(c++);
        std::string date = s.t(c++);
        d.year = date.size() >= 4 ? atoi(date.substr(0, 4).c_str()) : 0;
        d.date = date;
        d.prec = s.t(c++);
        std::string src = s.t(c++), dest = s.t(c++);
        d.name_raw = src + (dest.empty() ? "" : "\n" + dest);
        // EXIF/IPTC/XMP as "Key = value" lines (skipping our own Xmp.jev.* echoes)
        json m = json::parse(s.t(c++), nullptr, false);
        if (m.is_object())
            for (auto& [k, v] : m.items()) {
                if (k == "__error" || util::starts_with(k, "Xmp.jev.") || !v.contains("v")) continue;
                for (auto& x : v["v"])
                    if (x.is_string()) d.exif += k + " = " + x.get<std::string>() + "\n";
            }
        Photo tp;  // just the tag columns, for effective_tags()
        std::string caption = s.t(c++), scene = s.t(c++), objects = join_json_list(s.t(c++));
        tp.tags = s.t(c++);
        std::string landmark = s.t(c++), location = s.t(c++), vtext = s.t(c++);
        tp.clip_tags = s.t(c++);
        for (auto& t : parse_tag_list(tp.clip_tags)) d.clip_tags.push_back(t);
        std::string clip_scene = s.t(c++);
        const void* blob = sqlite3_column_blob(s.s, c);
        int bytes = sqlite3_column_bytes(s.s, c++);
        if (blob && bytes > 0) d.vec.assign(static_cast<const float*>(blob), static_cast<const float*>(blob) + bytes / sizeof(float));
        d.clip_model = s.t(c++);
        tp.user_tags = s.t(c++);
        GenInfo gen = GenInfo::from_json(s.t(c++));
        std::string gen_kw = s.t(c++);
        tp.tag_fix = s.t(c++);
        std::string name_kw = s.t(c++);
        d.favorite = s.i(c++) != 0;
        // Only confident tags are searchable words: a tag the model gave 10% is a guess, not a description.
        std::string words;
        for (auto& t : effective_tags(tp)) {
            words += (words.empty() ? "" : ", ") + t;
            d.tags.push_back(util::lower(t));
        }
        std::string gen_words, name_words;
        for (auto& [t, conf] : parse_tag_list(gen_kw)) gen_words += (gen_words.empty() ? "" : ", ") + t;
        for (auto& [t, conf] : parse_tag_list(name_kw)) name_words += (name_words.empty() ? "" : ", ") + t;
        std::string fav = d.favorite ? "favorite" : "";
        bool own = !tp.user_tags.empty();  // the user's own tags replace the automatic scene too
        d.desc_raw = caption;
        for (const std::string* part : {own ? &fav : &scene, own ? &fav : &clip_scene, &words, &objects, &landmark, &location, &vtext, &gen_words,
                                        &name_words, &fav})
            if (!part->empty() && d.desc_raw.find(*part) == std::string::npos) d.desc_raw += (d.desc_raw.empty() ? "" : " | ") + *part;
        if (gen.found()) {
            // The prompt is its own field: long, and written by whoever made the picture.
            d.prompt_raw = gen.prompt + (gen.negative.empty() ? "" : "\nNegative: " + gen.negative) + "\n" + gen.tool + " " + gen.model;
            for (auto& [lo, w] : gen.loras) d.prompt_raw += " " + lo;
            d.prompt = util::lower(d.prompt_raw);
        }
        d.name = util::lower(d.name_raw);
        d.exif = util::lower(d.exif);
        d.desc = util::lower(d.desc_raw);
        out.push_back(std::move(d));
    }
    return out;
}

void Db::save_name_keywords(int64_t id, const std::string& j) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET name_keywords=? WHERE id=?");
    s.b(j).b(id);
    s.run();
    reindex(id);
}

void Db::set_fix_checked(int64_t id, const std::string& key) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET fix_checked=? WHERE id=?");
    s.b(key).b(id);
    s.run();
}

void Db::set_favorite(int64_t id, bool on) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE photos SET favorite=? WHERE id=?");
    s.b(on).b(id);
    s.run();
}

void Db::save_tag_fix(int64_t id, const std::string& j) {
    Lock l(mu_);
    // Changed tags make written keywords out of date ("base" = placed, metadata to refresh).
    Stmt s(db_, "UPDATE photos SET tag_fix=?, meta_state=CASE WHEN meta_state='full' THEN 'base' ELSE meta_state END WHERE id=?");
    s.b(j).b(id);
    s.run();
    reindex(id);
}

void Db::replace_corrections(const std::vector<int64_t>& photo_ids, const std::vector<Correction>& pending) {
    Lock l(mu_);
    exec("BEGIN");
    for (int64_t id : photo_ids) {
        Stmt d(db_, "DELETE FROM corrections WHERE photo_id=? AND status='pending'");
        d.b(id);
        d.run();
    }
    for (auto& c : pending) {
        Stmt i(db_, R"(INSERT INTO corrections(photo_id, field, action, current, proposed, confidence, reason, source, status, at)
                       VALUES(?,?,?,?,?,?,?,?,'pending',?))");
        i.b(c.photo_id).b(c.field).b(c.action).b(c.current).b(c.proposed).b(c.confidence).b(c.reason).b(c.source).b(util::now_iso());
        i.run();
    }
    exec("COMMIT");
}

std::vector<Correction> Db::corrections(const std::string& folder, bool pending_only) {
    Lock l(mu_);
    std::string sql = R"(SELECT c.id, c.photo_id, c.field, c.action, c.current, c.proposed, c.confidence, c.reason, c.source, c.status, p.src_path
                         FROM corrections c JOIN photos p ON p.id=c.photo_id WHERE 1)";
    if (pending_only) sql += " AND c.status='pending'";
    if (!folder.empty()) sql += " AND (p.src_path=? OR substr(p.src_path,1,length(?))=?)";
    sql += " ORDER BY c.confidence DESC, p.src_path";
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/");
    std::vector<Correction> out;
    while (s.step()) {
        Correction c;
        int k = 0;
        c.id = s.i(k++); c.photo_id = s.i(k++); c.field = s.t(k++); c.action = s.t(k++); c.current = s.t(k++); c.proposed = s.t(k++);
        c.confidence = s.d(k++); c.reason = s.t(k++); c.source = s.t(k++); c.status = s.t(k++); c.path = s.t(k++);
        out.push_back(std::move(c));
    }
    return out;
}

void Db::set_correction_status(int64_t id, const std::string& status) {
    Lock l(mu_);
    Stmt s(db_, "UPDATE corrections SET status=? WHERE id=?");
    s.b(status).b(id);
    s.run();
}

bool Db::correction_rejected(int64_t photo_id, const std::string& field, const std::string& current, const std::string& proposed) {
    Lock l(mu_);
    Stmt s(db_, "SELECT 1 FROM corrections WHERE photo_id=? AND field=? AND current=? AND proposed=? AND status IN ('rejected','applied')");
    s.b(photo_id).b(field).b(current).b(proposed);
    return s.step();
}

std::string Db::translation(const std::string& term) {
    Lock l(mu_);
    Stmt s(db_, "SELECT alts FROM translations WHERE term=?");
    s.b(term);
    return s.step() ? s.t(0) : "";
}

void Db::save_translation(const std::string& term, const std::string& j) {
    Lock l(mu_);
    Stmt s(db_, "INSERT OR REPLACE INTO translations(term, alts, at) VALUES(?,?,?)");
    s.b(term).b(j).b(util::now_iso());
    s.run();
}

bool Db::dest_taken(const std::string& dest) {
    Lock l(mu_);
    Stmt s(db_, "SELECT 1 FROM photos WHERE dest_path=?");
    s.b(dest);
    return s.step();
}

DirPlace Db::get_dir(const std::string& path) {
    Lock l(mu_);
    DirPlace d;
    Stmt s(db_, "SELECT location, location_conf, location_source, evidence FROM dirs WHERE path=?");
    s.b(path);
    if (s.step()) {
        d.found = true;
        d.location = s.t(0);
        d.conf = s.d(1);
        d.source = s.t(2);
        d.evidence = s.t(3);
    }
    return d;
}

void Db::put_dir(const std::string& path, const DirPlace& d) {
    Lock l(mu_);
    Stmt s(db_, "INSERT OR REPLACE INTO dirs VALUES(?,?,?,?,?,?)");
    s.b(path).b(d.location).b(d.conf).b(d.source).b(d.evidence).b(util::now_iso());
    s.run();
}

void Db::clear_dirs() {
    Lock l(mu_);
    exec("DELETE FROM dirs");
}

void Db::reindex(int64_t id) {
    Lock l(mu_);
    Photo p;
    if (!load(id, p)) return;
    std::string words;
    for (auto& t : effective_tags(p)) words += t + ", ";
    for (auto& [t, c] : parse_tag_list(p.gen_keywords)) words += t + ", ";
    GenInfo gen = GenInfo::from_json(p.gen_json);
    if (gen.found()) words += " | " + gen.tool + " " + gen.model + " " + gen.prompt;
    std::string text = p.caption + " | " + p.scene + " | " + p.clip_scene + " | " + words + " | " + join_json_list(p.objects) + " | " +
                       p.landmark + " | " + p.location + " | " + p.vision_text + " | " + util::basename(p.src_path) + " | " +
                       util::basename(util::dirname(p.src_path)) + " | " + p.date_value.substr(0, 10);
    Stmt d(db_, "DELETE FROM photos_fts WHERE rowid=?");
    d.b(id);
    d.run();
    Stmt i(db_, "INSERT INTO photos_fts(rowid, text) VALUES(?,?)");
    i.b(id).b(text);
    i.run();
    Stmt u(db_, "UPDATE photos SET search_text=? WHERE id=?");
    u.b(util::lower(text)).b(id);
    u.run();
}

static const char* kUnder = " AND (src_path=? OR substr(src_path,1,length(?))=?)";

std::vector<PhotoRow> Db::query(const std::string& search, bool review_only, const std::string& month, int limit, const std::string& folder) {
    Lock l(mu_);
    std::string sql = R"(SELECT id, date_value, date_prec, date_source, date_decider, location, scene, tags, dest_path, src_path,
                         vision_status, date_conf, needs_review, dup_of, COALESCE(similar_to,0), size, COALESCE(width,0),
                         COALESCE(height,0), COALESCE(clip_tags,''), COALESCE(clip_scene,''), COALESCE(user_tags,''), COALESCE(tag_fix,''),
                         COALESCE(favorite,0) FROM photos WHERE 1)";
    std::vector<std::string> binds;
    for (auto& raw : util::split(search, ' ')) {
        std::string t = util::trim(raw);
        if (t.empty()) continue;
        // trigram FTS needs >= 3 characters; shorter terms (e.g. 2-character Chinese words) use LIKE.
        if (util::utf8_len(t) >= 3) {
            sql += " AND id IN (SELECT rowid FROM photos_fts WHERE photos_fts MATCH ?)";
            binds.push_back("\"" + util::replace_all(t, "\"", "\"\"") + "\"");
        } else {
            sql += " AND search_text LIKE ?";
            binds.push_back("%" + util::lower(t) + "%");
        }
    }
    if (!folder.empty()) {
        sql += kUnder;
        binds.push_back(folder);
        binds.push_back(folder + "/");
        binds.push_back(folder + "/");
    }
    if (review_only) sql += " AND (needs_review=1 OR date_value IS NULL OR date_value='')";
    if (month == "undated") sql += " AND (date_value IS NULL OR date_value='')";
    else if (!month.empty()) { sql += " AND date_value LIKE ?"; binds.push_back(month + "%"); }
    sql += " ORDER BY CASE WHEN date_value IS NULL OR date_value='' THEN 1 ELSE 0 END, date_value, src_path LIMIT " + std::to_string(limit);
    Stmt s(db_, sql.c_str());
    for (auto& b : binds) s.b(b);
    std::vector<PhotoRow> out;
    while (s.step()) {
        PhotoRow r;
        int c = 0;
        r.id = s.i(c++); r.date_value = s.t(c++); r.date_prec = s.t(c++); r.date_source = s.t(c++); r.decider = s.t(c++);
        r.location = s.t(c++); r.scene = s.t(c++);
        Photo tp;
        tp.tags = s.t(c++);
        r.dest_path = s.t(c++); r.src_path = s.t(c++);
        r.vision_status = s.t(c++); r.date_conf = s.d(c++); r.needs_review = s.i(c++) != 0; r.dup_of = s.i(c++);
        r.similar_to = s.i(c++); r.size = s.i(c++); r.width = int(s.i(c++)); r.height = int(s.i(c++));
        tp.clip_tags = s.t(c++);
        std::string clip_scene = s.t(c++);
        tp.user_tags = s.t(c++);
        tp.tag_fix = s.t(c++);
        r.favorite = s.i(c++) != 0;
        for (auto& t : effective_tags(tp)) r.tags += (r.tags.empty() ? "" : ", ") + t;
        if (r.scene.empty()) r.scene = clip_scene;
        out.push_back(std::move(r));
    }
    return out;
}

std::vector<MonthInfo> Db::months(const std::string& folder) {
    Lock l(mu_);
    std::vector<MonthInfo> out;
    std::string sql = R"(SELECT CASE WHEN date_value IS NULL OR date_value='' THEN 'undated'
                                WHEN date_prec='year' THEN substr(date_value,1,4) ELSE substr(date_value,1,7) END AS m, COUNT(*),
                   SUM(size) FROM photos WHERE 1)";
    if (!folder.empty()) sql += kUnder;
    sql += " GROUP BY m ORDER BY m";
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/");
    while (s.step()) out.push_back({s.t(0), int(s.i(1)), s.i(2)});
    return out;
}

DbStats Db::stats(const std::string& folder) {
    Lock l(mu_);
    DbStats st;
    std::string sql = R"(SELECT COUNT(*), SUM(resolved), SUM(resolved AND needs_review), SUM(resolved AND (date_value IS NULL OR date_value='')),
                   SUM(vision_status='done'), SUM(vision_status='failed'), SUM(dest_path IS NOT NULL), SUM(dup_of>0),
                   SUM(location IS NOT NULL AND location<>''), COALESCE(SUM(size),0),
                   COALESCE(SUM(CASE WHEN dup_of>0 THEN size ELSE 0 END),0), SUM(COALESCE(similar_to,0)>0), SUM(clip_model IS NOT NULL AND clip_model<>'') FROM photos WHERE 1)";
    if (!folder.empty()) sql += kUnder;
    Stmt s(db_, sql.c_str());
    if (!folder.empty()) s.b(folder).b(folder + "/").b(folder + "/");
    if (s.step()) {
        st.total = int(s.i(0)); st.resolved = int(s.i(1)); st.review = int(s.i(2)); st.undated = int(s.i(3));
        st.vision_done = int(s.i(4)); st.vision_failed = int(s.i(5)); st.organized = int(s.i(6)); st.dups = int(s.i(7));
        st.located = int(s.i(8));
        st.total_bytes = s.i(9); st.dup_bytes = s.i(10); st.similar = int(s.i(11)); st.tagged = int(s.i(12));
    }
    return st;
}
