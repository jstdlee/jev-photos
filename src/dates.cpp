#include "dates.h"

#include <algorithm>
#include <cmath>
#include <regex>

#include "nlohmann/json.hpp"

using json = nlohmann::json;
using util::Civil;
using util::make_civil;

namespace {

struct SrcInfo {
    const char* id;
    const char* label;
    const char* family;
    double base;
    bool upper_bound;  // the photo cannot be newer than this (file/edit times)
};

const SrcInfo kSrc[DS_COUNT] = {
    {"exif-original", "camera EXIF original time", "embedded", 0.95, false},
    {"exif-digitized", "camera EXIF digitized time", "embedded", 0.90, false},
    {"xmp-created", "XMP/IPTC creation date", "embedded", 0.85, false},
    {"gps", "GPS satellite time", "gps", 0.80, false},
    {"exif-modify", "EXIF modify time", "embedded", 0.55, true},
    {"name-datetime", "date and time in the file name", "name", 0.80, false},
    {"name-date", "date in the file name", "name", 0.70, false},
    {"name-unix", "timestamp in the file name", "name", 0.72, false},
    {"name-month", "month in the file name", "name", 0.40, false},
    {"dir-date", "date in the folder name", "dir", 0.55, false},
    {"dir-month", "month in the folder name", "dir", 0.45, false},
    {"dir-year", "year in the folder name", "dir", 0.30, false},
    {"fs-birth", "file creation time", "fs", 0.30, true},
    {"fs-mtime", "file modified time", "fs", 0.30, true},
    {"neighbor", "neighbouring photos in the same camera sequence", "neighbor", 0.60, false},
    {"manual", "set by hand", "manual", 1.00, false},
};

constexpr int64_t kDay = 86400;
constexpr int64_t kSlack = 36 * 3600;  // time-zone + clock slack

bool embedded(DateSource s) { return std::string(kSrc[s].family) == "embedded"; }
bool fs_src(DateSource s) { return s == DS_FS_BIRTH || s == DS_FS_MTIME; }

DateCandidate make(DateSource s, const Civil& c, const std::string& raw, double decay = 1.0) {
    DateCandidate d;
    d.src = s;
    d.when = c;
    d.base = kSrc[s].base * decay;
    d.weight = d.base;
    d.raw = raw;
    return d;
}

int64_t day_of(const Civil& c) { return util::days_from_civil(c.y, c.mo, c.d); }

// ---- name patterns

struct Pat {
    std::regex re;
    DateSource file_src, dir_src;  // DS_COUNT = not used for that kind of name
    util::Precision prec;
    int groups;  // y,m,d,H,M,S count used
};

const std::vector<Pat>& patterns() {
    static const std::vector<Pat> p = [] {
        auto R = [](const char* s) { return std::regex(s, std::regex::ECMAScript | std::regex::icase | std::regex::optimize); };
        std::vector<Pat> v;
        // IMG_20190512_143022, PXL_20210101_123456789, 20190512143022, VID-20190512-143022
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])[ _T.-]?([01][0-9]|2[0-3])([0-5][0-9])([0-5][0-9])"),
                     DS_NAME_DATETIME, DS_DIR_DATE, util::P_SECOND, 6});
        // 2019-05-12 14.30.22, Screenshot_2019-05-12-14-30-22, Photo 2019-05-12 at 14.30.22, 2019-05-12_14-30
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})[-_.](0?[1-9]|1[0-2])[-_.](0?[1-9]|[12][0-9]|3[01])(?:[ _T.-]+at[ _]|[ _T.-]+)([01]?[0-9]|2[0-3])[-_.:h]([0-5][0-9])(?:[-_.:m]([0-5][0-9]))?(?![0-9])"),
                     DS_NAME_DATETIME, DS_DIR_DATE, util::P_SECOND, 6});
        // IMG-20190512-WA0001, 20190512
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?![0-9])"),
                     DS_NAME_DATE, DS_DIR_DATE, util::P_DAY, 3});
        // 2019-05-12, 2019.5.12, 2019年5月12日
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})(?:[-_.]|\xE5\xB9\xB4)(0?[1-9]|1[0-2])(?:[-_.]|\xE6\x9C\x88)(0?[1-9]|[12][0-9]|3[01])(?![0-9])"),
                     DS_NAME_DATE, DS_DIR_DATE, util::P_DAY, 3});
        // 2019-05, 2019.05, 2019年5月
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})(?:[-_.]|\xE5\xB9\xB4)(0?[1-9]|1[0-2])(?![0-9])"),
                     DS_NAME_MONTH, DS_DIR_MONTH, util::P_MONTH, 2});
        // 201905 (folders only; in file names six digits are too ambiguous)
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})(0[1-9]|1[0-2])(?![0-9])"), DS_COUNT, DS_DIR_MONTH, util::P_MONTH, 2});
        // 2019 (folders only)
        v.push_back({R("(?:^|[^0-9])((?:19|20)[0-9]{2})(?![0-9])"), DS_COUNT, DS_DIR_YEAR, util::P_YEAR, 1});
        return v;
    }();
    return p;
}

int month_from_name(const std::string& s) {
    static const char* m[] = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};
    std::string l = util::lower(s.substr(0, 3));
    for (int i = 0; i < 12; i++)
        if (l == m[i]) return i + 1;
    return 0;
}

}  // namespace

const char* date_source_id(DateSource s) { return s < DS_COUNT ? kSrc[s].id : "none"; }
const char* date_source_label(DateSource s) { return s < DS_COUNT ? kSrc[s].label : "none"; }

std::vector<DateCandidate> date_candidates_from_name(const std::string& name, bool is_dir, int level, int64_t now) {
    std::vector<DateCandidate> out;
    std::vector<std::pair<size_t, size_t>> used;
    auto overlaps = [&](size_t a, size_t b) {
        for (auto& [x, y] : used)
            if (a < y && x < b) return true;
        return false;
    };
    double decay = is_dir ? std::pow(0.9, level) : 1.0;

    for (auto& p : patterns()) {
        DateSource src = is_dir ? p.dir_src : p.file_src;
        if (src == DS_COUNT) continue;
        for (auto it = std::sregex_iterator(name.begin(), name.end(), p.re); it != std::sregex_iterator(); ++it) {
            const std::smatch& m = *it;
            size_t a = size_t(m.position(1)), b = size_t(m.position(0) + m.length(0));
            if (overlaps(a, b)) continue;
            auto g = [&](int i) { return i <= p.groups && m[i].matched ? atoi(m[i].str().c_str()) : 0; };
            util::Precision prec = p.prec;
            if (prec == util::P_SECOND && !m[6].matched) prec = util::P_MINUTE;
            if (is_dir && prec > util::P_DAY) prec = util::P_DAY;  // a folder names a day, not a moment
            Civil c = make_civil(g(1), p.groups >= 2 ? g(2) : 1, p.groups >= 3 ? g(3) : 1, g(4), g(5), g(6), prec);
            if (!c.valid()) continue;
            used.push_back({a, b});
            out.push_back(make(src, c, m.str(0), decay));
        }
    }
    // "May 2019" / "May_2019" in folder names
    if (is_dir) {
        static const std::regex mon(R"((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ _.-]*((?:19|20)[0-9]{2}))",
                                    std::regex::ECMAScript | std::regex::icase);
        for (auto it = std::sregex_iterator(name.begin(), name.end(), mon); it != std::sregex_iterator(); ++it) {
            size_t a = size_t(it->position(0)), b = a + size_t(it->length(0));
            if (overlaps(a, b)) {
                // "May 2019" also matched the bare-year rule; the month is more precise, so replace it.
                auto yr = std::find_if(out.begin(), out.end(), [](const DateCandidate& d) { return d.src == DS_DIR_YEAR; });
                if (yr == out.end()) continue;
                out.erase(yr);
            }
            Civil c = make_civil(atoi((*it)[2].str().c_str()), month_from_name((*it)[1].str()), 1, 0, 0, 0, util::P_MONTH);
            if (c.valid()) out.push_back(make(DS_DIR_MONTH, c, it->str(0), decay));
        }
    }
    // Unix timestamps (mmexport1557671422000, wx_camera_1557671422000) only when nothing calendar-like matched.
    if (!is_dir && out.empty()) {
        static const std::regex unix_re("(?:^|[^0-9])(1[0-9]{12}|1[0-9]{9})(?![0-9])");
        std::smatch m;
        if (std::regex_search(name, m, unix_re)) {
            std::string d = m[1].str();
            int64_t t = std::stoll(d);
            if (d.size() == 13) t /= 1000;
            if (t > 978307200 && t <= now + kDay) {  // 2001-01-01 .. now
                DateCandidate c = make(DS_NAME_UNIX, util::local_from_epoch(t), d);
                c.notes.push_back("epoch seconds shown in this computer's time zone");
                out.push_back(c);
            }
        }
    }
    return out;
}

std::vector<DateCandidate> collect_date_candidates(const DateInputs& in) {
    std::vector<DateCandidate> c;
    const MetaMap empty;
    const MetaMap& m = in.meta ? *in.meta : empty;
    int64_t now = in.now ? in.now : util::now_epoch();

    auto add_meta = [&](DateSource s, const std::string& key) {
        if (!m.has(key)) return;
        std::string raw = m.get(key);
        Civil t = util::parse_exif_datetime(raw);
        DateCandidate d = make(s, t, key + " = " + raw);
        if (!t.valid()) { d.weight = 0; d.notes.push_back("unparseable value"); }
        // The same value can be mirrored in several XMP fields; count it once.
        for (auto& e : c)
            if (e.src == s && t.valid() && e.when.valid() && e.when.secs() == t.secs() && e.when.prec == t.prec) return;
        c.push_back(d);
    };
    add_meta(DS_EXIF_ORIGINAL, "Exif.Photo.DateTimeOriginal");
    add_meta(DS_EXIF_DIGITIZED, "Exif.Photo.DateTimeDigitized");
    add_meta(DS_XMP_CREATED, "Xmp.exif.DateTimeOriginal");
    add_meta(DS_XMP_CREATED, "Xmp.photoshop.DateCreated");
    add_meta(DS_XMP_CREATED, "Xmp.xmp.CreateDate");
    if (m.has("Iptc.Application2.DateCreated")) {
        std::string raw = m.get("Iptc.Application2.DateCreated") + " " + m.get("Iptc.Application2.TimeCreated");
        Civil t = util::parse_exif_datetime(raw);
        if (t.valid()) {
            bool dup = false;
            for (auto& e : c) dup |= e.src == DS_XMP_CREATED && e.when.valid() && day_of(e.when) == day_of(t);
            if (!dup) c.push_back(make(DS_XMP_CREATED, t, "Iptc.Application2.DateCreated = " + raw));
        }
    }
    add_meta(DS_EXIF_MODIFY, "Exif.Image.DateTime");

    int64_t gps = 0;
    if (meta_gps_time(m, gps)) {
        int off = 0;
        DateCandidate d;
        if (meta_offset(m, off)) {
            d = make(DS_GPS, util::civil_from_secs(gps + off, util::P_SECOND), "GPS UTC + OffsetTime");
        } else {
            d = make(DS_GPS, util::local_from_epoch(gps), "GPS UTC");
            d.notes.push_back("UTC shown in this computer's time zone");
        }
        c.push_back(d);
    }

    for (auto& d : date_candidates_from_name(util::stem(in.filename), false, 0, now)) c.push_back(d);
    for (size_t i = 0; i < in.dirs.size() && i < 4; i++)
        for (auto& d : date_candidates_from_name(in.dirs[i], true, int(i), now)) c.push_back(d);

    if (in.btime > 0) {
        DateCandidate d = make(DS_FS_BIRTH, util::local_from_epoch(in.btime), "statx btime");
        // A copy gets a fresh birth time but keeps its mtime: birth after modify means "copied here", not "taken".
        if (in.mtime > 0 && in.btime > in.mtime + 60) {
            d.weight = 0;
            d.notes.push_back("created after last modified: copy time");
        }
        c.push_back(d);
    }
    if (in.mtime > 0) {
        DateCandidate d = make(DS_FS_MTIME, util::local_from_epoch(in.mtime), "st_mtime");
        if (in.bulk_mtime) {
            d.weight *= 0.15;
            d.notes.push_back("shared by many files in the folder (copy time)");
        }
        c.push_back(d);
    }

    // ---- sanity rules
    Civil now_c = util::local_from_epoch(now);
    int64_t fs_min = 0;
    for (auto& d : c)
        if (fs_src(d.src) && d.when.valid() && (fs_min == 0 || d.when.secs() < fs_min)) fs_min = d.when.secs();

    for (auto& d : c) {
        if (d.weight <= 0 || !d.when.valid()) { d.weight = 0; continue; }
        const Civil& t = d.when;
        if (t.secs() > now_c.secs() + kDay) { d.weight = 0; d.notes.push_back("in the future"); continue; }
        if (t.y < in.min_year) { d.weight = 0; d.notes.push_back("before the minimum year"); continue; }
        if ((t.y == 1970 || t.y == 1980) && t.mo == 1 && t.d == 1) { d.weight = 0; d.notes.push_back("epoch sentinel"); continue; }
        if (embedded(d.src) || d.src == DS_GPS) {
            if (t.prec >= util::P_DAY && t.mo == 1 && t.d == 1 && (t.y <= 2005 || (t.h == 0 && t.mi < 10))) {
                d.weight *= 0.3;
                d.notes.push_back("looks like a camera clock at its reset default");
            }
        }
        if (fs_src(d.src) && t.y < 1995) { d.weight *= 0.5; d.notes.push_back("implausibly old file time"); }
        if (!fs_src(d.src) && d.src != DS_EXIF_MODIFY && fs_min > 0 && t.secs() > fs_min + kSlack) {
            d.weight *= 0.5;
            d.notes.push_back("later than the file's own timestamp");
        }
    }
    return c;
}

double date_agree(const DateCandidate& a, const DateCandidate& b) {
    if (!a.when.valid() || !b.when.valid()) return 0;
    util::Precision p = std::min(a.when.prec, b.when.prec);
    const Civil &x = a.when, &y = b.when;
    if (p >= util::P_MINUTE) {
        int64_t diff = std::llabs(x.secs() - y.secs());
        if (diff <= 120) return 1.0;
        int64_t q = diff % 900;
        if (diff <= 14 * 3600 + 120 && (q <= 120 || q >= 780)) return 0.7;  // whole time-zone offset apart
        int64_t dd = std::llabs(day_of(x) - day_of(y));
        return dd == 0 ? 0.5 : dd == 1 ? 0.3 : 0.0;
    }
    if (p == util::P_DAY) {
        int64_t dd = std::llabs(day_of(x) - day_of(y));
        return dd == 0 ? 0.9 : dd == 1 ? 0.4 : 0.0;
    }
    if (p == util::P_MONTH) return x.y == y.y && x.mo == y.mo ? 0.7 : 0.0;
    if (p == util::P_YEAR) return x.y == y.y ? 0.5 : 0.0;
    return 0;
}

namespace {

double independence(const DateCandidate& a, const DateCandidate& b) {
    return std::string(kSrc[a.src].family) == kSrc[b.src].family ? 0.35 : 1.0;
}

// k speaks against w: it disagrees, and it is not merely a later upper bound (copy/edit time).
bool contradicts(const DateCandidate& w, const DateCandidate& k) {
    if (date_agree(w, k) >= 0.3) return false;
    if (kSrc[k.src].upper_bound) return k.when.secs() < w.when.secs() - kSlack;
    return true;
}

// Fill when/source/confidence from a chosen winner index.
void finalize(DateDecision& d, int win, double review_below) {
    auto& c = d.candidates;
    const DateCandidate& w = c[win];
    d.when = w.when;
    d.source = date_source_id(w.src);
    // A coarse winner borrows finer detail from the strongest agreeing, finer candidate.
    if (w.when.prec < util::P_MINUTE) {
        int best = -1;
        for (int j = 0; j < int(c.size()); j++) {
            if (j == win || c[j].rejected() || c[j].weight < 0.25 || c[j].when.prec <= w.when.prec) continue;
            bool inside = w.when.prec == util::P_DAY ? day_of(c[j].when) == day_of(w.when)
                        : w.when.prec == util::P_MONTH ? (c[j].when.y == w.when.y && c[j].when.mo == w.when.mo)
                                                       : c[j].when.y == w.when.y;
            if (inside && (best < 0 || c[j].weight > c[best].weight)) best = j;
        }
        if (best >= 0) {
            d.when = c[best].when;
            d.source += std::string("+") + (w.when.prec == util::P_DAY ? "time:" : "detail:") + date_source_id(c[best].src);
        }
    }
    double s_alt = 0;
    for (int k = 0; k < int(c.size()); k++)
        if (k != win && !c[k].rejected() && contradicts(w, c[k])) s_alt = std::max(s_alt, c[k].support);
    d.margin = w.support / (w.support + s_alt);
    d.confidence = d.margin * std::min(1.0, w.support);
    d.needs_review = d.confidence < review_below;
}

}  // namespace

DateDecision decide_from_candidates(std::vector<DateCandidate> cands, double review_below) {
    DateDecision d;
    d.candidates = std::move(cands);
    auto& c = d.candidates;
    for (size_t i = 0; i < c.size(); i++) {
        c[i].support = 0;
        if (c[i].rejected()) continue;
        double s = c[i].weight;
        for (size_t j = 0; j < c.size(); j++)
            if (j != i && !c[j].rejected()) s += c[j].weight * date_agree(c[i], c[j]) * independence(c[i], c[j]);
        c[i].support = s;
    }
    std::vector<int> order;
    for (int i = 0; i < int(c.size()); i++)
        if (!c[i].rejected()) order.push_back(i);
    if (order.empty()) {
        d.source = "none";
        d.needs_review = true;
        return d;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::fabs(c[a].support - c[b].support) > 1e-9) return c[a].support > c[b].support;
        if (c[a].when.prec != c[b].when.prec) return c[a].when.prec > c[b].when.prec;
        return c[a].base > c[b].base;
    });
    int win = order[0];
    // File/modify times only bound the capture date from above (a copy is newer than the photo, never older). If
    // one of them won but a real source (EXIF, name, folder...) gives an earlier, consistent date, that source wins.
    if (kSrc[c[win].src].upper_bound) {
        int alt = -1;
        for (int i : order)
            if (!kSrc[c[i].src].upper_bound && c[i].weight >= 0.25 && c[i].when.secs() <= c[win].when.secs() + kSlack &&
                (alt < 0 || c[i].support > c[alt].support))
                alt = i;
        if (alt >= 0) win = alt;
    }
    // Manual dates always win.
    for (int i : order)
        if (c[i].src == DS_MANUAL) win = i;
    d.alternatives.push_back(win);
    for (int k : order) {
        if (int(d.alternatives.size()) >= 4) break;
        if (k == win || !contradicts(c[win], c[k])) continue;
        bool distinct = true;
        for (int a : d.alternatives) distinct &= date_agree(c[a], c[k]) < 0.3;
        if (distinct) d.alternatives.push_back(k);
    }
    finalize(d, win, review_below);
    if (c[win].src == DS_MANUAL) {
        d.decider = "manual";
        d.confidence = 1;
        d.needs_review = false;
    }
    return d;
}

DateDecision decide_date(const DateInputs& in, double review_below) {
    return decide_from_candidates(collect_date_candidates(in), review_below);
}

std::string date_option_text(const DateDecision& d, int idx) {
    const DateCandidate& c = d.candidates[idx];
    std::string s = c.when.pretty() + " from " + date_source_label(c.src);
    int shown = 0;
    for (auto& o : d.candidates) {
        if (&o == &c || o.rejected() || shown >= 2) continue;
        if (std::string(kSrc[o.src].family) != kSrc[c.src].family && date_agree(c, o) >= 0.5) {
            s += std::string(shown ? ", " : "; matches ") + date_source_label(o.src);
            shown++;
        }
    }
    if (!c.notes.empty()) s += " (" + c.notes[0] + ")";
    return s;
}

void apply_jev_date(DateDecision& d, const std::vector<double>& probs, double weight, double review_below) {
    if (probs.size() != d.alternatives.size() || probs.size() < 2) return;
    double total = 0;
    for (int a : d.alternatives) total += d.candidates[a].support;
    if (total <= 0) return;
    std::vector<double> fin;
    for (size_t k = 0; k < probs.size(); k++)
        fin.push_back((1 - weight) * d.candidates[d.alternatives[k]].support / total + weight * probs[k]);
    size_t best = size_t(std::max_element(fin.begin(), fin.end()) - fin.begin());
    std::string note = "jev p=[";
    for (size_t k = 0; k < probs.size(); k++) note += util::fmt("%s%.2f", k ? "," : "", probs[k]);
    note += "] blended=[";
    for (size_t k = 0; k < fin.size(); k++) note += util::fmt("%s%.2f", k ? "," : "", fin[k]);
    d.jev_note = note + "]";
    int win = d.alternatives[best];
    if (best != 0) std::swap(d.alternatives[0], d.alternatives[best]);
    finalize(d, win, review_below);
    // Blended margin: how clearly the combined vote separates the winner from the runner-up.
    double second = 0;
    for (size_t k = 0; k < fin.size(); k++)
        if (k != best) second = std::max(second, fin[k]);
    d.margin = fin[best] / (fin[best] + second);
    d.confidence = d.margin * std::min(1.0, d.candidates[win].support);
    d.needs_review = d.confidence < review_below;
    d.decider = "rules+jev";
}

std::string DateDecision::to_json() const {
    json j;
    j["when"] = when.valid() ? when.pretty() : "";
    j["confidence"] = std::round(confidence * 1000) / 1000;
    j["margin"] = std::round(margin * 1000) / 1000;
    j["source"] = source;
    j["decider"] = decider;
    j["needs_review"] = needs_review;
    if (!jev_note.empty()) j["jev"] = jev_note;
    j["alternatives"] = alternatives;
    json arr = json::array();
    for (auto& c : candidates) {
        json e = {{"src", date_source_id(c.src)}, {"when", c.when.valid() ? c.when.pretty() : ""},
                  {"base", std::round(c.base * 1000) / 1000}, {"weight", std::round(c.weight * 1000) / 1000},
                  {"support", std::round(c.support * 1000) / 1000}, {"raw", c.raw}};
        if (!c.notes.empty()) e["notes"] = c.notes;
        arr.push_back(e);
    }
    j["candidates"] = arr;
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}
