#include "meta.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <set>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

static const char* kJevNs = "http://ns.jev.local/photos/1.0/";

std::string meta_description(const MetaMap& m) {
    std::string v = m.get("Xmp.dc.description");
    if (util::starts_with(v, "lang=")) {
        size_t sp = v.find(' ');
        v = sp == std::string::npos ? "" : v.substr(sp + 1);
    }
    return util::trim(v);
}

std::vector<std::string> MetaMap::list(const std::string& k) const {
    std::vector<std::string> out;
    auto it = kv.find(k);
    if (it == kv.end()) return out;
    auto t = type.find(k);
    bool bag = t != type.end() && (t->second == "XmpBag" || t->second == "XmpSeq");
    for (auto& v : it->second) {
        if (bag) {
            for (auto& part : util::split(v, ','))
                if (auto s = util::trim(part); !s.empty()) out.push_back(s);
        } else if (auto s = util::trim(v); !s.empty()) {
            out.push_back(s);
        }
    }
    return out;
}

std::string MetaMap::to_json() const {
    json j = json::object();
    for (auto& [k, vs] : kv) {
        json arr = json::array();
        for (auto& v : vs) arr.push_back(json(v));  // exiv2 may emit invalid UTF-8; replaced on dump
        auto t = type.find(k);
        j[k] = {{"t", t == type.end() ? "" : t->second}, {"v", arr}};
    }
    if (!error.empty()) j["__error"] = error;
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

MetaMap MetaMap::from_json(const std::string& s) {
    MetaMap m;
    json j = json::parse(s, nullptr, false);
    if (!j.is_object()) return m;
    for (auto& [k, v] : j.items()) {
        if (k == "__error") { m.error = v.get<std::string>(); continue; }
        m.type[k] = v.value("t", "");
        for (auto& x : v["v"]) m.kv[k].push_back(x.get<std::string>());
    }
    return m;
}

bool exiv2_available() { return util::which("exiv2"); }

MetaMap read_meta(const std::string& path) {
    MetaMap m;
    // -Pkyv: key, type, raw value. -q silences warnings about odd makernotes.
    util::ProcResult r = util::run({"exiv2", "-q", "-Pkyv", "--", path}, "", 60);
    if (r.rc == 127) { m.error = "exiv2 not found"; return m; }
    for (auto& line : util::split(r.out, '\n')) {
        if (line.empty()) continue;
        size_t k_end = line.find(' ');
        if (k_end == std::string::npos) continue;
        std::string key = line.substr(0, k_end);
        size_t t_start = line.find_first_not_of(' ', k_end);
        if (t_start == std::string::npos) continue;
        size_t t_end = line.find(' ', t_start);
        std::string type = line.substr(t_start, t_end == std::string::npos ? std::string::npos : t_end - t_start);
        std::string value;
        if (t_end != std::string::npos) {
            size_t v_start = line.find_first_not_of(' ', t_end);
            if (v_start != std::string::npos) value = line.substr(v_start);
        }
        // Skip bulky binary blobs (maker notes, thumbnails); they are useless for decisions and bloat the DB.
        if (type == "Undefined" && value.size() > 200) continue;
        if (value.size() > 4000) value = value.substr(0, 4000);
        m.kv[key].push_back(value);
        m.type[key] = type;
    }
    // exiv2 exits non-zero both for "no metadata" (fine) and for unreadable/unsupported files.
    if (r.rc != 0 && m.kv.empty()) {
        std::string e = util::trim(r.err);
        if (e.find("No Exif data") == std::string::npos && !e.empty()) m.error = e.substr(0, 300);
    }
    return m;
}

static double rational(const std::string& s) {
    long a = 0, b = 1;
    if (sscanf(s.c_str(), "%ld/%ld", &a, &b) == 2 && b != 0) return double(a) / double(b);
    return atof(s.c_str());
}

static bool dms(const std::string& v, double& out) {
    auto parts = util::split(util::trim(v), ' ');
    if (parts.size() < 3) return false;
    out = rational(parts[0]) + rational(parts[1]) / 60.0 + rational(parts[2]) / 3600.0;
    return std::isfinite(out);
}

bool meta_gps(const MetaMap& m, double& lat, double& lon) {
    if (!m.has("Exif.GPSInfo.GPSLatitude") || !m.has("Exif.GPSInfo.GPSLongitude")) return false;
    if (!dms(m.get("Exif.GPSInfo.GPSLatitude"), lat) || !dms(m.get("Exif.GPSInfo.GPSLongitude"), lon)) return false;
    if (util::trim(m.get("Exif.GPSInfo.GPSLatitudeRef")) == "S") lat = -lat;
    if (util::trim(m.get("Exif.GPSInfo.GPSLongitudeRef")) == "W") lon = -lon;
    // (0,0) is the classic "no fix" value.
    if (std::fabs(lat) < 1e-6 && std::fabs(lon) < 1e-6) return false;
    return std::fabs(lat) <= 90 && std::fabs(lon) <= 180;
}

bool meta_gps_time(const MetaMap& m, int64_t& epoch) {
    util::Civil d = util::parse_exif_datetime(m.get("Exif.GPSInfo.GPSDateStamp"));
    if (!d.valid() || d.prec < util::P_DAY) return false;
    auto t = util::split(util::trim(m.get("Exif.GPSInfo.GPSTimeStamp")), ' ');
    if (t.size() < 3) return false;
    int h = int(rational(t[0])), mi = int(rational(t[1])), s = int(rational(t[2]));
    util::Civil c = util::make_civil(d.y, d.mo, d.d, h, mi, s, util::P_SECOND);
    if (!c.valid()) return false;
    epoch = c.secs();  // the naive clock of a UTC reading is epoch time
    return true;
}

bool meta_offset(const MetaMap& m, int& offset_s) {
    for (const char* k : {"Exif.Photo.OffsetTimeOriginal", "Exif.Photo.OffsetTime", "Exif.Photo.OffsetTimeDigitized"}) {
        std::string v = util::trim(m.get(k));
        int hh = 0, mm = 0;
        char sign = 0;
        if (v.size() >= 6 && sscanf(v.c_str(), "%c%2d:%2d", &sign, &hh, &mm) == 3 && (sign == '+' || sign == '-')) {
            offset_s = (hh * 3600 + mm * 60) * (sign == '-' ? -1 : 1);
            return true;
        }
    }
    return false;
}

int meta_orientation(const MetaMap& m) {
    int o = atoi(m.get("Exif.Image.Orientation").c_str());
    return o >= 1 && o <= 8 ? o : 1;
}

std::string sidecar_path(const std::string& image_path) {
    std::string d = util::dirname(image_path);
    return (d.empty() ? "" : d + "/") + util::stem(image_path) + ".xmp";
}

static std::string clean_value(std::string v, size_t max_bytes) {
    for (auto& c : v)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    v = util::trim(v);
    if (v.size() > max_bytes) {
        size_t cut = max_bytes;
        while (cut > 0 && (static_cast<unsigned char>(v[cut]) & 0xC0) == 0x80) cut--;  // stay on a UTF-8 boundary
        v = v.substr(0, cut);
    }
    return v;
}

static std::string xmp_date(const util::Civil& c) {
    switch (c.prec) {
        case util::P_YEAR: return util::fmt("%04d", c.y);
        case util::P_MONTH: return util::fmt("%04d-%02d", c.y, c.mo);
        case util::P_DAY: return util::fmt("%04d-%02d-%02d", c.y, c.mo, c.d);
        case util::P_MINUTE: return util::fmt("%04d-%02d-%02dT%02d:%02d", c.y, c.mo, c.d, c.h, c.mi);
        default: return util::fmt("%04d-%02d-%02dT%02d:%02d:%02d", c.y, c.mo, c.d, c.h, c.mi, c.s);
    }
}

WriteResult write_meta(const std::string& image_path, const MetaMap& existing, const MetaPlan& p, MetaTarget target, bool dry_run) {
    WriteResult r;
    r.target = target == MetaTarget::Sidecar ? sidecar_path(image_path) : image_path;
    bool embed = target == MetaTarget::Embed;
    std::vector<std::string>& cmd = r.commands;
    cmd.push_back(std::string("reg jev ") + kJevNs);

    // Rule: a key that already has a value is never touched (Xmp.jev.* is ours and may be refreshed).
    auto absent = [&](const std::string& k) { return !existing.has(k); };

    // --- date
    if (p.date.valid() && p.date_trusted) {
        bool has_original = existing.has("Exif.Photo.DateTimeOriginal") || existing.has("Xmp.exif.DateTimeOriginal");
        if (embed && !has_original && p.date.prec >= util::P_MINUTE)
            cmd.push_back("set Exif.Photo.DateTimeOriginal Ascii " + p.date.exif());
        if (!embed && !has_original && p.date.prec >= util::P_MINUTE)
            cmd.push_back("set Xmp.exif.DateTimeOriginal XmpText " + xmp_date(p.date));
        if (absent("Xmp.photoshop.DateCreated"))
            cmd.push_back("set Xmp.photoshop.DateCreated XmpText " + xmp_date(p.date));
    }
    if (p.date.valid()) {
        cmd.push_back("set Xmp.jev.ResolvedDate XmpText " + xmp_date(p.date));
        cmd.push_back("set Xmp.jev.DateSource XmpText " + clean_value(p.date_source, 200));
        cmd.push_back("set Xmp.jev.DateConfidence XmpText " + util::fmt("%.2f", p.date_conf));
    }

    // --- description
    std::string cap = clean_value(p.caption, 2000);
    if (!cap.empty()) cmd.push_back("set Xmp.jev.Caption XmpText " + cap);
    std::string ours = clean_value(p.description.empty() ? p.caption : p.description, 2000);
    if (!ours.empty()) {
        std::string E = clean_value(meta_description(existing), 4000), prev = clean_value(existing.get("Xmp.jev.Description"), 4000);
        static const std::string sep = " \xE2\x80\x94 ";  // " — "
        std::string next;
        if (E.empty()) { next = ours; r.desc_change = "add"; }
        else if (!prev.empty() && E == prev) { if (p.rewrite_keywords && ours != E) { next = ours; r.desc_change = "update"; } }
        else if (!prev.empty() && E.size() > prev.size() + sep.size() && E.compare(E.size() - prev.size(), prev.size(), prev) == 0 &&
                 E.compare(E.size() - prev.size() - sep.size(), sep.size(), sep) == 0) {
            if (p.rewrite_keywords && ours != prev) { next = E.substr(0, E.size() - prev.size()) + ours; r.desc_change = "update"; }
        } else if (E.find(ours) != std::string::npos) {
            // already there
        } else if (p.desc_action == "append") {
            next = clean_value(E + sep + ours, 6000);
            r.desc_change = "append";
        } else if (p.desc_action == "replace") {
            next = ours;
            r.desc_change = "replace";
            if (!existing.has("Xmp.jev.PreviousDescription")) cmd.push_back("set Xmp.jev.PreviousDescription XmpText " + E);
        }
        if (!next.empty()) {
            cmd.push_back("set Xmp.dc.description LangAlt lang=\"x-default\" " + next);
            cmd.push_back("set Xmp.jev.Description XmpText " + ours);
        }
    }
    if (auto s = clean_value(p.scene, 200); !s.empty()) cmd.push_back("set Xmp.jev.Scene XmpText " + s);
    if (auto s = clean_value(p.landmark, 200); !s.empty()) cmd.push_back("set Xmp.jev.Landmark XmpText " + s);

    // --- location
    std::string loc = clean_value(p.location, 200);
    if (!loc.empty()) {
        bool has_loc = existing.has("Xmp.iptc.Location") || existing.has("Xmp.photoshop.City") ||
                       existing.has("Iptc.Application2.City") || existing.has("Iptc.Application2.SubLocation");
        if (!has_loc) cmd.push_back("set Xmp.iptc.Location XmpText " + loc);
        cmd.push_back("set Xmp.jev.Location XmpText " + loc);
        cmd.push_back("set Xmp.jev.LocationSource XmpText " + clean_value(p.location_source, 100));
    }

    // --- keywords: append only the ones not already present (case-insensitive)
    std::set<std::string> have_xmp, have_iptc, wanted;
    std::vector<std::string> current = existing.list("Xmp.dc.subject");
    for (auto& k : current) have_xmp.insert(util::lower(k));
    for (auto& k : existing.list("Iptc.Application2.Keywords")) have_iptc.insert(util::lower(k));
    for (auto& k : p.keywords) wanted.insert(util::lower(clean_value(util::replace_all(k, ",", " "), 64)));
    // IPTC IIM is legacy; extend it only where the file already uses it, so we never create an IPTC block
    // with an undeclared character set.
    bool use_iptc = embed && !have_iptc.empty();
    bool first_bag = !existing.has("Xmp.dc.subject");
    // Which keywords are ours from an earlier run (and so may be replaced)?
    std::set<std::string> ours_before;
    for (auto& k : util::split(existing.get("Xmp.jev.Keywords"), ','))  // XmpText "a, b"
        if (!util::trim(k).empty()) ours_before.insert(util::lower(util::trim(k)));
    for (auto& k : p.previous_keywords) ours_before.insert(util::lower(k));
    std::set<std::string> theirs;  // keywords the file carries independently of us
    for (auto& k : current)
        if (!ours_before.count(util::lower(k))) theirs.insert(util::lower(k));
    if (p.rewrite_keywords && !ours_before.empty()) {
        std::vector<std::string> keep;
        for (auto& k : current) {
            std::string lk = util::lower(k);
            if (!ours_before.count(lk) || wanted.count(lk)) keep.push_back(k);
            else r.removed.push_back(k);
        }
        if (!r.removed.empty()) {
            // exiv2 cannot drop one item of a bag: delete the bag, then put back everything that stays.
            cmd.push_back("del Xmp.dc.subject");
            have_xmp.clear();
            first_bag = true;
            r.kept = keep;
            for (auto& k : keep) {
                cmd.push_back(std::string("set Xmp.dc.subject ") + (first_bag ? "XmpBag " : "") + k);
                first_bag = false;
                have_xmp.insert(util::lower(k));
            }
        }
    }
    std::string ours_now;
    for (auto& raw : p.keywords) {
        // exiv2 splits XmpBag values on ", " so commas cannot appear inside one keyword.
        std::string k = clean_value(util::replace_all(raw, ",", " "), 64);
        if (k.empty()) continue;
        std::string lk = util::lower(k);
        if (!theirs.count(lk) && ours_now.find(k) == std::string::npos) ours_now += (ours_now.empty() ? "" : ", ") + k;
        if (!have_xmp.count(lk)) {
            cmd.push_back(std::string("set Xmp.dc.subject ") + (first_bag ? "XmpBag " : "") + k);
            first_bag = false;
            have_xmp.insert(lk);
        }
        if (use_iptc && !have_iptc.count(lk)) {
            cmd.push_back("add Iptc.Application2.Keywords String " + k);
            have_iptc.insert(lk);
        }
    }
    // Remember which keywords are ours, so a later rewrite can replace exactly those.
    if (!ours_now.empty()) cmd.push_back("set Xmp.jev.Keywords XmpText " + ours_now);

    // exiv2 rejects "set Key Type" with an empty value; drop those.
    cmd.erase(std::remove_if(cmd.begin() + 1, cmd.end(), [](const std::string& c) {
                  if (util::starts_with(c, "del ")) return false;
                  auto parts = util::split(c, ' ');
                  bool typed = parts.size() >= 3 && (parts[2] == "XmpText" || parts[2] == "XmpBag" || parts[2] == "Ascii" || parts[2] == "String");
                  return parts.size() < 3 || (typed && util::trim(c.substr(c.find(parts[2]) + parts[2].size())).empty());
              }),
              cmd.end());
    // Count what really changes: new fields and keywords (a bag rebuilt after a rewrite does not count its kept items).
    for (auto& c : cmd)
        if (!util::starts_with(c, "reg ") && !util::starts_with(c, "set Xmp.jev.") && !util::starts_with(c, "del ")) r.added++;
    if (!r.removed.empty()) r.added -= int(current.size() - r.removed.size());
    if (cmd.size() <= 1) { r.ok = true; return r; }
    if (dry_run) { r.ok = true; return r; }

    if (!embed && !util::file_exists(r.target)) {
        // Fresh sidecar: an empty XMP packet that exiv2 can then extend.
        static const char* empty_packet =
            "<?xpacket begin=\"\xEF\xBB\xBF\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
            "<rdf:Description rdf:about=\"\"/></rdf:RDF></x:xmpmeta>\n<?xpacket end=\"w\"?>\n";
        if (!util::write_file(r.target, empty_packet)) { r.error = "cannot create sidecar " + r.target; return r; }
    }

    struct stat before{};
    bool have_times = stat(r.target.c_str(), &before) == 0;
    std::string cmd_path = util::temp_path("exiv2.cmd");
    std::string body;
    for (auto& c : cmd) body += c + "\n";
    util::write_file(cmd_path, body, 0600);
    util::ProcResult pr = util::run({"exiv2", "-q", "-m", cmd_path, "--", r.target}, "", 120);
    unlink(cmd_path.c_str());
    if (have_times && embed) {  // metadata edits should not look like a new file version
        util::set_file_times(r.target, ST_ATIME(before), ST_MTIME(before));
    }
    if (pr.rc != 0) {
        r.error = util::trim(pr.err).substr(0, 400);
        if (r.error.empty()) r.error = "exiv2 exit " + std::to_string(pr.rc);
        return r;
    }
    r.ok = true;
    return r;
}
