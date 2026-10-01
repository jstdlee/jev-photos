// Image metadata through the exiv2 CLI: read everything, write only what is missing (never overwrite).
#pragma once

#include <map>
#include <string>
#include <vector>

#include "util.h"

struct MetaMap {
    std::map<std::string, std::vector<std::string>> kv;  // key -> values (IPTC datasets repeat)
    std::map<std::string, std::string> type;             // key -> exiv2 type name
    std::string error;                                    // set when exiv2 could not read the file

    bool has(const std::string& k) const { auto it = kv.find(k); return it != kv.end() && !it->second.empty() && !util::trim(it->second[0]).empty(); }
    std::string get(const std::string& k) const { auto it = kv.find(k); return it == kv.end() || it->second.empty() ? "" : it->second[0]; }
    std::vector<std::string> list(const std::string& k) const;  // XmpBag "a, b" and repeated IPTC split into items
    std::string to_json() const;
    static MetaMap from_json(const std::string& j);
};

bool exiv2_available();
MetaMap read_meta(const std::string& path);

// GPS in decimal degrees from Exif.GPSInfo.*; false when absent/invalid.
bool meta_gps(const MetaMap& m, double& lat, double& lon);
// UTC GPS timestamp (GPSDateStamp + GPSTimeStamp) as epoch seconds; false when absent.
bool meta_gps_time(const MetaMap& m, int64_t& epoch_utc);
// "+08:00" style offset in seconds from Exif.Photo.OffsetTimeOriginal/OffsetTime.
bool meta_offset(const MetaMap& m, int& offset_s);
int meta_orientation(const MetaMap& m);  // 1..8, 1 when absent

// Everything we might add. Empty/invalid members are skipped.
struct MetaPlan {
    util::Civil date;
    double date_conf = 0;
    std::string date_source;
    bool date_trusted = false;  // confidence above the review threshold
    std::string caption, scene, landmark, location, location_source;
    std::vector<std::string> keywords;
    // Rewrite: keywords jev-photos itself wrote earlier (listed in Xmp.jev.Keywords, or given here) that are no longer
    // among `keywords` are removed. Keywords the file had on its own are never touched.
    bool rewrite_keywords = false;
    std::vector<std::string> previous_keywords;
    // Description (Xmp.dc.description). An empty field is filled. One that already has text is changed only as
    // desc_action says: "append" (after it, " — " between), "replace" (the old text is kept in
    // Xmp.jev.PreviousDescription), anything else leaves it. Text jev-photos wrote itself (Xmp.jev.Description) is
    // refreshed when rewrite_keywords is on.
    std::string description, desc_action;
};
std::string meta_description(const MetaMap& m);  // Xmp.dc.description text without its lang="..." prefix

struct WriteResult {
    bool ok = false;
    int added = 0;               // number of fields/items added
    std::vector<std::string> removed;  // our own earlier keywords taken out (rewrite)
    std::vector<std::string> kept;     // keywords put back after the bag was rebuilt (not new)
    std::string desc_change;           // "", "add", "append", "replace", "update" (our own earlier text)
    std::string error, target;   // target = file actually modified (image or sidecar)
    std::vector<std::string> commands;
};

enum class MetaTarget { Embed, Sidecar };
// existing = metadata of the destination file as it is now (read again just before writing).
WriteResult write_meta(const std::string& image_path, const MetaMap& existing, const MetaPlan& plan, MetaTarget target, bool dry_run);
std::string sidecar_path(const std::string& image_path);
