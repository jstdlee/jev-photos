// Small shared helpers: strings, naive civil date-times, hashing, base64, processes, files.
#pragma once

#include "compat.h"

#include <cstdint>
#include <string>
#include <vector>

namespace util {

// ---- strings
std::string trim(const std::string& s);
std::string lower(const std::string& s);  // ASCII only; UTF-8 bytes pass through
std::vector<std::string> split(const std::string& s, char sep);
bool starts_with(const std::string& s, const std::string& p);
bool ends_with(const std::string& s, const std::string& p);
std::string replace_all(std::string s, const std::string& from, const std::string& to);
size_t utf8_len(const std::string& s);
bool has_cjk(const std::string& s);
bool is_digits(const std::string& s);
std::string sh_quote(const std::string& s);  // single-quote for /bin/sh
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));

// ---- naive (wall-clock, zone-less) date-times, as photo metadata stores them
enum Precision { P_NONE = 0, P_YEAR, P_MONTH, P_DAY, P_MINUTE, P_SECOND };
const char* precision_name(Precision p);
Precision precision_from(const std::string& s);

struct Civil {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    Precision prec = P_NONE;
    bool valid() const { return prec != P_NONE; }
    int64_t secs() const;                 // seconds since 1970-01-01 00:00 on the naive clock
    std::string str() const;              // "YYYY-MM-DD HH:MM:SS" (lower fields zeroed by precision)
    std::string pretty() const;           // honours precision: "2019-05", "2019-05-12", ...
    std::string exif() const;             // "YYYY:MM:DD HH:MM:SS"
};
int64_t days_from_civil(int y, int m, int d);
void civil_from_days(int64_t z, int& y, int& m, int& d);
Civil civil_from_secs(int64_t s, Precision p);
bool valid_date(int y, int m, int d);
Civil make_civil(int y, int mo, int d, int h, int mi, int s, Precision p);  // returns invalid on bad fields
Civil parse_exif_datetime(const std::string& v);  // "YYYY:MM:DD HH:MM:SS", "YYYY-MM-DDTHH:MM:SS+08:00", "YYYY-MM-DD"
Civil parse_db_datetime(const std::string& v);     // Civil::str() format + precision name
Civil local_from_epoch(int64_t epoch_utc);          // system timezone
int64_t now_epoch();
std::string now_iso();

// ---- hashing / encoding
std::string sha256_file(const std::string& path);  // hex; empty on error
// Fast duplicate detection (xxHash XXH3): quick = size + first/last 64 KiB (reads <= 128 KiB),
// full = XXH3-128 of the whole file. Both hex; empty on error.
std::string quick_hash(const std::string& path, int64_t size);
std::string content_hash(const std::string& path);
bool files_equal(const std::string& a, const std::string& b);  // byte-for-byte
std::string human_size(int64_t bytes);                          // "4.2 MB"
std::string base64(const unsigned char* data, size_t n);

// ---- files / processes
bool file_exists(const std::string& p);
bool dir_exists(const std::string& p);
bool read_file(const std::string& p, std::string& out);
bool write_file(const std::string& p, const std::string& data, int mode = 0644);
std::string home();
std::string config_dir();  // ~/.config/jev-photos, %APPDATA%\jev-photos (created)
std::string data_dir();    // ~/.local/share/jev-photos, %LOCALAPPDATA%\jev-photos (created)
std::string cache_dir();   // ~/.cache/jev-photos, %LOCALAPPDATA%\jev-photos\cache (created)
// A path as this app stores it: forward slashes on every system (Windows accepts them), no trailing slash.
std::string norm_path(const std::string& p);
std::string canonical(const std::string& p);  // absolute, symlinks resolved where they exist, norm_path form
// Keep a file's modification time (and access time) as they were, after rewriting it.
void set_file_times(const std::string& p, int64_t atime, int64_t mtime);
std::string basename(const std::string& p);
std::string dirname(const std::string& p);
std::string stem(const std::string& p);
std::string ext_lower(const std::string& p);  // "jpg" (no dot)
bool mkdirs(const std::string& p);
bool copy_file_preserve(const std::string& from, const std::string& to, std::string& err);  // never overwrites
std::string temp_path(const std::string& tag);  // unique path under $TMPDIR

struct ProcResult {
    int rc = -1;
    std::string out, err;
};
// Runs argv[0] with args directly (no shell), capturing stdout/stderr, optional stdin.
ProcResult run(const std::vector<std::string>& argv, const std::string& stdin_data = "", int timeout_s = 120);
bool which(const std::string& exe);  // on PATH (Windows: also next to the app, .exe added)

// ---- desktop
bool have_trash();                                              // a restorable Trash / Recycle Bin is available
bool trash(const std::string& path);                            // move one file there
void open_path(const std::string& path_or_url);                 // default app / file manager / browser (detached)
std::string choose_folder(const std::string& title);           // folder dialog; "" when cancelled (blocking)
#ifdef _WIN32
std::wstring widen(const std::string& utf8);
void attach_console();  // stdout/stderr to the parent's console, if there is one
std::string narrow(const std::wstring& w);
#endif

}  // namespace util
