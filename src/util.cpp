#include "util.h"

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <sys/utime.h>
#else
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

namespace fs = std::filesystem;
#ifndef _WIN32
extern char** environ;
#endif

namespace util {

// ---------------------------------------------------------------------------
// strings

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) a++;
    while (b > a && isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

std::string lower(const std::string& s) {
    std::string r = s;
    for (auto& c : r)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return r;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); i++)
        if (i == s.size() || s[i] == sep) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    return out;
}

bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) s.replace(pos, from.size(), to);
    return s;
}

size_t utf8_len(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80) n++;
    return n;
}

bool has_cjk(const std::string& s) {
    // CJK Unified Ideographs, Hiragana/Katakana and Hangul all start with lead bytes 0xE3..0xED.
    for (unsigned char c : s)
        if (c >= 0xE3 && c <= 0xED) return true;
    return false;
}

bool is_digits(const std::string& s) {
    if (s.empty()) return false;
    for (unsigned char c : s)
        if (!isdigit(c)) return false;
    return true;
}

std::string sh_quote(const std::string& s) { return "'" + replace_all(s, "'", "'\\''") + "'"; }

std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    char buf[1024];
    int n = vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    if (n < (int)sizeof buf) return std::string(buf, n < 0 ? 0 : n);
    std::string big(n + 1, '\0');
    va_start(ap, f);
    vsnprintf(big.data(), big.size(), f, ap);
    va_end(ap);
    big.resize(n);
    return big;
}

// ---------------------------------------------------------------------------
// civil time (Howard Hinnant's algorithms)

int64_t days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + int64_t(doe) - 719468;
}

void civil_from_days(int64_t z, int& y, int& m, int& d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = unsigned(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = int(doy - (153 * mp + 2) / 5 + 1);
    m = int(mp < 10 ? mp + 3 : mp - 9);
    y = int(yoe + era * 400 + (m <= 2));
}

const char* precision_name(Precision p) {
    switch (p) {
        case P_YEAR: return "year";
        case P_MONTH: return "month";
        case P_DAY: return "day";
        case P_MINUTE: return "minute";
        case P_SECOND: return "second";
        default: return "none";
    }
}

Precision precision_from(const std::string& s) {
    if (s == "year") return P_YEAR;
    if (s == "month") return P_MONTH;
    if (s == "day") return P_DAY;
    if (s == "minute") return P_MINUTE;
    if (s == "second") return P_SECOND;
    return P_NONE;
}

bool valid_date(int y, int m, int d) {
    static const int mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (y < 1800 || y > 2200 || m < 1 || m > 12 || d < 1) return false;
    int md = mdays[m - 1] + (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0));
    return d <= md;
}

Civil make_civil(int y, int mo, int d, int h, int mi, int s, Precision p) {
    Civil c;
    if (p >= P_MONTH && (mo < 1 || mo > 12)) return c;
    if (p >= P_DAY && !valid_date(y, mo, d)) return c;
    if (p < P_DAY && (y < 1800 || y > 2200)) return c;
    if (p >= P_MINUTE && (h < 0 || h > 23 || mi < 0 || mi > 59)) return c;
    if (p >= P_SECOND && (s < 0 || s > 60)) return c;
    c.y = y;
    c.mo = p >= P_MONTH ? mo : 1;
    c.d = p >= P_DAY ? d : 1;
    c.h = p >= P_MINUTE ? h : 0;
    c.mi = p >= P_MINUTE ? mi : 0;
    c.s = p >= P_SECOND ? std::min(s, 59) : 0;
    c.prec = p;
    return c;
}

int64_t Civil::secs() const { return days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s; }

std::string Civil::str() const { return fmt("%04d-%02d-%02d %02d:%02d:%02d", y, mo, d, h, mi, s); }

std::string Civil::pretty() const {
    switch (prec) {
        case P_YEAR: return fmt("%04d", y);
        case P_MONTH: return fmt("%04d-%02d", y, mo);
        case P_DAY: return fmt("%04d-%02d-%02d", y, mo, d);
        case P_MINUTE: return fmt("%04d-%02d-%02d %02d:%02d", y, mo, d, h, mi);
        case P_SECOND: return str();
        default: return "";
    }
}

std::string Civil::exif() const { return fmt("%04d:%02d:%02d %02d:%02d:%02d", y, mo, d, h, mi, s); }

Civil civil_from_secs(int64_t t, Precision p) {
    int64_t days = t >= 0 ? t / 86400 : (t - 86399) / 86400;
    int64_t rem = t - days * 86400;
    Civil c;
    civil_from_days(days, c.y, c.mo, c.d);
    c.h = int(rem / 3600);
    c.mi = int(rem % 3600 / 60);
    c.s = int(rem % 60);
    c.prec = p;
    return make_civil(c.y, c.mo, c.d, c.h, c.mi, c.s, p);
}

Civil parse_exif_datetime(const std::string& raw) {
    std::string v = trim(raw);
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    char a, b, c, e, f;
    int n = sscanf(v.c_str(), "%4d%c%2d%c%2d%c%2d%c%2d%c%2d", &y, &a, &mo, &b, &d, &c, &h, &e, &mi, &f, &s);
    if (n >= 5 && (a == ':' || a == '-') && a == b) {
        if (n >= 11) return make_civil(y, mo, d, h, mi, s, P_SECOND);
        if (n >= 9) return make_civil(y, mo, d, h, mi, 0, P_MINUTE);
        return make_civil(y, mo, d, 0, 0, 0, P_DAY);
    }
    if (n >= 3 && (a == ':' || a == '-')) return make_civil(y, mo, 1, 0, 0, 0, P_MONTH);
    if (n >= 1 && v.size() == 4 && is_digits(v)) return make_civil(y, 1, 1, 0, 0, 0, P_YEAR);
    return {};
}

Civil parse_db_datetime(const std::string& v) {
    auto bar = v.find('|');
    Civil c = parse_exif_datetime(v.substr(0, bar));
    if (c.valid() && bar != std::string::npos) c = make_civil(c.y, c.mo, c.d, c.h, c.mi, c.s, precision_from(v.substr(bar + 1)));
    return c;
}

Civil local_from_epoch(int64_t epoch) {
    time_t t = (time_t)epoch;
    struct tm tm{};
    localtime_r(&t, &tm);
    return make_civil(tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, P_SECOND);
}

int64_t now_epoch() { return (int64_t)time(nullptr); }

std::string now_iso() {
    Civil c = local_from_epoch(now_epoch());
    return c.str();
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), streaming over a file

namespace {
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char buf[64];
    size_t blen = 0;
    uint64_t total = 0;

    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = uint32_t(p[i * 4]) << 24 | uint32_t(p[i * 4 + 1]) << 16 | uint32_t(p[i * 4 + 2]) << 8 | p[i * 4 + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const unsigned char* p, size_t n) {
        total += n;
        while (n) {
            size_t take = std::min(n, 64 - blen);
            memcpy(buf + blen, p, take);
            blen += take; p += take; n -= take;
            if (blen == 64) { block(buf); blen = 0; }
        }
    }
    std::string hex() {
        uint64_t bits = total * 8;
        unsigned char pad = 0x80, zero = 0;
        update(&pad, 1);
        while (blen != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - 8 * i));
        update(len, 8);
        std::string out;
        for (uint32_t v : h) out += fmt("%08x", v);
        return out;
    }
};
}  // namespace

std::string sha256_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    Sha256 s;
    std::vector<unsigned char> buf(1 << 16);
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) s.update(buf.data(), n);
    bool err = ferror(f);
    fclose(f);
    return err ? "" : s.hex();
}

std::string quick_hash(const std::string& path, int64_t size) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return "";
    const int64_t kChunk = 64 * 1024;
    std::vector<unsigned char> buf(size_t(std::min<int64_t>(size, 2 * kChunk)) + 8);
    memcpy(buf.data(), &size, 8);  // the size is part of the key: same head/tail but different length never match
    size_t n = 8;
    auto rd = [&](int64_t off, int64_t len) {
        while (len > 0) {
#ifdef _WIN32
            if (_lseeki64(fd, off, SEEK_SET) != off) return false;
            int r = _read(fd, buf.data() + n, unsigned(len));
#else
            ssize_t r = pread(fd, buf.data() + n, size_t(len), off);
#endif
            if (r <= 0) return false;
            n += size_t(r); off += r; len -= r;
        }
        return true;
    };
    bool ok = size <= 2 * kChunk ? rd(0, size) : (rd(0, kChunk) && rd(size - kChunk, kChunk));
    close(fd);
    if (!ok) return "";
    return fmt("%016llx", (unsigned long long)XXH3_64bits(buf.data(), n));
}

std::string content_hash(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return "";
#ifndef _WIN32
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    XXH3_state_t* st = XXH3_createState();
    XXH3_128bits_reset(st);
    std::vector<char> buf(1 << 20);
    bool ok = true;
    for (;;) {
        auto r = read(fd, buf.data(), (unsigned)buf.size());
        if (r == 0) break;
        if (r < 0) { if (errno == EINTR) continue; ok = false; break; }
        XXH3_128bits_update(st, buf.data(), size_t(r));
    }
    close(fd);
    XXH128_hash_t h = XXH3_128bits_digest(st);
    XXH3_freeState(st);
    return ok ? fmt("%016llx%016llx", (unsigned long long)h.high64, (unsigned long long)h.low64) : "";
}

bool files_equal(const std::string& a, const std::string& b) {
    int fa = open(a.c_str(), O_RDONLY | O_CLOEXEC), fb = open(b.c_str(), O_RDONLY | O_CLOEXEC);
    bool eq = fa >= 0 && fb >= 0;
    struct stat sa{}, sb{};
    if (eq) eq = fstat(fa, &sa) == 0 && fstat(fb, &sb) == 0 && sa.st_size == sb.st_size;
    std::vector<char> x(1 << 20), y(1 << 20);
    while (eq) {
        auto ra = read(fa, x.data(), (unsigned)x.size());
        if (ra <= 0) { eq = ra == 0; break; }
        decltype(ra) got = 0;
        while (got < ra) {
            auto rb = read(fb, y.data() + got, (unsigned)(ra - got));
            if (rb <= 0) { eq = false; break; }
            got += rb;
        }
        if (eq) eq = memcmp(x.data(), y.data(), size_t(ra)) == 0;
    }
    if (fa >= 0) close(fa);
    if (fb >= 0) close(fb);
    return eq;
}

std::string human_size(int64_t b) {
    const char* u[] = {"B", "KB", "MB", "GB", "TB"};
    double v = double(b);
    int i = 0;
    while (v >= 1024 && i < 4) { v /= 1024; i++; }
    return i == 0 ? fmt("%lld B", (long long)b) : fmt(v < 10 ? "%.1f %s" : "%.0f %s", v, u[i]);
}

std::string base64(const unsigned char* d, size_t n) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = uint32_t(d[i]) << 16 | (i + 1 < n ? uint32_t(d[i + 1]) << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        out += t[v >> 18 & 63];
        out += t[v >> 12 & 63];
        out += i + 1 < n ? t[v >> 6 & 63] : '=';
        out += i + 2 < n ? t[v & 63] : '=';
    }
    return out;
}

// ---------------------------------------------------------------------------
// files

bool file_exists(const std::string& p) {
    struct stat st{};
    return !p.empty() && stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const std::string& p) {
    struct stat st{};
    return !p.empty() && stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool read_file(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

bool write_file(const std::string& p, const std::string& data, int mode) {
    int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, mode);
    if (fd < 0) return false;
    size_t off = 0;
    while (off < data.size()) {
        auto w = write(fd, data.data() + off, (unsigned)(data.size() - off));
        if (w <= 0) { close(fd); return false; }
        off += size_t(w);
    }
    return close(fd) == 0;
}

static std::string env(const char* k) {
    const char* v = getenv(k);
    return v && *v ? norm_path(v) : "";
}

std::string home() {
#ifdef _WIN32
    std::string h = env("USERPROFILE");
#else
    std::string h = env("HOME");
#endif
    return h.empty() ? "." : h;
}

static std::string app_dir(const char* xdg, const char* xdg_default, const char* win_env, const char* win_sub) {
#ifdef _WIN32
    (void)xdg; (void)xdg_default;
    std::string base = env(win_env);
    std::string d = (base.empty() ? home() : base) + "/jev-photos" + win_sub;
#else
    (void)win_env; (void)win_sub;
    std::string base = env(xdg);
    std::string d = (base.empty() ? home() + xdg_default : base) + "/jev-photos";
#endif
    mkdirs(d);
    return d;
}
std::string config_dir() { return app_dir("XDG_CONFIG_HOME", "/.config", "APPDATA", ""); }
std::string data_dir() { return app_dir("XDG_DATA_HOME", "/.local/share", "LOCALAPPDATA", ""); }
std::string cache_dir() { return app_dir("XDG_CACHE_HOME", "/.cache", "LOCALAPPDATA", "/cache"); }

std::string norm_path(const std::string& p) {
    std::string r = p;
#ifdef _WIN32
    for (auto& c : r)
        if (c == '\\') c = '/';
#endif
    while (r.size() > 1 && r.back() == '/' && !(r.size() == 3 && r[1] == ':')) r.pop_back();
    return r;
}

// fs::path from/to UTF-8 (on Windows a plain std::string would go through the ANSI code page).
static fs::path upath(const std::string& p) { return fs::u8path(p); }
static std::string ustr(const fs::path& p) { return norm_path(p.u8string()); }

std::string canonical(const std::string& p) {
    std::error_code ec;
    fs::path c = fs::weakly_canonical(upath(p), ec);
    return ec ? norm_path(p) : ustr(c);
}

std::string basename(const std::string& p) { return upath(p).filename().u8string(); }
std::string dirname(const std::string& p) { return ustr(upath(p).parent_path()); }
std::string stem(const std::string& p) { return upath(p).stem().u8string(); }
std::string ext_lower(const std::string& p) {
    std::string e = upath(p).extension().u8string();
    return lower(e.empty() ? e : e.substr(1));
}

std::string extension(const std::string& p) { return upath(p).extension().u8string(); }

bool mkdirs(const std::string& p) {
    std::error_code ec;
    fs::create_directories(upath(p), ec);
    return dir_exists(p);
}

void set_file_times(const std::string& p, int64_t atime, int64_t mtime) {
#ifdef _WIN32
    struct __utimbuf64 t{atime, mtime};
    _wutime64(widen(p).c_str(), &t);
#else
    struct timespec ts[2] = {{(time_t)atime, 0}, {(time_t)mtime, 0}};
    utimensat(AT_FDCWD, p.c_str(), ts, 0);
#endif
}

#ifndef _WIN32  // Windows: os_win.cpp
bool copy_file_preserve(const std::string& from, const std::string& to, std::string& err) {
    int in = open(from.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0) { err = fmt("open %s: %s", from.c_str(), strerror(errno)); return false; }
    struct stat st{};
    fstat(in, &st);
    // O_EXCL: the library never overwrites an existing file.
    std::string part = to + ".part";
    int out = open(part.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (out < 0) { err = fmt("create %s: %s", part.c_str(), strerror(errno)); close(in); return false; }
    std::vector<char> buf(1 << 20);
    bool ok = true;
    for (;;) {
        ssize_t n = read(in, buf.data(), buf.size());
        if (n == 0) break;
        if (n < 0) { ok = false; err = fmt("read: %s", strerror(errno)); break; }
        for (ssize_t off = 0; off < n;) {
            ssize_t w = write(out, buf.data() + off, size_t(n - off));
            if (w <= 0) { ok = false; err = fmt("write: %s", strerror(errno)); break; }
            off += w;
        }
        if (!ok) break;
    }
    close(in);
    if (ok && fsync(out) != 0) ok = false;
    close(out);
    if (ok) {
        struct timespec ts[2] = {st.st_atim, st.st_mtim};  // keep the source's times on the copy (nanoseconds too)
        utimensat(AT_FDCWD, part.c_str(), ts, 0);
        // link() fails if `to` appeared meanwhile, so we still never clobber anything.
        if (link(part.c_str(), to.c_str()) != 0) { ok = false; err = fmt("link %s: %s", to.c_str(), strerror(errno)); }
    }
    unlink(part.c_str());
    return ok;
}

#endif

std::string temp_path(const std::string& tag) {
    static std::atomic<unsigned> seq{0};
#ifdef _WIN32
    std::string t = env("TEMP");
    if (t.empty()) t = cache_dir();
#else
    std::string t = env("TMPDIR");
    if (t.empty()) t = "/tmp";
#endif
    return fmt("%s/jev-photos-%d-%u-%s", t.c_str(), (int)getpid(), seq++, tag.c_str());
}

#ifndef _WIN32  // Windows: os_win.cpp
// ---------------------------------------------------------------------------
// processes

ProcResult run(const std::vector<std::string>& argv, const std::string& stdin_data, int timeout_s) {
    ProcResult r;
    int pin[2], pout[2], perr[2];
    if (pipe2(pin, O_CLOEXEC) || pipe2(pout, O_CLOEXEC) || pipe2(perr, O_CLOEXEC)) { r.err = "pipe failed"; return r; }
    // posix_spawn, not fork(): glibc spawns with CLONE_VM|CLONE_VFORK, so the child never gets a copy-on-write copy
    // of our address space. Forking a process that has live NVIDIA GL mappings (unified memory on GB10) while
    // the UI thread renders crashes the driver in the parent.
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pin[0], 0);
    posix_spawn_file_actions_adddup2(&fa, pout[1], 1);
    posix_spawn_file_actions_adddup2(&fa, perr[1], 2);
    std::vector<char*> args;
    for (auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
    args.push_back(nullptr);
    pid_t pid = -1;
    int sp = posix_spawnp(&pid, args[0], &fa, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (sp != 0) {
        close(pin[0]); close(pin[1]); close(pout[0]); close(pout[1]); close(perr[0]); close(perr[1]);
        r.rc = sp == ENOENT ? 127 : -1;
        r.err = std::string("cannot run ") + argv[0] + ": " + strerror(sp);
        return r;
    }
    close(pin[0]); close(pout[1]); close(perr[1]);
    size_t in_off = 0;
    if (stdin_data.empty()) { close(pin[1]); pin[1] = -1; }
    else fcntl(pin[1], F_SETFL, O_NONBLOCK);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    bool out_open = true, err_open = true, timed_out = false;
    char buf[65536];
    while (out_open || err_open) {
        struct pollfd fds[3];
        int nf = 0, io = -1, ie = -1, ii = -1;
        if (out_open) { io = nf; fds[nf++] = {pout[0], POLLIN, 0}; }
        if (err_open) { ie = nf; fds[nf++] = {perr[0], POLLIN, 0}; }
        if (pin[1] >= 0) { ii = nf; fds[nf++] = {pin[1], POLLOUT, 0}; }
        int left = int(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
        if (left <= 0) { timed_out = true; break; }
        if (poll(fds, nf, std::min(left, 500)) < 0 && errno != EINTR) break;
        if (io >= 0 && fds[io].revents) {
            ssize_t n = read(pout[0], buf, sizeof buf);
            if (n > 0) r.out.append(buf, size_t(n)); else out_open = false;
        }
        if (ie >= 0 && fds[ie].revents) {
            ssize_t n = read(perr[0], buf, sizeof buf);
            if (n > 0) r.err.append(buf, size_t(n)); else err_open = false;
        }
        if (ii >= 0 && fds[ii].revents) {
            ssize_t n = write(pin[1], stdin_data.data() + in_off, stdin_data.size() - in_off);
            if (n > 0) in_off += size_t(n);
            if (n < 0 || in_off >= stdin_data.size()) { close(pin[1]); pin[1] = -1; }
        }
    }
    if (pin[1] >= 0) close(pin[1]);
    close(pout[0]); close(perr[0]);
    if (timed_out) kill(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);
    r.rc = timed_out ? -2 : WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (timed_out) r.err += "\n(timed out)";
    return r;
}

bool which(const std::string& exe) {
    const char* path = getenv("PATH");
    if (!path) return false;
    for (auto& d : split(path, ':'))
        if (!d.empty() && access((d + "/" + exe).c_str(), X_OK) == 0) return true;
    return false;
}


// ---------------------------------------------------------------------------
// desktop

bool system_reduce_motion() {  // GNOME's switch (other desktops: not set, full motion)
    ProcResult r = run({"gsettings", "get", "org.gnome.desktop.interface", "enable-animations"}, "", 3);
    return r.rc == 0 && trim(r.out) == "false";
}

bool have_trash() { return which("gio"); }
bool trash(const std::string& path) { return run({"gio", "trash", "--", path}, "", 30).rc == 0; }
void open_path(const std::string& p) {
    std::string target = p;
    std::thread([target] { run({"xdg-open", target}, "", 30); }).detach();
}
std::string choose_folder(const std::string& title) {
    ProcResult r = run({"zenity", "--file-selection", "--directory", "--title=" + title}, "", 3600);
    if (r.rc == 127) r = run({"kdialog", "--getexistingdirectory", home()}, "", 3600);
    return r.rc == 0 ? norm_path(trim(r.out)) : "";
}
#endif

}  // namespace util
