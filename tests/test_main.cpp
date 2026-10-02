// Unit tests for the decision model and metadata safety. Run: build/jev-photos-tests
#include <sys/stat.h>
#include <filesystem>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>

#include "dates.h"
#include "genmeta.h"
#include "db.h"
#include "pipeline.h"
#include "dupes.h"
#include "thumbs.h"
#include "search.h"
#include "location.h"
#include "meta.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "util.h"
#include "vision.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (cond) g_pass++;                                                      \
        else { g_fail++; fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)
#define CHECK_EQ(a, b)                                                                                     \
    do {                                                                                                   \
        auto _a = (a); auto _b = (b);                                                                      \
        if (_a == _b) g_pass++;                                                                            \
        else { g_fail++; fprintf(stderr, "FAIL %s:%d  %s == %s  (got \"%s\")\n", __FILE__, __LINE__, #a, #b, std::string(_a).c_str()); } \
    } while (0)

static const int64_t kNow = 1790000000;  // 2026-09-21

static std::string first(const std::string& name, bool dir = false) {
    auto v = date_candidates_from_name(name, dir, 0, kNow);
    if (v.empty()) return "-";
    return std::string(date_source_id(v[0].src)) + " " + v[0].when.pretty();
}

static int64_t epoch_local(int y, int mo, int d, int h, int mi, int s) {
    struct tm tm{};
    tm.tm_year = y - 1900; tm.tm_mon = mo - 1; tm.tm_mday = d; tm.tm_hour = h; tm.tm_min = mi; tm.tm_sec = s; tm.tm_isdst = -1;
    return mktime(&tm);
}

static MetaMap meta_with(std::initializer_list<std::pair<const char*, const char*>> kv) {
    MetaMap m;
    for (auto& [k, v] : kv) { m.kv[k].push_back(v); m.type[k] = "Ascii"; }
    return m;
}

static void rm_rf(const std::string& dir) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::u8path(dir), ec);
}
static void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());  // "" removes it
#else
    if (v.empty()) unsetenv(k);
    else setenv(k, v.c_str(), 1);
#endif
}

static void test_civil() {
    for (int64_t d : {-1000LL, 0LL, 10957LL, 20000LL}) {
        int y, m, dd;
        util::civil_from_days(d, y, m, dd);
        CHECK(util::days_from_civil(y, m, dd) == d);
    }
    CHECK(!util::make_civil(2019, 2, 29, 0, 0, 0, util::P_DAY).valid());
    CHECK(util::make_civil(2020, 2, 29, 0, 0, 0, util::P_DAY).valid());
    CHECK_EQ(util::parse_exif_datetime("2019:05:12 14:30:22").pretty(), "2019-05-12 14:30:22");
    CHECK_EQ(util::parse_exif_datetime("2019-05-12T14:30:22+08:00").pretty(), "2019-05-12 14:30:22");
    CHECK_EQ(util::parse_exif_datetime("2019-05").pretty(), "2019-05");
    CHECK(!util::parse_exif_datetime("0000:00:00 00:00:00").valid());
    CHECK(!util::parse_exif_datetime("    :  :     :  :  ").valid());
}

static void test_name_patterns() {
    CHECK_EQ(first("IMG_20190512_143022"), "name-datetime 2019-05-12 14:30:22");
    CHECK_EQ(first("PXL_20210101_123456789"), "name-datetime 2021-01-01 12:34:56");
    CHECK_EQ(first("Screenshot_2019-05-12-14-30-22"), "name-datetime 2019-05-12 14:30:22");
    CHECK_EQ(first("Photo 2019-05-12 at 14.30.22"), "name-datetime 2019-05-12 14:30:22");
    CHECK_EQ(first("2019-05-12 14.30"), "name-datetime 2019-05-12 14:30");
    CHECK_EQ(first("IMG-20190512-WA0001"), "name-date 2019-05-12");
    CHECK_EQ(first("2019年5月12日 生日"), "name-date 2019-05-12");
    CHECK_EQ(first("DSC01234"), "-");
    CHECK_EQ(first("IMG_20191332_000000"), "-");  // month 13
    CHECK(util::starts_with(first("mmexport1557671422000"), "name-unix 2019-05-1"));
    CHECK_EQ(first("IMG_1234"), "-");
    CHECK_EQ(first("201905", false), "-");        // six digits in a file name: ambiguous
    // folders
    CHECK_EQ(first("2019-05 Paris trip", true), "dir-month 2019-05");
    CHECK_EQ(first("201905", true), "dir-month 2019-05");
    CHECK_EQ(first("2019", true), "dir-year 2019");
    CHECK_EQ(first("May 2019", true), "dir-month 2019-05");
    CHECK_EQ(first("20190512 生日", true), "dir-date 2019-05-12");
    CHECK_EQ(first("2019.05.12_西湖", true), "dir-date 2019-05-12");
    CHECK_EQ(first("Camera", true), "-");
}

static DateDecision decide(const MetaMap& m, const std::string& file, std::vector<std::string> dirs, int64_t mtime, bool bulk = false) {
    DateInputs in;
    in.filename = file;
    in.dirs = dirs;
    in.meta = &m;
    in.mtime = mtime;
    in.bulk_mtime = bulk;
    in.now = kNow;
    return decide_date(in, 0.55);
}

static void test_decisions() {
    // a) camera EXIF plus a later copy mtime: EXIF wins with high confidence; the copy time is not a contradiction.
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2019:05:12 14:30:22"}, {"Exif.Photo.DateTimeDigitized", "2019:05:12 14:30:22"}});
        auto d = decide(m, "DSC01234.JPG", {"DCIM"}, epoch_local(2024, 3, 1, 9, 0, 0));
        CHECK_EQ(d.when.str(), "2019-05-12 14:30:22");
        CHECK_EQ(d.source, "exif-original");
        CHECK(d.confidence > 0.9);
        CHECK(!d.needs_review);
        CHECK(d.alternatives.size() == 1);
    }
    // b) camera clock reset to 2000-01-01 vs a dated file name: the name wins.
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2000:01:01 00:00:05"}});
        auto d = decide(m, "IMG_20190512_143022.jpg", {}, epoch_local(2019, 5, 12, 14, 30, 22));
        CHECK_EQ(d.when.str(), "2019-05-12 14:30:22");
        CHECK(util::starts_with(d.source, "name-datetime"));
        CHECK(!d.needs_review);
    }
    // c) WhatsApp (EXIF stripped): date from name, time of day borrowed from an agreeing mtime.
    {
        MetaMap m;
        auto d = decide(m, "IMG-20190512-WA0001.jpg", {"WhatsApp Images"}, epoch_local(2019, 5, 12, 18, 2, 3));
        CHECK_EQ(d.when.str(), "2019-05-12 18:02:03");
        CHECK_EQ(d.source, "name-date+time:fs-mtime");
        CHECK(d.confidence > 0.9);
    }
    // d) only a file time: low confidence -> review
    {
        MetaMap m;
        auto d = decide(m, "DSC01234.jpg", {}, epoch_local(2021, 7, 4, 10, 0, 0));
        CHECK_EQ(d.source, "fs-mtime");
        CHECK(d.needs_review);
        CHECK(d.confidence < 0.4);
    }
    // e) folder month vs a bulk-copy mtime: the folder wins, still flagged for review (month precision only)
    {
        MetaMap m;
        auto d = decide(m, "DSC01234.jpg", {"2019-05 Paris trip"}, epoch_local(2024, 3, 1, 9, 0, 0), true);
        CHECK_EQ(d.when.pretty(), "2019-05");
        CHECK_EQ(d.source, "dir-month");
    }
    // f) EXIF vs a conflicting dated name: EXIF leads, margin is low, jev can flip it.
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2018:03:01 10:00:00"}});
        auto d = decide(m, "20200101_120000.jpg", {}, epoch_local(2024, 6, 1, 12, 0, 0));  // mtime = unrelated copy time
        CHECK_EQ(d.source, "exif-original");
        CHECK(d.alternatives.size() == 2);
        CHECK(d.margin < 0.7);
        std::string opt = date_option_text(d, d.alternatives[1]);
        CHECK(opt.find("file name") != std::string::npos);
        apply_jev_date(d, {0.1, 0.9}, 0.3, 0.55);
        CHECK_EQ(d.when.str(), "2020-01-01 12:00:00");
        CHECK_EQ(d.decider, "rules+jev");
        CHECK(d.needs_review);  // a flipped close call is still worth a look
    }
    // g) GPS UTC + offset agrees with EXIF local time
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2019:05:12 14:30:22"}, {"Exif.Photo.OffsetTimeOriginal", "+08:00"},
                            {"Exif.GPSInfo.GPSDateStamp", "2019:05:12"}, {"Exif.GPSInfo.GPSTimeStamp", "6/1 30/1 20/1"}});
        auto d = decide(m, "DSC.jpg", {}, 0);
        CHECK_EQ(d.when.str(), "2019-05-12 14:30:22");
        bool gps_agrees = false;
        for (auto& c : d.candidates)
            if (c.src == DS_GPS) gps_agrees = c.when.str() == "2019-05-12 14:30:20";
        CHECK(gps_agrees);
        CHECK(d.candidates[0].support > 1.5);
    }
    // h) future and sentinel values are rejected
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2099:01:01 10:00:00"}, {"Exif.Image.DateTime", "1970:01:01 00:00:00"}});
        auto d = decide(m, "x.jpg", {}, epoch_local(2020, 6, 1, 8, 0, 0));
        CHECK_EQ(d.source, "fs-mtime");
        for (auto& c : d.candidates)
            if (c.src != DS_FS_MTIME) CHECK(c.rejected());
    }
    // i) manual always wins
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2019:05:12 14:30:22"}});
        DateInputs in;
        in.filename = "a.jpg";
        in.meta = &m;
        in.now = kNow;
        auto c = collect_date_candidates(in);
        DateCandidate mc;
        mc.src = DS_MANUAL;
        mc.when = util::make_civil(1998, 7, 1, 0, 0, 0, util::P_DAY);
        mc.base = mc.weight = 1;
        c.push_back(mc);
        auto d = decide_from_candidates(c, 0.55);
        CHECK_EQ(d.when.pretty(), "1998-07-01");
        CHECK_EQ(d.decider, "manual");
        CHECK(!d.needs_review);
    }
    // j) EXIF later than the file's own mtime is suspicious (edited clock) and loses weight
    {
        auto m = meta_with({{"Exif.Photo.DateTimeOriginal", "2023:01:05 10:00:00"}});
        auto d = decide(m, "x.jpg", {}, epoch_local(2019, 5, 12, 14, 0, 0));
        CHECK(d.candidates[0].weight < 0.5);
        CHECK(d.needs_review);
    }
    // l) a folder year beats agreeing copy times (file times are only upper bounds)
    {
        MetaMap m;
        DateInputs in;
        in.filename = "fabio.jpg";
        in.dirs = {"2021 Summer"};
        in.meta = &m;
        in.mtime = epoch_local(2026, 9, 30, 23, 0, 0);
        in.btime = epoch_local(2026, 9, 30, 23, 0, 0);
        in.now = epoch_local(2026, 10, 1, 0, 0, 0);
        auto d = decide_date(in, 0.55);
        CHECK_EQ(d.when.pretty(), "2021");
        CHECK_EQ(d.source, "dir-year");
    }
    // k) nothing at all
    {
        MetaMap m;
        auto d = decide(m, "x.jpg", {}, 0);
        CHECK(!d.when.valid());
        CHECK(d.needs_review);
    }
}

static void test_location() {
    auto pc = place_candidates({"2019-05 Paris trip", "Travel"}, "IMG_0001");
    CHECK(!pc.empty() && pc[0].text == "Paris" && pc[0].travel_cue);
    auto d = decide_place(pc, {rules_vote(pc, 1.0)}, 0.5);
    CHECK_EQ(d.text, "Paris");

    auto g = place_candidates({"Grandma", "DCIM"}, "DSC0001");
    CHECK(g.size() == 1 && g[0].text == "Grandma");
    CHECK_EQ(decide_place(g, {rules_vote(g, 1.0)}, 0.5).text, "");  // not without a model or a trip cue

    auto z = place_candidates({"巴黎旅行"}, "");
    CHECK(z.size() == 1 && z[0].text == "巴黎");
    CHECK(place_candidates({"新建文件夹", "Camera Roll", "WhatsApp Images"}, "IMG_20190512_143022").empty());
    auto w = place_candidates({"2019.05.12_西湖_family"}, "");
    CHECK(!w.empty() && w[0].text == "西湖");
    auto t = place_candidates({"Trip to New York"}, "");
    CHECK(t.size() == 1 && t[0].text == "New York" && t[0].travel_cue);

    // a VL vote decides between two plain names
    auto two = place_candidates({"Tokyo", "Birthday"}, "");
    PlaceVote vl{"vl", 0.6, {1.0, 0.0}, ""};
    CHECK_EQ(decide_place(two, {rules_vote(two, 0.25), vl}, 0.5).text, "Tokyo");
    CHECK(match_candidate(two, "tokyo") == 0);
    CHECK(match_candidate(two, "") == -1);
    // jev alone cannot promote a candidate that neither the rules nor the VL back
    PlaceVote jev{"jev", 0.15, {0.9, 0.05}, ""};
    CHECK_EQ(decide_place(two, {rules_vote(two, 0.25), jev}, 0.5).text, "");
    auto hw = place_candidates({"2018-07 Hawaii vacation"}, "");
    PlaceVote jev2{"jev", 0.15, {0.6}, ""};
    CHECK_EQ(decide_place(hw, {rules_vote(hw, 0.25), jev2}, 0.5).text, "Hawaii");
    CHECK(place_candidates({"100CANON", "src"}, "").empty());
}

static void test_json_extract() {
    CHECK_EQ(extract_json_object("```json\n{\"a\": \"}\"}\n```"), "{\"a\": \"}\"}");
    CHECK_EQ(extract_json_object("<think>{x}</think> ok {\"b\":{\"c\":1}} tail"), "{\"b\":{\"c\":1}}");
    CHECK_EQ(extract_json_object("no json"), "");
}

static void test_meta_roundtrip() {
    if (!exiv2_available()) { fprintf(stderr, "skip meta test: no exiv2\n"); return; }
    std::string dir = util::temp_path("t");
    util::mkdirs(dir);
    std::string img = dir + "/a.jpg";
    unsigned char px[16 * 16 * 3];
    for (int i = 0; i < 16 * 16 * 3; i++) px[i] = (unsigned char)(i * 7);
    CHECK(stbi_write_jpg(img.c_str(), 16, 16, 3, px, 90));
    util::write_file(dir + "/c", "set Exif.Photo.DateTimeOriginal Ascii 2015:06:01 08:00:00\nset Xmp.dc.subject XmpBag family\n"
                                 "set Xmp.dc.description LangAlt lang=\"x-default\" Original caption\n");
    CHECK(util::run({"exiv2", "-q", "-m", dir + "/c", img}).rc == 0);
    struct stat before{};
    stat(img.c_str(), &before);

    MetaMap m = read_meta(img);
    CHECK_EQ(m.get("Exif.Photo.DateTimeOriginal"), "2015:06:01 08:00:00");
    MetaPlan p;
    p.date = util::make_civil(2019, 5, 12, 14, 30, 22, util::P_SECOND);  // disagrees on purpose
    p.date_trusted = true;
    p.date_conf = 0.9;
    p.date_source = "name-datetime";
    p.caption = "A new caption";
    p.location = "Paris";
    p.keywords = {"Family", "beach", "sunset, orange sky"};
    WriteResult w = write_meta(img, m, p, MetaTarget::Embed, false);
    CHECK(w.ok);
    MetaMap after = read_meta(img);
    CHECK_EQ(after.get("Exif.Photo.DateTimeOriginal"), "2015:06:01 08:00:00");                 // never overwritten
    CHECK(after.get("Xmp.dc.description").find("Original caption") != std::string::npos);     // never overwritten
    CHECK_EQ(after.get("Xmp.photoshop.DateCreated"), "2019-05-12T14:30:22");                  // added: was missing
    CHECK_EQ(after.get("Xmp.iptc.Location"), "Paris");
    CHECK_EQ(after.get("Xmp.jev.Caption"), "A new caption");
    auto kws = after.list("Xmp.dc.subject");
    CHECK(kws.size() == 3);  // family (kept, not duplicated as "Family"), beach, "sunset  orange sky"
    struct stat st{};
    stat(img.c_str(), &st);
    CHECK(ST_MTIME(st) == ST_MTIME(before));  // file time preserved

    WriteResult again = write_meta(img, after, p, MetaTarget::Embed, false);
    CHECK(again.ok && again.added == 0);  // idempotent
    CHECK_EQ(after.get("Xmp.jev.Keywords"), "beach, sunset  orange sky");  // ours: "family" was the file's own

    // Rewrite: our earlier keywords that are no longer wanted go; the file's own keyword stays.
    MetaPlan p2 = p;
    p2.keywords = {"beach", "kids"};
    p2.rewrite_keywords = true;
    WriteResult rw = write_meta(img, after, p2, MetaTarget::Embed, false);
    CHECK(rw.ok && rw.removed.size() == 1 && rw.removed[0] == "sunset  orange sky");
    MetaMap after2 = read_meta(img);
    auto k2 = after2.list("Xmp.dc.subject");
    CHECK(k2.size() == 3);  // family, beach, kids
    bool fam = false, kids = false, sunset = false;
    for (auto& k : k2) { fam |= k == "family"; kids |= k == "kids"; sunset |= k.find("sunset") != std::string::npos; }
    CHECK(fam && kids && !sunset);
    CHECK_EQ(after2.get("Xmp.jev.Keywords"), "beach, kids");
    // Without rewrite nothing is taken out.
    MetaPlan p3 = p2;
    p3.keywords = {"dog"};
    p3.rewrite_keywords = false;
    WriteResult nr = write_meta(img, after2, p3, MetaTarget::Embed, true);
    CHECK(nr.removed.empty() && nr.added == 1);

    // sidecar mode leaves the image bytes alone
    std::string img2 = dir + "/b.jpg";
    CHECK(stbi_write_jpg(img2.c_str(), 16, 16, 3, px, 90));
    std::string sha = util::sha256_file(img2);
    WriteResult sc = write_meta(img2, read_meta(img2), p, MetaTarget::Sidecar, false);
    CHECK(sc.ok);
    CHECK_EQ(util::sha256_file(img2), sha);
    MetaMap side = read_meta(sidecar_path(img2));
    CHECK_EQ(side.get("Xmp.exif.DateTimeOriginal"), "2019-05-12T14:30:22");
    CHECK(side.list("Xmp.dc.subject").size() == 3);

    // copy never clobbers
    std::string err;
    CHECK(!util::copy_file_preserve(img2, img, err));
    CHECK(util::copy_file_preserve(img2, dir + "/c.jpg", err));
    CHECK_EQ(util::sha256_file(dir + "/c.jpg"), sha);
    rm_rf(dir);
}

static void test_sha() {
    std::string p = util::temp_path("sha");
    util::write_file(p, "abc");
    CHECK_EQ(util::sha256_file(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    unlink(p.c_str());
    const unsigned char b[] = {'M', 'a', 'n'};
    CHECK_EQ(util::base64(b, 3), "TWFu");
    CHECK_EQ(util::base64(b, 2), "TWE=");
}

static void test_hashes_and_dupes() {
    std::string dir = util::temp_path("dup");
    util::mkdirs(dir);
    std::string big(300000, 'x'), big2 = big;
    big2[150000] = 'y';  // differs only in the middle: same size, same head/tail
    util::write_file(dir + "/a", big);
    util::write_file(dir + "/b", big);
    util::write_file(dir + "/c", big2);
    util::write_file(dir + "/d", big + "z");
    CHECK(util::quick_hash(dir + "/a", 300000) == util::quick_hash(dir + "/b", 300000));
    CHECK(util::quick_hash(dir + "/a", 300000) == util::quick_hash(dir + "/c", 300000));  // quick hash cannot see the middle...
    CHECK(util::content_hash(dir + "/a") != util::content_hash(dir + "/c"));              // ...the full hash can
    CHECK(util::content_hash(dir + "/a") == util::content_hash(dir + "/b"));
    CHECK(util::content_hash(dir + "/a").size() == 32);
    CHECK(util::quick_hash(dir + "/a", 300000) != util::quick_hash(dir + "/d", 300001));
    CHECK(util::files_equal(dir + "/a", dir + "/b"));
    CHECK(!util::files_equal(dir + "/a", dir + "/c"));
    CHECK(!util::files_equal(dir + "/a", dir + "/d"));
    util::write_file(dir + "/s1", "tiny");
    util::write_file(dir + "/s2", "tinz");
    CHECK(util::quick_hash(dir + "/s1", 4) != util::quick_hash(dir + "/s2", 4));  // small files are hashed whole
    CHECK_EQ(util::human_size(512), "512 B");
    CHECK_EQ(util::human_size(4404019), "4.2 MB");

    // keeper: pinned > already in library > richer metadata > shorter path > older id
    DupRow r1, r2, r3;
    r1.id = 1; r1.src_path = "/p/deep/er/IMG_1.jpg"; r1.meta_len = 100;
    r2.id = 2; r2.src_path = "/p/IMG_1.jpg"; r2.meta_len = 100;
    r3.id = 3; r3.src_path = "/p/x/IMG_1.jpg"; r3.meta_len = 900;
    CHECK(choose_keeper({&r1, &r2}) == 2);
    CHECK(choose_keeper({&r1, &r2, &r3}) == 3);
    CHECK(looks_like_copy("/p/IMG_0004 (1).JPG") && looks_like_copy("/p/backup/IMG_1.jpg") && looks_like_copy("/p/IMG_1 copy.jpg"));
    CHECK(looks_like_copy("/p/备份/IMG_1.jpg") && looks_like_copy("/p/IMG_1 - 副本.jpg"));
    CHECK(!looks_like_copy("/p/DCIM/100CANON/IMG_0004.JPG") && !looks_like_copy("/p/2019-05 Paris/IMG_0001.jpg"));
    DupRow o1, o2;
    o1.id = 5; o1.src_path = "/p/backup/IMG_1.jpg";
    o2.id = 6; o2.src_path = "/p/DCIM/100CANON/IMG_1.jpg";
    CHECK(choose_keeper({&o1, &o2}) == 6);  // not the shorter backup path
    r1.dest_path = "/lib/2019/x.jpg";
    CHECK(choose_keeper({&r1, &r2, &r3}) == 1);
    r2.keep = true;
    CHECK(choose_keeper({&r1, &r2, &r3}) == 2);

    // dHash: a resized + re-compressed copy and a rotated copy are near; a different picture is far
    const int W = 240, H = 160;
    std::vector<unsigned char> img(W * H * 3), small((W / 2) * (H / 2) * 3), other(W * H * 3), rot(H * W * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char v = (unsigned char)(128 + 100 * std::sin(x / 17.0) * std::cos(y / 23.0));
            unsigned char o = (unsigned char)((x * 7 + y * 3) % 256 > 128 ? 230 : 20);
            for (int k = 0; k < 3; k++) {
                img[(y * W + x) * 3 + k] = (unsigned char)(v + k * 20);
                other[(y * W + x) * 3 + k] = o;
                rot[((x) * H + (H - 1 - y)) * 3 + k] = (unsigned char)(v + k * 20);  // rotated 90 degrees clockwise
            }
        }
    for (int y = 0; y < H / 2; y++)
        for (int x = 0; x < W / 2; x++)
            for (int k = 0; k < 3; k++) small[(y * (W / 2) + x) * 3 + k] = img[((y * 2) * W + x * 2) * 3 + k];
    CHECK(stbi_write_jpg((dir + "/orig.jpg").c_str(), W, H, 3, img.data(), 95));
    CHECK(stbi_write_jpg((dir + "/small.jpg").c_str(), W / 2, H / 2, 3, small.data(), 60));
    CHECK(stbi_write_jpg((dir + "/other.jpg").c_str(), W, H, 3, other.data(), 90));
    CHECK(stbi_write_jpg((dir + "/rot.jpg").c_str(), H, W, 3, rot.data(), 90));
    uint64_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
    CHECK(image_dhash(dir + "/orig.jpg", 1, h0));
    CHECK(image_dhash(dir + "/small.jpg", 1, h1));
    CHECK(image_dhash(dir + "/other.jpg", 1, h2));
    CHECK(image_dhash(dir + "/rot.jpg", 8, h3));  // stored rotated, EXIF orientation 8 turns it back
    CHECK(hamming64(h0, h1) <= 6);
    CHECK(hamming64(h0, h3) <= 6);
    CHECK(hamming64(h0, h2) > 12);
    DupRow a, b, c;
    a.id = 10; a.dhash = h0; a.width = W; a.height = H; a.size = 50000;
    b.id = 11; b.dhash = h1; b.width = W / 2; b.height = H / 2; b.size = 9000;
    c.id = 12; c.dhash = h2; c.width = W; c.height = H;
    auto links = similar_links({&a, &b, &c}, 6);
    CHECK(links.size() == 1 && links[0].id == 11 && links[0].to == 10);  // the larger original is the representative
    rm_rf(dir);
}

static void test_thumbs() {
    std::string dir = util::temp_path("thumbs");
    util::mkdirs(dir);
#ifdef _WIN32
    const char* cache_var = "LOCALAPPDATA";
    const std::string cache_sub = "/jev-photos/cache/thumbs/";
    std::string saved_cache = getenv(cache_var) ? getenv(cache_var) : "";
#else
    const char* cache_var = "XDG_CACHE_HOME";
    const std::string cache_sub = "/jev-photos/thumbs/";
    std::string saved_cache = getenv(cache_var) ? getenv(cache_var) : "";
#endif
    set_env(cache_var, dir);
    CHECK(thumb_side_for("/p/IMG_1.JPG", 512, 1024) == 512);
    CHECK(thumb_side_for("/p/desktop__about.png", 512, 1024) == 1024);
    CHECK(thumb_side_for("/p/Screenshot_2021-11-02.jpg", 512, 1024) == 1024);
    CHECK(thumb_side_for("/p/微信截图_1.jpg", 512, 1024) == 1024);
    // a 3000x2000 photo stored sideways (orientation 6) -> upright 341x512 thumbnail, made once
    std::vector<unsigned char> px(3000 * 2000 * 3, 90);
    std::string img = dir + "/big.jpg";
    CHECK(stbi_write_jpg(img.c_str(), 3000, 2000, 3, px.data(), 80));
    ThumbKey k{img, 123, 456};
    std::string err, t1 = ensure_thumb(k, 6, 512, err);
    CHECK(!t1.empty() && util::starts_with(t1, dir + cache_sub));
    int w = 0, h = 0, n = 0;
    CHECK(stbi_info(t1.c_str(), &w, &h, &n) && w == 341 && h == 512);
    struct stat a{}, b{};
    stat(t1.c_str(), &a);
    CHECK(ensure_thumb(k, 6, 512, err) == t1);  // cached: not rewritten
    stat(t1.c_str(), &b);
#ifdef _WIN32
    CHECK(ST_MTIME(a) == ST_MTIME(b) && a.st_size == b.st_size);
#else
    CHECK(a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ino == b.st_ino);
#endif
    CHECK(thumb_path({img, 123, 457}, 512) != t1);  // a changed file gets a new thumbnail
    rm_rf(dir);
    set_env(cache_var, saved_cache);
}

static void test_date_ranges() {
    CHECK(date_in_range("2019-05-12 14:30:22", "second", "2019-05-01", "2019-05-31"));
    CHECK(!date_in_range("2019-06-01 00:00:00", "day", "2019-05-01", "2019-05-31"));
    CHECK(date_in_range("2019-01-01 00:00:00", "year", "2019-06-01", "2019-06-30"));    // a year-only date covers June
    CHECK(date_in_range("2019-05-01 00:00:00", "month", "2019-05-20", ""));             // a month-only date covers the 20th
    CHECK(!date_in_range("2019-05-01 00:00:00", "month", "2019-06-01", ""));
    CHECK(!date_in_range("", "", "2019-01-01", ""));                                      // undated never matches a range
    CHECK(date_in_range("", "", "", ""));
}

static void test_naming() {
    CHECK_EQ(name_prefix("IMG_20190512_123456"), "IMG");
    CHECK_EQ(name_prefix("IMG-20190512-WA0003"), "IMG");
    CHECK_EQ(name_prefix("Paris trip 2019-05-12 (3)"), "Paris trip");
    CHECK_EQ(name_prefix("Screenshot 2024-01-02 at 10.11.12"), "Screenshot");
    CHECK_EQ(name_prefix("Qwen_image_2.1_00044"), "Qwen_image_2.1");
    CHECK_EQ(name_prefix("z-image-turbo_00001_"), "z-image-turbo");
    CHECK_EQ(name_prefix("clipspace-mask-1789656145849"), "clipspace-mask");
    CHECK_EQ(name_prefix("微信图片_20190512123456"), "微信图片");
    CHECK_EQ(name_prefix("DSC01234"), "DSC");
    CHECK_EQ(name_prefix("20190512_00001"), "");  // an organized name has no prefix to keep
    CHECK_EQ(name_prefix("IMG_20190512_00001"), "IMG");  // idempotent on our own prefixed names
    CHECK_EQ(name_prefix("1099"), "");
    CHECK_EQ(name_prefix("girl2"), "girl2");  // a single digit is part of the name
    CHECK_EQ(name_prefix("Beach copy 2"), "Beach");
    std::string day;
    int sn = 0;
    CHECK(parse_serial_name("20190512_00007", "_", day, sn) && day == "20190512" && sn == 7);
    CHECK(parse_serial_name("IMG_20190512_00042", "_", day, sn) && day == "20190512" && sn == 42);
    CHECK(parse_serial_name("Paris trip-20190512-00003", "-", day, sn) && day == "20190512" && sn == 3);
    CHECK(parse_serial_name("2019051200011", "", day, sn) && day == "20190512" && sn == 11);
    CHECK(!parse_serial_name("IMG_1234", "_", day, sn));
}

static void test_tags() {
    auto t = parse_tag_list(R"([["woman",0.62],["beach",0.3],["app screen",0.08]])");
    CHECK(t.size() == 3 && t[0].first == "woman" && std::fabs(t[0].second - 0.62f) < 1e-6);
    CHECK(parse_tag_list(R"(["mine","ours"])").size() == 2);
    Photo p;
    p.clip_tags = R"([["woman",0.62],["beach",0.3],["app screen",0.08]])";
    auto e = effective_tags(p);
    CHECK(e.size() == 2 && e[0] == "woman" && e[1] == "beach");  // the 8% guess is not a tag
    p.user_tags = R"(["Grandma","garden"])";
    e = effective_tags(p);
    CHECK(e.size() == 2 && e[0] == "Grandma");  // the user's own list replaces the automatic one
}

static void test_genmeta() {
    // A1111 / Forge / Civitai text
    GenInfo a = parse_a1111("masterpiece, 1girl, (red hat:1.2), beach <lora:detail_tweaker:0.6>\nNegative prompt: lowres, bad hands\n"
                            "Steps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 6.5, Seed: 12345, Size: 832x1216, "
                            "Model hash: abc123, Model: dreamshaper_8, Lora hashes: \"detail_tweaker: 1a2b\", Version: v1.10");
    CHECK_EQ(a.tool, "A1111");
    CHECK(a.prompt.find("1girl") != std::string::npos && a.negative == "lowres, bad hands");
    CHECK(a.steps == 28 && std::fabs(a.cfg - 6.5) < 1e-9 && a.seed == "12345" && a.size == "832x1216");
    CHECK_EQ(a.model, "dreamshaper_8");
    CHECK_EQ(a.sampler, "DPM++ 2M");
    CHECK(a.loras.size() == 1 && a.loras[0].first == "detail_tweaker");
    CHECK_EQ(clean_prompt(a.prompt), "masterpiece, 1girl, red hat, beach");
    auto kw = prompt_keywords(a.prompt);
    CHECK(kw.size() == 3 && kw[0] == "1girl" && kw[1] == "red hat" && kw[2] == "beach");  // "masterpiece" is boilerplate
    CHECK(prompt_keywords("A cinematic photo of an old fisherman mending his nets at dawn, warm light").empty());  // prose
    auto lk = local_keywords("A cinematic photo of an old fisherman mending his nets at dawn, fisherman smiling, warm light");
    CHECK(!lk.empty() && lk[0] == "fisherman");  // most frequent meaningful word first; "cinematic", "photo", "light" dropped
    auto zk = local_keywords("海边的女孩，戴着草帽、微笑");
    CHECK(std::find(zk.begin(), zk.end(), "戴着草帽") != zk.end());

    // ComfyUI graph: text reached through a concatenate node; negative zeroed; seed from a seed node; LoRA; input image
    GenInfo c = parse_comfy_prompt(R"({
      "3": {"class_type": "KSampler", "inputs": {"seed": ["9", 0], "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple",
            "model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0]}},
      "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "models/sdxl/juggernaut.safetensors"}},
      "5": {"class_type": "LoraLoader", "inputs": {"lora_name": "style/ink.safetensors", "strength_model": 0.7, "model": ["4", 0], "clip": ["4", 1]}},
      "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["10", 0], "clip": ["5", 1]}},
      "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
      "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
      "9": {"class_type": "SeedNode", "inputs": {"seed": 42}},
      "10": {"class_type": "StringConcatenate", "inputs": {"string_a": "a red fox in the snow", "string_b": ["11", 0], "delimiter": ", "}},
      "11": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "ink painting"}},
      "12": {"class_type": "LoadImage", "inputs": {"image": "fox.png"}}})");
    CHECK_EQ(c.tool, "ComfyUI");
    CHECK(c.prompt.find("a red fox in the snow") != std::string::npos && c.prompt.find("ink painting") != std::string::npos);
    CHECK(c.negative.empty());  // ConditioningZeroOut is not a prompt
    CHECK(c.seed == "42" && c.steps == 20 && std::fabs(c.cfg - 3.5) < 1e-9 && c.sampler == "euler");
    CHECK_EQ(c.model, "juggernaut.safetensors");
    CHECK(c.loras.size() == 1 && c.loras[0].first == "ink.safetensors" && std::fabs(c.loras[0].second - 0.7) < 1e-9);
    CHECK(c.size == "1024x768" && c.sources.size() == 1);
    // A node holding both prompts (Qwen-Image style): positive and negative come from the right keys
    GenInfo q = parse_comfy_prompt(R"({"1": {"class_type": "KSampler", "inputs": {"seed": 7, "positive": ["2", 0], "negative": ["2", 1]}},
      "2": {"class_type": "TextEncodeQwenImage", "inputs": {"prompt": "a cat on a sofa", "negative_prompt": "blurry"}}})");
    CHECK(q.prompt == "a cat on a sofa" && q.negative == "blurry");
    CHECK(GenInfo::from_json(c.to_json()).prompt == c.prompt);

    // PNG text chunks: tEXt written by hand into a real PNG
    std::string dir = util::temp_path("gen");
    util::mkdirs(dir);
    std::string png = dir + "/g.png";
    unsigned char px[4 * 4 * 3] = {0};
    CHECK(stbi_write_png(png.c_str(), 4, 4, 3, px, 12));
    std::string data;
    util::read_file(png, data);
    std::string body = std::string("parameters") + '\0' + "a lighthouse at night\nSteps: 10, Sampler: Euler a, CFG scale: 7, Seed: 9, Size: 4x4";
    auto be32 = [](uint32_t v) { std::string s(4, '\0'); s[0] = char(v >> 24); s[1] = char(v >> 16); s[2] = char(v >> 8); s[3] = char(v); return s; };
    std::string chunk = be32(uint32_t(body.size())) + "tEXt" + body + be32(0);  // CRC is not checked by our reader
    data.insert(8 + 25, chunk);  // after the signature and IHDR
    util::write_file(png, data);
    auto t = png_text_chunks(png);
    CHECK(t.count("parameters") == 1);
    GenInfo pg = read_gen_info(png, MetaMap{});
    CHECK(pg.tool == "A1111" && pg.prompt == "a lighthouse at night" && pg.seed == "9");
    rm_rf(dir);
    // Midjourney: the description ending in Job ID
    MetaMap mj;
    mj.kv["Xmp.dc.description"] = {"lang=\"x-default\" a castle in the clouds --ar 16:9 Job ID: 1234-abcd"};
    GenInfo m = read_gen_info("/nonexistent.jpg", mj);
    CHECK(m.tool == "Midjourney" && m.prompt == "a castle in the clouds --ar 16:9");
}

static void test_descriptions() {
    CHECK(placeholder_description("OLYMPUS DIGITAL CAMERA", "P1010001"));
    CHECK(placeholder_description("SONY DSC", "x"));
    CHECK(placeholder_description("IMG_2041", "x"));
    CHECK(placeholder_description("P1010001", "P1010001"));
    CHECK(!placeholder_description("Grandma's 80th birthday at the lake house", "x"));
    CHECK(!placeholder_description("sunset", "x"));
    CHECK(description_covers("Grandma's birthday: cake and family at the lake", "Shows: cake, family, lake."));
    CHECK(!description_covers("Grandma at the lake", "Shows: birthday party, cake."));

    // write_meta: fill / keep / append / replace (with backup) / refresh our own text
    std::string dir = util::temp_path("desc");
    util::mkdirs(dir);
    std::string img = dir + "/d.jpg";
    unsigned char px[16 * 16 * 3] = {0};
    CHECK(stbi_write_jpg(img.c_str(), 16, 16, 3, px, 90));
    MetaPlan p;
    p.description = "Shows: dog, beach.";
    p.rewrite_keywords = true;
    WriteResult w = write_meta(img, read_meta(img), p, MetaTarget::Embed, false);
    CHECK(w.ok && w.desc_change == "add");
    CHECK_EQ(meta_description(read_meta(img)), "Shows: dog, beach.");
    p.description = "Shows: dog, beach, sunset.";  // our own earlier text is refreshed
    w = write_meta(img, read_meta(img), p, MetaTarget::Embed, false);
    CHECK(w.desc_change == "update");
    CHECK_EQ(meta_description(read_meta(img)), "Shows: dog, beach, sunset.");
    // someone else's text: kept unless told otherwise
    util::write_file(dir + "/c", "reg jev http://ns.jev.local/photos/1.0/\nset Xmp.dc.description LangAlt lang=\"x-default\" Max at Bondi\ndel Xmp.jev.Description\n");
    CHECK(util::run({"exiv2", "-q", "-m", dir + "/c", img}).rc == 0);
    w = write_meta(img, read_meta(img), p, MetaTarget::Embed, false);
    CHECK(w.desc_change.empty());
    CHECK_EQ(meta_description(read_meta(img)), "Max at Bondi");
    p.desc_action = "append";
    w = write_meta(img, read_meta(img), p, MetaTarget::Embed, false);
    CHECK(w.desc_change == "append");
    CHECK_EQ(meta_description(read_meta(img)), "Max at Bondi \xE2\x80\x94 Shows: dog, beach, sunset.");
    p.description = "Shows: dog, beach, sunset, sea.";  // the appended part is ours: refreshed, theirs stays
    p.desc_action = "";
    w = write_meta(img, read_meta(img), p, MetaTarget::Embed, false);
    CHECK(w.desc_change == "update");
    CHECK_EQ(meta_description(read_meta(img)), "Max at Bondi \xE2\x80\x94 Shows: dog, beach, sunset, sea.");
    util::write_file(dir + "/c", "reg jev http://ns.jev.local/photos/1.0/\nset Xmp.dc.description LangAlt lang=\"x-default\" OLYMPUS DIGITAL CAMERA\ndel Xmp.jev.Description\n");
    CHECK(util::run({"exiv2", "-q", "-m", dir + "/c", img}).rc == 0);
    p.desc_action = "replace";
    w = write_meta(img, read_meta(img), p, MetaTarget::Embed, false);
    MetaMap after = read_meta(img);
    CHECK(w.desc_change == "replace");
    CHECK_EQ(meta_description(after), "Shows: dog, beach, sunset, sea.");
    CHECK_EQ(after.get("Xmp.jev.PreviousDescription"), "OLYMPUS DIGITAL CAMERA");  // backed up
    rm_rf(dir);
}

static void test_query_language() {
    std::string err;
    QNode n = parse_query("(beach | sea) -night tag:dog", err);
    CHECK(err.empty() && n.kind == QNode::AND && n.kids.size() == 3);
    CHECK(n.kids[0].kind == QNode::OR && n.kids[0].kids.size() == 2 && n.kids[0].kids[1].term == "sea");
    CHECK(n.kids[1].kind == QNode::NOT && n.kids[1].kids[0].term == "night");
    CHECK(n.kids[2].field == "tag" && n.kids[2].term == "dog");
    n = parse_query("cat OR dog NOT \"hot dog\"", err);
    CHECK(n.kind == QNode::OR && n.kids.size() == 2 && n.kids[1].kind == QNode::AND && n.kids[1].kids[1].kind == QNode::NOT &&
          n.kids[1].kids[1].kids[0].term == "hot dog");
    CHECK(query_has_logic("beach | sea"));
    CHECK(query_has_logic("is:fav"));
    CHECK(query_has_logic("(a b)"));
    CHECK(!query_has_logic("kids on a beach"));
    CHECK(!query_has_logic("beach -snow"));   // plain search handles -word
    CHECK(!query_has_logic("ratio 16:9"));    // not a field
    std::vector<std::string> t;
    query_terms(parse_query("(海边 | beach) is:fav", err), t);
    CHECK(t.size() == 2 && t[0] == "海边");
    parse_query("(beach", err);
    CHECK(!err.empty());
}

static void test_name_words() {
    auto w = name_words("a_beautiful_sunset_over_the_mountains_with_lake_4k_wallpaper_8213");
    CHECK(w.size() == 6 && w[0] == "beautiful" && w[1] == "sunset" && w[5] == "wallpaper");
    CHECK(long_name(name_prefix("a_beautiful_sunset_over_the_mountains_with_lake_4k_wallpaper_8213")));
    CHECK(!long_name("Paris trip"));
    CHECK(name_words("IMG_20190512_123456").empty());
    CHECK(name_words("KQE7ZGB0M14N0G4E9J6QY3DAQ0").empty());  // an id, not words
    auto z = name_words("东京旅行_2019-05-12_夜景");
    CHECK(z.size() == 2 && z[0] == "东京旅行");
    auto g = name_words("Qwen_image_2.1_00044");
    CHECK(g.size() == 1 && g[0] == "qwen");
}

static void test_corrections() {
    std::string dir = util::temp_path("corr"), err;
    util::mkdirs(dir);
    Db db;
    CHECK(db.open(dir + "/c.sqlite", err));
    Photo p;
    p.src_path = dir + "/a.png";
    p.src_root = dir;
    int64_t id = db.upsert_scan(p);
    db.save_clip(id, "m", {1.0f, 0.0f}, R"([["screenshot",0.7],["woman",0.6]])", "", "v");
    Correction c;
    c.photo_id = id; c.field = "tag"; c.action = "remove"; c.current = "screenshot"; c.confidence = 0.8;
    CHECK(apply_correction(db, c, err));
    Correction add = c;
    add.action = "add"; add.current = ""; add.proposed = "harbor";
    CHECK(apply_correction(db, add, err));
    Photo q;
    CHECK(db.load(id, q));
    auto e = effective_tags(q);
    CHECK(e.size() == 2 && e[0] == "woman" && e[1] == "harbor");  // the original CLIP tags stay stored; the fix is a layer
    CHECK(q.clip_tags.find("screenshot") != std::string::npos);
    // decided proposals are not made again
    db.replace_corrections({id}, {c});
    auto pend = db.corrections("", true);
    CHECK(pend.size() == 1);
    db.set_correction_status(pend[0].id, "rejected");
    CHECK(db.correction_rejected(id, "tag", "screenshot", ""));
    CHECK(db.corrections("", true).empty());
    db.close();
    rm_rf(dir);
}

static void test_search_filters() {
    std::string dir = util::temp_path("sf"), err;
    util::mkdirs(dir);
    Db db;
    CHECK(db.open(dir + "/s.sqlite", err));
    auto add = [&](const std::string& name, int64_t size, int w, int h, const std::string& date, const std::string& tags) {
        Photo p;
        p.src_path = dir + "/" + name;
        p.src_root = dir;
        p.size = size; p.width = w; p.height = h; p.ext = util::ext_lower(name);
        int64_t id = db.upsert_scan(p);
        p.id = id;
        p.date_value = date; p.date_prec = "second"; p.needs_review = false;
        db.save_date(p);
        db.save_clip(id, "m", {1.0f, 0.0f}, tags, "", "v");
        return id;
    };
    add("beach.jpg", 3000000, 4000, 3000, "2019-05-12 10:00:00", R"([["beach",0.8],["sea",0.6]])");
    add("cat.png", 200000, 800, 1200, "2021-01-02 10:00:00", R"([["cat",0.9]])");
    int64_t fav = add("night.jpg", 1500000, 3000, 2000, "2021-07-01 22:00:00", R"([["night city",0.7]])");
    db.set_favorite(fav, true);
    Config c;
    c.clip_enabled = false;
    c.llm_enabled = false;
    Searcher se;
    se.load(db, dir, 0);
    auto n = [&](const std::string& q) {
        SearchQuery sq;
        sq.text = q;
        sq.mode = SM_SMART;
        return se.run(c, sq).hits.size();
    };
    CHECK(n("size:>1mb") == 2);
    CHECK(n("size:100kb..2mb") == 2);
    CHECK(n("w:>=3000") == 2);
    CHECK(n("is:portrait") == 1);
    CHECK(n("ext:png") == 1);
    CHECK(n("date:2021") == 2);
    CHECK(n("date:2019-05") == 1);
    CHECK(n("date:>2020") == 2);
    CHECK(n("year:2018..2020") == 1);
    CHECK(n("is:fav") == 1);
    CHECK(n("(beach | cat) -ext:png") == 1);
    CHECK(n("tag:cat OR tag:\"night city\"") == 2);
    CHECK(n("~beech") == 1);  // a typo still finds beach
    CHECK(n("mp:>10") == 1);  // 12 MP yes, 6 MP no
    db.close();
    rm_rf(dir);
}

static void test_journal() {
    std::string dir = util::temp_path("journal"), err;
    util::mkdirs(dir);
    Db db;
    CHECK(db.open(dir + "/j.sqlite", err));
    for (int r = 0; r < 12; r++)
        for (int k = 0; k < 3; k++) {
            JournalEntry e;
            e.run = util::fmt("2026-10-02T01:%02d:00", r);
            e.kind = k == 2 ? "meta" : "rename";
            e.photo_id = k == 2 ? 1 : k + 1;  // the meta write is to photo 1 again: 2 photos per run
            e.src = "/a" + std::to_string(k);
            e.dst = "/b" + std::to_string(k);
            db.journal_add(e);
        }
    auto runs = db.journal_runs(50);
    CHECK(runs.size() == 12 && runs[0].run == "2026-10-02T01:11:00" && runs[0].count == 2 && !runs[0].undone);
    CHECK(runs[0].kinds.find("meta") != std::string::npos && runs[0].kinds.find("rename") != std::string::npos);
    auto es = db.journal_entries(runs[0].run);
    CHECK(es.size() == 3 && es[0].src == "/a0" && es[2].kind == "meta");
    for (auto& e : es) db.journal_mark_undone(e.id);
    CHECK(db.journal_runs(1)[0].undone);
    db.journal_mark_undone(db.journal_entries(runs[1].run)[0].id);  // partly undone is not undone
    CHECK(!db.journal_runs(2)[1].undone);
    auto old = db.journal_prune(10);
    CHECK(old.size() == 2 && old[0] == "2026-10-02T01:01:00");
    CHECK(db.journal_runs(50).size() == 10);
}

int main() {
    test_journal();
    test_civil();
    test_name_patterns();
    test_decisions();
    test_location();
    test_json_extract();
    test_sha();
    test_meta_roundtrip();
    test_hashes_and_dupes();
    test_thumbs();
    test_date_ranges();
    test_naming();
    test_tags();
    test_genmeta();
    test_descriptions();
    test_query_language();
    test_name_words();
    test_corrections();
    test_search_filters();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
