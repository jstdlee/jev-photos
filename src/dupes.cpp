#include "dupes.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <regex>

#include "util.h"

bool looks_like_copy(const std::string& path) {
    static const std::regex name_re(R"((\(\d+\)|[ _-]copy( \d+)?|\xE5\x89\xAF\xE6\x9C\xAC.*)$)", std::regex::icase);  // "(1)", " copy", 副本
    static const std::regex dir_re(R"((^|/)(backups?|copy|copies|duplicates?|old|\xE5\xA4\x87\xE4\xBB\xBD|\xE5\x89\xAF\xE6\x9C\xAC)(/|$))",
                                   std::regex::icase);  // 备份, 副本
    std::string stem = util::stem(path), dir = util::dirname(path);
    return std::regex_search(stem, name_re) || std::regex_search(dir, dir_re);
}

int64_t choose_keeper(const std::vector<const DupRow*>& g) {
    if (g.empty()) return 0;
    auto better = [](const DupRow* a, const DupRow* b) {  // true when a should be kept over b
        if (a->keep != b->keep) return a->keep;
        bool ao = !a->dest_path.empty(), bo = !b->dest_path.empty();
        if (ao != bo) return ao;
        if (a->meta_len != b->meta_len) return a->meta_len > b->meta_len;
        bool ac = looks_like_copy(a->src_path), bc = looks_like_copy(b->src_path);
        if (ac != bc) return !ac;
        if (a->mtime != b->mtime && a->mtime > 0 && b->mtime > 0) return a->mtime < b->mtime;  // the original is usually older
        if (a->src_path.size() != b->src_path.size()) return a->src_path.size() < b->src_path.size();
        return a->id < b->id;
    };
    const DupRow* best = g[0];
    for (auto* r : g)
        if (better(r, best)) best = r;
    return best->id;
}

uint64_t dhash_from_gray9x8(const unsigned char* g) {
    uint64_t h = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) h = (h << 1) | (g[y * 9 + x] > g[y * 9 + x + 1] ? 1u : 0u);
    return h;
}

int hamming64(uint64_t a, uint64_t b) { return __builtin_popcountll(a ^ b); }

std::vector<SimilarLink> similar_links(const std::vector<const DupRow*>& rows, int threshold) {
    size_t n = rows.size();
    std::vector<size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<size_t(size_t)> find = [&](size_t x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
    auto aspect_ok = [](const DupRow* a, const DupRow* b) {
        if (a->width <= 0 || a->height <= 0 || b->width <= 0 || b->height <= 0) return true;  // unknown dims: trust the hash
        double ra = double(a->width) / a->height, rb = double(b->width) / b->height;
        auto close = [](double x, double y) { return std::fabs(x - y) / std::max(x, y) < 0.06; };
        return close(ra, rb) || close(ra, 1.0 / rb);  // a rotated copy stored with swapped dimensions
    };
    // Flat images (all-black, all-white frames) hash to ~0 and would match each other; leave them out.
    auto informative = [](uint64_t h) { int b = __builtin_popcountll(h); return b >= 4 && b <= 60; };
    for (size_t i = 0; i < n; i++) {
        if (!informative(rows[i]->dhash)) continue;
        for (size_t j = i + 1; j < n; j++)
            if (informative(rows[j]->dhash) && hamming64(rows[i]->dhash, rows[j]->dhash) <= threshold && aspect_ok(rows[i], rows[j]))
                parent[find(i)] = find(j);
    }
    std::vector<std::vector<size_t>> groups(n);
    for (size_t i = 0; i < n; i++) groups[find(i)].push_back(i);
    std::vector<SimilarLink> out;
    for (auto& g : groups) {
        if (g.size() < 2) continue;
        size_t rep = g[0];
        for (size_t i : g) {
            int64_t pi = int64_t(rows[i]->width) * rows[i]->height, pr = int64_t(rows[rep]->width) * rows[rep]->height;
            if (pi > pr || (pi == pr && rows[i]->size > rows[rep]->size)) rep = i;
        }
        for (size_t i : g)
            if (i != rep) out.push_back({rows[i]->id, rows[rep]->id, hamming64(rows[i]->dhash, rows[rep]->dhash)});
    }
    return out;
}
