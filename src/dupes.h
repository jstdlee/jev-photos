// Duplicate logic kept pure for testing: keeper choice, perceptual hash, near-duplicate clustering.
//
// Exact duplicates are found in three tiers so most files are never fully read:
//   1. file size (free, from stat)                    -> only same-size files can be identical
//   2. quick hash: XXH3-64 of size + first/last 64 KiB -> reads at most 128 KiB per candidate
//   3. full hash: XXH3-128 of the whole file           -> only for files that still collide
//   4. optional byte-for-byte comparison with the kept copy before anything is marked
// Near duplicates (resized / re-saved / re-compressed copies) use a 64-bit dHash of the oriented image;
// they are reported for review, never skipped automatically.
#pragma once

#include <cstdint>
#include <vector>

#include "db.h"

// Which copy of an exact-duplicate group stays: the one the user pinned, else one already in the library,
// else the richest metadata, else one whose name/folder does not look like a copy ("(1)", " copy", 副本,
// backup/, 备份/), else the oldest mtime, else the shortest path, else the oldest id.
bool looks_like_copy(const std::string& path);
int64_t choose_keeper(const std::vector<const DupRow*>& group);

// dHash from a 9x8 grayscale thumbnail (row-major): bit = left pixel brighter than its right neighbour.
uint64_t dhash_from_gray9x8(const unsigned char* g);
int hamming64(uint64_t a, uint64_t b);

struct SimilarLink {
    int64_t id, to;
    int dist;
};
// Clusters rows whose dHash differs by <= threshold bits and whose aspect ratios match (rotation-aware).
// Each cluster's representative is the largest image (pixels, then bytes); the others link to it.
std::vector<SimilarLink> similar_links(const std::vector<const DupRow*>& rows, int threshold);
