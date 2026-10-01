// Capture-date recovery: collect dated evidence, score it, cluster agreeing evidence, pick a winner.
//
// Model (see docs/decision-model.md):
//   1. Every source yields candidates with a base reliability w (EXIF original 0.95 ... file mtime 0.30).
//   2. Sanity rules reject impossible values (future, before min_year, 1970/1980 epoch sentinels) and
//      down-weight suspicious ones (camera-reset Jan 1 dates, times later than the file itself,
//      bulk-copy mtimes).
//   3. Support: S_i = w_i + sum_j w_j * agree(i,j) * independence(i,j). Agreement understands
//      precision (a date-only name agrees with any time that day) and time-zone shifts.
//   4. The best-supported candidate wins; a date-only winner borrows the time of day from an agreeing
//      finer candidate. Filesystem/modify times are upper bounds: being later than the winner is not
//      a contradiction.
//   5. confidence = margin * strength, margin = S_win / (S_win + S_best_contradicting).
//      Low margin -> ask the jev decision API (blended by weight); low confidence -> needs review.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "meta.h"
#include "util.h"

enum DateSource {
    DS_EXIF_ORIGINAL, DS_EXIF_DIGITIZED, DS_XMP_CREATED, DS_GPS, DS_EXIF_MODIFY,
    DS_NAME_DATETIME, DS_NAME_DATE, DS_NAME_UNIX, DS_NAME_MONTH,
    DS_DIR_DATE, DS_DIR_MONTH, DS_DIR_YEAR,
    DS_FS_BIRTH, DS_FS_MTIME, DS_NEIGHBOR, DS_MANUAL, DS_COUNT
};
const char* date_source_id(DateSource s);     // "exif-original", ...
const char* date_source_label(DateSource s);  // human text, used in jev option text too

struct DateCandidate {
    DateSource src = DS_COUNT;
    util::Civil when;
    double base = 0;     // reliability before sanity rules
    double weight = 0;   // after sanity rules (0 = rejected)
    double support = 0;  // S_i
    std::string raw;     // original text / field
    std::vector<std::string> notes;
    bool rejected() const { return weight <= 0; }
};

struct DateInputs {
    std::string filename;                 // base name
    std::vector<std::string> dirs;        // parent dir names, nearest first (inside the source root)
    const MetaMap* meta = nullptr;
    int64_t mtime = 0, btime = 0;         // epoch seconds; 0 = unknown
    bool bulk_mtime = false;              // many files share this mtime minute -> copy time
    int64_t now = 0;
    int min_year = 1900;
};

struct DateDecision {
    util::Civil when;              // invalid when nothing usable
    double confidence = 0;
    double margin = 1;
    std::string source;            // e.g. "exif-original", "name-date+time:fs-mtime"
    std::string decider = "rules"; // rules | rules+jev | manual
    bool needs_review = true;
    std::vector<DateCandidate> candidates;
    std::vector<int> alternatives; // indices of distinct competing cluster anchors, best first (winner first)
    std::string jev_note;
    std::string to_json() const;
};

std::vector<DateCandidate> date_candidates_from_name(const std::string& name, bool is_dir, int level, int64_t now);
std::vector<DateCandidate> collect_date_candidates(const DateInputs& in);
double date_agree(const DateCandidate& a, const DateCandidate& b);
DateDecision decide_date(const DateInputs& in, double review_below);
DateDecision decide_from_candidates(std::vector<DateCandidate> cands, double review_below);

// Short, self-describing option text for the jev choice question (fits Julia's 48-token options).
std::string date_option_text(const DateDecision& d, int cand_index);
// Blend jev probabilities (by alternatives order) into the decision.
void apply_jev_date(DateDecision& d, const std::vector<double>& probs, double weight, double review_below);
