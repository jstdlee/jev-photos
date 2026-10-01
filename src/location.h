// Location from folder/file names, decided per folder by a weighted vote of rules, the VL model and jev.
#pragma once

#include <string>
#include <vector>

struct PlaceCandidate {
    std::string text;      // cleaned phrase, e.g. "Paris" from "2019-05 Paris trip"
    std::string origin;    // the folder/file name it came from
    int level = 0;         // 0 = parent folder, 1 = grandparent ..., -1 = file name
    bool travel_cue = false;  // a trip/旅行 affix was stripped: strong hint of a place
    double prior = 0;      // rules score in [0,1]
};

// Folder names nearest first (inside the source root) and the file stem.
std::vector<PlaceCandidate> place_candidates(const std::vector<std::string>& dirs, const std::string& file_stem);

struct PlaceVote {
    std::string voter;                 // rules | vl | jev
    double weight = 0;
    std::vector<double> scores;        // per candidate, same order; plus implicit "none" = 1 - sum
    std::string note;
};

struct PlaceDecision {
    std::string text, source;  // source: "folder:<name>" / "file-name"
    double confidence = 0;
    std::string evidence_json;
};

PlaceDecision decide_place(const std::vector<PlaceCandidate>& cands, const std::vector<PlaceVote>& votes, double accept);
PlaceVote rules_vote(const std::vector<PlaceCandidate>& cands, double weight);
// Maps a free-text answer (e.g. from the VL model) onto candidate indices; -1 when it matches none.
int match_candidate(const std::vector<PlaceCandidate>& cands, const std::string& answer);
