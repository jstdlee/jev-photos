// Client for the jev decision API (djev / laya / julia): POST {base}/v1/systemone.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "config.h"

struct JevChoice {
    bool ok = false;
    std::string error, choice;
    std::vector<double> probs;  // aligned with the options passed in
};

// One choice question. Options are (id, self-describing text); Julia's independent mode judges the text alone,
// so each text must stand on its own. state_json is an object (may be "{}").
JevChoice jev_choice(const Config& c, const std::string& instructions, const std::vector<std::pair<std::string, std::string>>& options,
                     const std::string& state_json);
bool jev_health(const Config& c, std::string& detail);
