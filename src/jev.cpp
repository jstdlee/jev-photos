#include "jev.h"

#include "http.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

JevChoice jev_choice(const Config& c, const std::string& instructions, const std::vector<std::pair<std::string, std::string>>& options,
                     const std::string& state_json) {
    JevChoice r;
    if (options.size() < 2 || options.size() > 20) {
        r.error = "jev needs 2-20 options";
        return r;
    }
    json criteria = json::object();
    for (auto& [id, text] : options) criteria[id] = text;
    json state = json::parse(state_json.empty() ? "{}" : state_json, nullptr, false);
    if (!state.is_object()) state = json::object();
    json req = {{"model", c.jev_model},
                {"state", state},
                {"questions", {{"q", {{"type", "choice"}, {"instructions", instructions}, {"criteria", criteria}}}}}};
    HttpResponse h = http_post_json(url_join(c.jev_url, "/v1/systemone"), req.dump(-1, ' ', false, json::error_handler_t::replace),
                                    c.jev_key, c.jev_timeout);
    if (!h.ok()) {
        r.error = h.error + (h.body.empty() ? "" : ": " + h.body.substr(0, 300));
        return r;
    }
    json j = json::parse(h.body, nullptr, false);
    const json* a = j.is_object() && j.contains("answers") && j["answers"].contains("q") ? &j["answers"]["q"] : nullptr;
    if (!a || !a->contains("probabilities")) {
        r.error = "unexpected jev response: " + h.body.substr(0, 300);
        return r;
    }
    r.choice = a->value("choice", "");
    for (auto& [id, text] : options) {
        const json& p = (*a)["probabilities"];
        r.probs.push_back(p.contains(id) && p[id].is_number() ? p[id].get<double>() : 0.0);
    }
    r.ok = true;
    return r;
}

bool jev_health(const Config& c, std::string& detail) {
    HttpResponse h = http_get(url_join(c.jev_url, "/health"), c.jev_key, 5);
    detail = h.ok() ? "ok" : h.error;
    return h.ok();
}
