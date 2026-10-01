#include "llm.h"

#include "http.h"
#include "nlohmann/json.hpp"
#include "util.h"

using json = nlohmann::json;

std::string llm_ask(const Config& c, const std::string& prompt, int max_tokens, bool json_reply, std::string& err) {
    json req = {{"model", c.llm_model},
                {"temperature", 0},
                {"max_tokens", max_tokens},
                {"messages", json::array({{{"role", "user"}, {"content", prompt}}})}};
    // Reasoning models spend their whole budget thinking unless told not to (Qwen3 chat template switch).
    if (c.llm_no_think) req["chat_template_kwargs"] = {{"enable_thinking", false}};
    if (json_reply) req["response_format"] = {{"type", "json_object"}};
    HttpResponse h = http_post_json(url_join(c.llm_url, "/chat/completions"), req.dump(-1, ' ', false, json::error_handler_t::replace),
                                    c.llm_key, c.llm_timeout);
    if (!h.ok()) {
        err = h.error + (h.body.empty() ? "" : ": " + h.body.substr(0, 200));
        return "";
    }
    json j = json::parse(h.body, nullptr, false);
    if (!j.is_object() || !j.contains("choices") || j["choices"].empty()) {
        err = "unexpected LLM response: " + h.body.substr(0, 200);
        return "";
    }
    const json& m = j["choices"][0]["message"];
    std::string out = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : "";
    if (out.empty()) err = "the LLM returned no text (reasoning model? keep \"disable thinking\" on)";
    return out;
}

bool llm_health(const Config& c, std::string& detail) {
    HttpResponse h = http_get(url_join(c.llm_url, "/models"), c.llm_key, 5);
    if (!h.ok()) {
        detail = h.error;
        return false;
    }
    json j = json::parse(h.body, nullptr, false);
    bool found = false;
    if (j.contains("data"))
        for (auto& m : j["data"]) found |= m.value("id", "") == c.llm_model;
    detail = found ? c.llm_model + " ready" : "server up, model " + c.llm_model + " not listed";
    return found;
}
