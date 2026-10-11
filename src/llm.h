// Chat LLM (OpenAI-compatible /chat/completions) used by the "Ask" search: it turns a natural-language request
// into keywords and conditions, and rates candidate photos. Default: the local Qwen3.8 server on :8888.
#pragma once

#include <string>
#include <vector>

#include "config.h"

// One user message -> the reply text. json_reply asks for a JSON object. Empty on failure (err set).
std::string llm_ask(const Config& c, const std::string& prompt, int max_tokens, bool json_reply, std::string& err);
bool llm_health(const Config& c, std::string& detail);
// The model names an OpenAI-compatible server lists at <url>/models (Ollama, vLLM, llama.cpp ...). False when it cannot be reached.
bool list_models(const std::string& url, const std::string& key, std::vector<std::string>& names, std::string& err);
