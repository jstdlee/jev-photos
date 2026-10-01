// Generation metadata of AI images: the prompt and settings that image tools embed in the files they save.
//
// Where it lives (read-only: these fields are never written, other tools parse them):
//   ComfyUI            PNG tEXt "prompt" (the API graph as JSON) + "workflow"; WebP/JPEG EXIF Make/Model "prompt:{...}"
//   A1111 / Forge      PNG tEXt "parameters"; JPEG/WebP Exif.Photo.UserComment (also what Civitai downloads carry):
//                      "<prompt>\nNegative prompt: ...\nSteps: 20, Sampler: ..., Seed: ..., Model: ..."
//   NovelAI            PNG tEXt "Description" (prompt) + "Comment" (JSON: prompt, uc, steps, scale, seed, sampler)
//   InvokeAI           PNG tEXt "invokeai_metadata" (JSON)
//   Fooocus, SwarmUI   PNG tEXt "parameters" holding JSON
//   Midjourney         XMP/EXIF description ending in "Job ID: <uuid>"
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "meta.h"

struct GenInfo {
    std::string tool;      // "ComfyUI", "A1111", "NovelAI", "InvokeAI", "Fooocus", "SwarmUI", "Midjourney"; empty = none found
    std::string field;     // where it was found, e.g. "PNG text: prompt"
    std::string prompt, negative;
    std::string model, sampler, scheduler, seed, size;
    int steps = 0;
    double cfg = 0;
    std::vector<std::pair<std::string, double>> loras;  // name, strength
    std::vector<std::string> sources;                   // input images (edits, img2img)
    bool found() const { return !tool.empty(); }
    std::string to_json() const;
    static GenInfo from_json(const std::string& j);
};

// PNG text chunks (tEXt, zTXt, iTXt; compressed ones inflated): keyword -> text.
std::map<std::string, std::string> png_text_chunks(const std::string& path);
// Everything we can find in the file (PNG chunks) and its metadata snapshot (EXIF/XMP via exiv2).
GenInfo read_gen_info(const std::string& path, const MetaMap& meta);

GenInfo parse_a1111(const std::string& text);         // "parameters" / UserComment text
GenInfo parse_comfy_prompt(const std::string& json);  // ComfyUI API graph

// The prompt as a readable description: <lora:...>, (word:1.2) weights, BREAK and embeddings removed.
std::string clean_prompt(const std::string& prompt);
// Tag-style prompts ("1girl, solo, hat, beach") -> their short phrases; empty for prose prompts.
std::vector<std::string> prompt_keywords(const std::string& prompt, size_t max_n = 20);
// Without an LLM (or when it declines): the prompt's most frequent meaningful words, and short CJK phrases.
std::vector<std::string> local_keywords(const std::string& prompt, size_t max_n = 10);
