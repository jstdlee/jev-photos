// Vision-language model client (OpenAI-compatible chat completions with an inline image).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"

struct VisionResult {
    bool ok = false;
    std::string error, raw;
    std::string caption, scene, landmark, text;
    std::vector<std::string> objects, tags;
    int people = 0;
};

// Loads any image stb can read (else the largest embedded preview via exiv2), applies EXIF orientation,
// downsizes to max_side and returns JPEG bytes.
bool prepare_image(const std::string& path, int orientation, int max_side, std::string& jpeg, std::string& err);
VisionResult vision_describe(const Config& c, const std::string& jpeg);
// 64-bit dHash of the oriented image (for near-duplicate detection); false when it cannot be decoded.
bool image_dhash(const std::string& path, int orientation, uint64_t& hash);

struct VlPlace {
    bool ok = false;
    std::string place, error;
    double confidence = 0;
};
// Text-only: which of these folder/file name parts names a geographic place?
VlPlace vl_place_from_names(const Config& c, const std::vector<std::string>& parts);
bool vl_health(const Config& c, std::string& detail);
bool& vl_loaded_flag();                            // last health check saw the model loaded (Ollama only)
bool vl_preload(const Config& c, std::string& err);  // Ollama: load the model now with num_ctx / keep_alive

// Extracts the first JSON object from model output (tolerates ``` fences and <think> blocks).
std::string extract_json_object(const std::string& s);
