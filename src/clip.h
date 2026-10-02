// CLIP (ONNX Runtime): image and text embeddings for tagging and semantic search.
//
// Models are the Hugging Face "Xenova/clip-vit-*" ONNX exports:
//   vision_model_quantized.onnx  pixel_values [N,3,224,224] -> image_embeds [N,D]
//   text_model_quantized.onnx    input_ids [N,L] (int64)    -> text_embeds  [N,D]
// plus vocab.json / merges.txt for the byte-level BPE tokenizer. Embeddings are L2-normalised here, so a dot
// product is the cosine similarity.
#pragma once

#include <cstddef>
#include <cstdint>

#include <functional>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "config.h"

using Vec = std::vector<float>;
float dot(const Vec& a, const Vec& b);

class ClipTokenizer {
public:
    bool load(const std::string& dir, std::string& err);
    std::vector<int64_t> encode(const std::string& text, size_t max_len = 77) const;  // <bos> ... <eos>

private:
    std::vector<std::string> bpe(const std::string& word) const;
    std::map<std::string, int64_t> vocab_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    std::string byte_enc_[256];
    mutable std::map<std::string, std::vector<std::string>> cache_;
};

struct ClipInfo {
    std::string id;       // "clip-vit-base-patch32-fp32", or "clip-vit-base-patch32" for the int8 files
    std::string label;    // "ViT-B/32"
    std::string dir;
    std::string vision_file, text_file;  // vision_model.onnx or vision_model_quantized.onnx, ...
};
// Which model a config asks for (auto = ViT-L/14 when its files exist and a GPU provider loads, else ViT-B/32).
ClipInfo clip_choose(const Config& c);
std::string models_dir();
// For the status dot: model files present (and which model / device would be used).
bool clip_status(const Config& c, std::string& detail);
bool clip_files_present(const std::string& dir);
// Download a model's full-precision files from Hugging Face with curl (blocking; run it on a thread).
// which: "b32" (~600 MB) or "l14" (~1.7 GB). status gets a line per file. Returns false with err on failure.
bool clip_download(const std::string& which, std::string& err, const std::function<void(const std::string&)>& status);

class ClipModel {
public:
    ClipModel();
    ~ClipModel();
    // device: 0 auto, 1 CPU, 2 GPU (CUDA). Falls back to CPU when the GPU provider is not available.
    bool load(const ClipInfo& info, int device, int threads, bool need_vision, bool need_text, std::string& err);
    bool loaded() const { return vision_ != nullptr || text_ != nullptr; }
    const std::string& device_used() const { return device_used_; }
    const ClipInfo& info() const { return info_; }

    // images: preprocessed 3x224x224 CHW float each. Returns normalised embeddings.
    bool encode_images(const std::vector<std::vector<float>>& images, std::vector<Vec>& out, std::string& err);
    bool encode_text(const std::string& text, Vec& out, std::string& err);
    bool encode_texts(const std::vector<std::string>& texts, std::vector<Vec>& out, std::string& err);  // batched

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void* vision_ = nullptr;
    void* text_ = nullptr;
    ClipTokenizer tok_;
    ClipInfo info_;
    std::string device_used_;
};

// JPEG/PNG file (use the cached thumbnail) -> CLIP input: shorter side 224, centre crop, normalised CHW floats.
bool clip_preprocess(const std::string& path, std::vector<float>& chw, std::string& err);

// ---- tagging

struct TagDef {
    std::string tag, category;  // category: scene, object, people, event, time, food, document, style
};
// The editable vocabulary: ~/.config/jev-photos/tags.txt ("# category" headers, one tag per line); created from
// the built-in list on first use.
std::vector<TagDef> load_tag_vocabulary();
std::string tag_vocabulary_path();
bool save_tag_vocabulary(const std::vector<TagDef>& tags);  // writes tags.txt (grouped by category, in order)

struct TagIndex {  // text embeddings of the vocabulary prompts (ensembled), for one model
    std::string model_id;
    std::string vocab_hash;  // changes with the vocabulary or the prompts: stored tags from another hash are stale
    std::vector<TagDef> tags;
    std::vector<Vec> vecs;
};
bool build_tag_index(ClipModel& m, TagIndex& idx, std::string& err);  // cached on disk per model + vocabulary
std::string tag_vocabulary_hash(const std::string& model_id);  // what TagIndex::vocab_hash would be, without a model
std::vector<std::string> tag_prompts(const std::string& tag, const std::string& category);
// A search phrase embedded the same way as a tag (prompt ensemble).
bool encode_query(ClipModel& m, const std::string& text, Vec& out, std::string& err);

struct TagScores {  // softmax(100*cos) over the vocabulary, and the probability mass of each category
    std::vector<double> prob;
    std::map<std::string, double> mass;
    double share(const TagIndex& idx, size_t i) const;  // prob inside its own category
};
TagScores score_tags(const TagIndex& idx, const Vec& image);
double presence_scale(const std::string& category);  // category mass that counts as fully present
// Confidence (0..1) that `image` shows `phrase`, scored exactly like a tag: inside the category of the nearest
// vocabulary tag, times that category's presence. Used to verify proposed tags and by search.
double phrase_confidence(ClipModel& m, const TagIndex& idx, const std::string& phrase, const Vec& image, std::string& err);

struct ImageTags {
    std::string scene;
    std::vector<std::pair<std::string, float>> tags;  // tag, confidence 0..1 (share in its category x category presence)
};
ImageTags pick_tags(const TagIndex& idx, const Vec& image, int max_tags);
