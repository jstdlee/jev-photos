// Settings, persisted to ~/.config/jev-photos/config.ini (plain key=value, like gpu-hud).
#pragma once

#include <string>
#include <vector>

enum Layout { LAYOUT_Y_YM, LAYOUT_YM, LAYOUT_Y_M, LAYOUT_COUNT };  // 2019/2019-05, 2019-05, 2019/05
enum WriteMode { WRITE_EMBED, WRITE_SIDECAR, WRITE_DB_ONLY, WRITE_COUNT };
// copy+rename into month folders, move+rename (no copies), rename in place, or only add missing metadata
enum FileOp { OP_COPY, OP_MOVE, OP_RENAME, OP_METADATA, OP_COUNT };
enum NameStyle { NAME_DATE_SN, NAME_KEEP_PREFIX, NAME_COUNT };  // 20190512_00001, IMG_20190512_00001
// Existing description in a file: only ever fill an empty one; rules + LLM, with jev as a cross-check (they must agree,
// else you decide in Review); rules + LLM alone; or always ask.
enum DescPolicy { DESC_FILL_ONLY, DESC_AGREE, DESC_LLM, DESC_ASK, DESC_COUNT };
enum AskJudge { JUDGE_LLM, JUDGE_JEV, JUDGE_NONE, JUDGE_LLM_JEV, JUDGE_COUNT };

enum Theme { THEME_DARK, THEME_TOKYO, THEME_LIGHT, THEME_COUNT };

struct Config {
    std::string lang;  // "en" / "zh_CN"; empty follows $LANG
    float font_size = 16.0f;
    float accent[3] = {0.30f, 0.78f, 0.47f};
    int theme = THEME_DARK;            // Dark, Tokyo Night, Light
    bool system_titlebar = false;      // use the desktop's title bar instead of our own top bar
    bool first_run_done = false;       // the welcome screen was completed (or folders existed already)
    int goal = 0;                      // welcome choice: 0 browse only, 1 organized copies, 2 rename originals
    bool watch_folders = true;
    bool auto_meta = false;
    int motion = 0;                    // 0 follow the system, 1 full, 2 reduced (fades only, no movement)            // after each Analyze, write missing dates, keywords, descriptions, places into the files         // look for new photos in the folders now and then, and analyse them quietly
    int renderer = 0;  // 0 auto (GPU, software fallback if the driver crashes at start), 1 GPU only, 2 software

    // The photo folders (scanned together) and the one being looked at. folder is the view: one of folders, or
    // "" with view_all = every folder at once. Organized copies go to <each folder>/jev-organized unless output is set.
    std::vector<std::string> folders;
    std::string folder;
    bool view_all = false;
    std::vector<std::string> recent;   // recently used folders, newest first
    std::vector<std::string> fav_tags; // starred tags: one-click filters
    std::string output;                // optional override for the organized-copies folder
    std::string db_override;           // tests / CLI --db; empty = the shared catalog

    // Derived per run by effective(): what the pipeline scans and where it writes.
    std::vector<std::string> sources;
    std::string library;
    int layout = LAYOUT_Y_YM;
    int write_mode = WRITE_EMBED;
    int file_op = OP_RENAME;   
    std::string name_sep = "_";        // 20190512_00001.jpg
    int sn_digits = 5;
    int name_style = NAME_DATE_SN;     // NAME_KEEP_PREFIX: the original name without its dates/numbers, then date + sn
    bool rewrite_tags = true;          // keywords jev-photos wrote earlier are replaced by the current tags
    // AI images: keywords from the generation prompt (tag-style prompts directly, prose prompts via the LLM)
    bool gen_keywords = true;
    // Description field: what to write, and what to do when the file already has one
    bool write_description = true;     // write a description (the prompt for AI images, plus "Shows: <tags>")
    int desc_policy = DESC_AGREE;      // see DescPolicy

    // Jev decision API (djev / laya / julia envelope: POST {base}/v1/systemone)
    bool jev_enabled = true;
    std::string jev_url = "http://127.0.0.1:8011";
    std::string jev_model = "julia-1";
    std::string jev_key;
    int jev_timeout = 30;

    // Vision-language model (OpenAI-compatible /chat/completions: Ollama, vLLM, llama.cpp, OpenAI ...)
    bool vl_enabled = false;  // advanced: captions / text reading with a vision-language model
    std::string vl_url = "http://127.0.0.1:11434/v1";
    std::string vl_model = "qwen3-vl:2b-instruct";
    std::string vl_key;
    int vl_timeout = 180;
    int vl_max_side = 1024;       // thumbnail side for screenshots / text images (never more than 1024)
    int thumb_side = 512;         // thumbnail side for photos: what the model, similar-photo check and UI use
    int vl_device = 0;            // Ollama: 0 auto (GPU when it fits), 1 CPU only
    int vl_num_ctx = 4096;        // Ollama context; a <=1024 px photo needs ~1-1.5k tokens
    std::string vl_keep_alive = "30m";
    int vl_concurrency = 1;
    bool vl_json_mode = true;
    std::string vl_tag_lang = "English";  // language for caption/tags

    // CLIP: default tagger + semantic search index (ONNX Runtime, CPU by default)
    bool clip_enabled = true;
    std::string clip_model = "auto";  // auto | b32 | l14
    int clip_device = 0;              // 0 auto, 1 CPU, 2 GPU
    int clip_threads = 8;             // CPU threads for CLIP (the machine is shared)
    int clip_max_tags = 8;

    // Ask search: an LLM extracts keywords/conditions, local search narrows, a decider rates the candidates
    bool llm_enabled = true;
    std::string llm_url = "http://127.0.0.1:8888/v1";
    std::string llm_model = "Qwen3.8-Flash-Next";
    std::string llm_key;
    bool llm_no_think = true;     // Qwen3-style chat_template_kwargs.enable_thinking=false
    int llm_timeout = 120;
    int ask_decider = JUDGE_LLM_JEV;  // who rates the candidates; LLM + jev = the LLM rates, jev settles its close calls
    int ask_batch = 25;           // candidates per decision request
    int ask_max_candidates = 100;
    int ask_keep = 50;            // keep candidates rated at least this (0-100)
    float search_min_match = 0.30f;  // meaning match: the query's confidence against its tag category (as for tags)

    // Decision model
    float review_below = 0.55f;   // date confidence under this -> "needs review"
    float jev_margin = 0.70f;     // ask jev when the winning date cluster's margin is below this
    float jev_date_weight = 0.30f;
    float loc_w_rules = 0.25f, loc_w_vl = 0.60f, loc_w_jev = 0.15f;
    float loc_accept = 0.50f;
    int min_year = 1900;
    int scan_threads = 4;
    bool dry_run = false;
    bool preview_first = true;
    bool analyze_ask = true;      // GUI: Analyze first asks how much image analysis and tagging to do
    bool verify_dupes = true;     // byte-compare exact duplicates before marking them
    bool similar_check = false;   // also find near duplicates (decodes every image once)
    int similar_threshold = 6;    // max dHash bit difference for "similar"  // GUI: organize stops at the preview; changes need "Apply"
    std::string last_path;     // GUI: folder chosen for "Process path"
};

void load_config(Config& c, const std::string& path = "");
void save_config(const Config& c, const std::string& path = "");
std::string default_config_path();
std::string db_path(const Config& c);     // shared catalog: $XDG_DATA_HOME/jev-photos/catalog.sqlite
std::string output_dir(const Config& c);  // output, else <folder>/jev-organized (the viewed folder)
std::string library_for(const Config& c, const std::string& src_root);  // where a photo's organized copy goes
std::string view_scope(const Config& c);   // the viewed folder, or all folders one per line (for filters)
std::string all_scope(const Config& c);    // every folder, one per line (what Analyze scans)
void add_folder(Config& c, const std::string& folder);
void remove_folder(Config& c, const std::string& folder);
Config effective(const Config& c);        // sources = folders, library = output (empty = per folder)
void remember_folder(Config& c, const std::string& folder);
