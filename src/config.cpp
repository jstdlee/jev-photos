#include "config.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "util.h"

std::string default_config_path() { return util::config_dir() + "/config.ini"; }

std::string db_path(const Config& c) {
    if (!c.db_override.empty()) return c.db_override;
    const char* xdg = getenv("XDG_DATA_HOME");
    std::string d = (xdg && *xdg ? std::string(xdg) : util::home() + "/.local/share") + "/jev-photos";
    util::mkdirs(d);
    return d + "/catalog.sqlite";
}

std::string output_dir(const Config& c) {
    if (!c.output.empty()) return c.output;
    return c.folder.empty() ? "" : c.folder + "/jev-organized";
}

Config effective(const Config& c) {
    Config e = c;
    if (!c.folder.empty()) {
        e.sources = {c.folder};
        e.library = output_dir(c);
    }
    return e;
}

void remember_folder(Config& c, const std::string& folder) {
    c.folder = folder;
    c.recent.erase(std::remove(c.recent.begin(), c.recent.end(), folder), c.recent.end());
    c.recent.insert(c.recent.begin(), folder);
    if (c.recent.size() > 10) c.recent.resize(10);
}

static float fclamp(const std::string& v, float lo, float hi) { return std::clamp(std::stof(v), lo, hi); }
static int iclamp(const std::string& v, int lo, int hi) { return std::clamp(std::stoi(v), lo, hi); }

void load_config(Config& c, const std::string& path_in) {
    std::ifstream f(path_in.empty() ? default_config_path() : path_in);
    std::string line;
    bool had_sources = false;
    while (std::getline(f, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos || line[0] == '#') continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        try {
            if (k == "lang") c.lang = v;
            else if (k == "font_size") c.font_size = fclamp(v, 10, 32);
            else if (k == "accent") sscanf(v.c_str(), "%f,%f,%f", &c.accent[0], &c.accent[1], &c.accent[2]);
            else if (k == "renderer") c.renderer = iclamp(v, 0, 2);
            else if (k == "folder") c.folder = v;
            else if (k == "recent") { if (!v.empty()) c.recent.push_back(v); }
            else if (k == "fav_tag") { if (!v.empty()) c.fav_tags.push_back(v); }
            else if (k == "output") c.output = v;
            else if (k == "source") { if (c.folder.empty() && !v.empty() && !had_sources) c.folder = v; had_sources = true; }  // old configs
            else if (k == "layout") c.layout = iclamp(v, 0, LAYOUT_COUNT - 1);
            else if (k == "write_mode") c.write_mode = iclamp(v, 0, WRITE_COUNT - 1);
            else if (k == "file_op") c.file_op = iclamp(v, 0, OP_COUNT - 1);
            else if (k == "name_sep") c.name_sep = v == " " || v == "-" || v == "_" || v.empty() ? v : "_";
            else if (k == "sn_digits") c.sn_digits = iclamp(v, 3, 8);
            else if (k == "jev_enabled") c.jev_enabled = v == "1";
            else if (k == "jev_url") c.jev_url = v;
            else if (k == "jev_model") c.jev_model = v;
            else if (k == "jev_key") c.jev_key = v;
            else if (k == "jev_timeout") c.jev_timeout = iclamp(v, 1, 600);
            else if (k == "vl_enabled") c.vl_enabled = v == "1";
            else if (k == "vl_url") c.vl_url = v;
            else if (k == "vl_model") c.vl_model = v;
            else if (k == "vl_key") c.vl_key = v;
            else if (k == "vl_timeout") c.vl_timeout = iclamp(v, 5, 3600);
            else if (k == "vl_max_side") c.vl_max_side = iclamp(v, 256, 1024);
            else if (k == "llm_enabled") c.llm_enabled = v == "1";
            else if (k == "llm_url") c.llm_url = v;
            else if (k == "llm_model") c.llm_model = v;
            else if (k == "llm_key") c.llm_key = v;
            else if (k == "llm_no_think") c.llm_no_think = v == "1";
            else if (k == "llm_timeout") c.llm_timeout = iclamp(v, 5, 1800);
            else if (k == "ask_judge") c.ask_decider = iclamp(v, 0, JUDGE_COUNT - 1);
            else if (k == "ask_batch") c.ask_batch = iclamp(v, 5, 100);
            else if (k == "ask_max_candidates") c.ask_max_candidates = iclamp(v, 10, 1000);
            else if (k == "ask_keep") c.ask_keep = iclamp(v, 0, 100);
            else if (k == "search_min_match") c.search_min_match = fclamp(v, 0.05f, 0.95f);
            else if (k == "name_style") c.name_style = iclamp(v, 0, NAME_COUNT - 1);
            else if (k == "rewrite_tags") c.rewrite_tags = v == "1";
            else if (k == "gen_keywords") c.gen_keywords = v == "1";
            else if (k == "write_description") c.write_description = v == "1";
            else if (k == "desc_policy") c.desc_policy = iclamp(v, 0, DESC_COUNT - 1);
            else if (k == "clip_enabled") c.clip_enabled = v == "1";
            else if (k == "clip_model") c.clip_model = v == "b32" || v == "l14" ? v : "auto";
            else if (k == "clip_device") c.clip_device = iclamp(v, 0, 2);
            else if (k == "clip_threads") c.clip_threads = iclamp(v, 1, 64);
            else if (k == "clip_max_tags") c.clip_max_tags = iclamp(v, 1, 20);
            else if (k == "thumb_side") c.thumb_side = iclamp(v, 256, 1024);
            else if (k == "vl_device") c.vl_device = iclamp(v, 0, 1);
            else if (k == "vl_num_ctx") c.vl_num_ctx = iclamp(v, 2048, 65536);
            else if (k == "vl_keep_alive") c.vl_keep_alive = v.empty() ? "30m" : v;
            else if (k == "vl_concurrency") c.vl_concurrency = iclamp(v, 1, 16);
            else if (k == "vl_json_mode") c.vl_json_mode = v == "1";
            else if (k == "vl_tag_lang") c.vl_tag_lang = v;
            else if (k == "review_below") c.review_below = fclamp(v, 0, 1);
            else if (k == "jev_margin") c.jev_margin = fclamp(v, 0, 1);
            else if (k == "jev_date_weight") c.jev_date_weight = fclamp(v, 0, 1);
            else if (k == "loc_w_rules") c.loc_w_rules = fclamp(v, 0, 1);
            else if (k == "loc_w_vl") c.loc_w_vl = fclamp(v, 0, 1);
            else if (k == "loc_w_jev") c.loc_w_jev = fclamp(v, 0, 1);
            else if (k == "loc_accept") c.loc_accept = fclamp(v, 0, 1);
            else if (k == "min_year") c.min_year = iclamp(v, 1800, 2100);
            else if (k == "scan_threads") c.scan_threads = iclamp(v, 1, 32);
            else if (k == "dry_run") c.dry_run = v == "1";
            else if (k == "preview_first") c.preview_first = v == "1";
            else if (k == "last_path") c.last_path = v;
            else if (k == "verify_dupes") c.verify_dupes = v == "1";
            else if (k == "similar_check") c.similar_check = v == "1";
            else if (k == "similar_threshold") c.similar_threshold = iclamp(v, 0, 20);
        } catch (...) {
            // a malformed value keeps its default
        }
    }
}

void save_config(const Config& c, const std::string& path_in) {
    std::string path = path_in.empty() ? default_config_path() : path_in;
    std::ostringstream f;
    f << "lang=" << c.lang << "\nfont_size=" << c.font_size << "\naccent=" << c.accent[0] << "," << c.accent[1] << ","
      << c.accent[2] << "\n";
    f << "renderer=" << c.renderer << "\nfolder=" << c.folder << "\n";
    for (auto& r : c.recent) f << "recent=" << r << "\n";
    for (auto& t : c.fav_tags) f << "fav_tag=" << t << "\n";
    f << "output=" << c.output << "\nlayout=" << c.layout << "\nwrite_mode=" << c.write_mode << "\nfile_op=" << c.file_op
      << "\nname_sep=" << c.name_sep << "\nsn_digits=" << c.sn_digits << "\nname_style=" << c.name_style << "\nrewrite_tags=" << c.rewrite_tags << "\ngen_keywords=" << c.gen_keywords
      << "\nwrite_description=" << c.write_description << "\ndesc_policy=" << c.desc_policy
      << "\njev_enabled=" << c.jev_enabled << "\njev_url=" << c.jev_url << "\njev_model=" << c.jev_model
      << "\njev_key=" << c.jev_key << "\njev_timeout=" << c.jev_timeout
      << "\nvl_enabled=" << c.vl_enabled << "\nvl_url=" << c.vl_url << "\nvl_model=" << c.vl_model << "\nvl_key=" << c.vl_key
      << "\nvl_timeout=" << c.vl_timeout << "\nvl_max_side=" << c.vl_max_side << "\nllm_enabled=" << c.llm_enabled << "\nllm_url=" << c.llm_url << "\nllm_model=" << c.llm_model << "\nllm_key=" << c.llm_key
      << "\nllm_no_think=" << c.llm_no_think << "\nllm_timeout=" << c.llm_timeout << "\nask_judge=" << c.ask_decider
      << "\nask_batch=" << c.ask_batch << "\nask_max_candidates=" << c.ask_max_candidates << "\nask_keep=" << c.ask_keep
      << "\nsearch_min_match=" << c.search_min_match << "\nclip_enabled=" << c.clip_enabled << "\nclip_model=" << c.clip_model << "\nclip_device=" << c.clip_device
      << "\nclip_threads=" << c.clip_threads << "\nclip_max_tags=" << c.clip_max_tags << "\nthumb_side=" << c.thumb_side << "\nvl_device=" << c.vl_device << "\nvl_num_ctx=" << c.vl_num_ctx << "\nvl_keep_alive=" << c.vl_keep_alive << "\nvl_concurrency=" << c.vl_concurrency
      << "\nvl_json_mode=" << c.vl_json_mode << "\nvl_tag_lang=" << c.vl_tag_lang
      << "\nreview_below=" << c.review_below << "\njev_margin=" << c.jev_margin << "\njev_date_weight=" << c.jev_date_weight
      << "\nloc_w_rules=" << c.loc_w_rules << "\nloc_w_vl=" << c.loc_w_vl << "\nloc_w_jev=" << c.loc_w_jev
      << "\nloc_accept=" << c.loc_accept << "\nmin_year=" << c.min_year << "\nscan_threads=" << c.scan_threads
      << "\ndry_run=" << c.dry_run << "\npreview_first=" << c.preview_first << "\nlast_path=" << c.last_path
      << "\nverify_dupes=" << c.verify_dupes << "\nsimilar_check=" << c.similar_check << "\nsimilar_threshold=" << c.similar_threshold << "\n";
    util::write_file(path, f.str(), 0600);  // may hold API keys
}
