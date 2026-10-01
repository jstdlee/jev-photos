// Photo search over file names, EXIF/IPTC/XMP, descriptions (tags, scene, place, captions) and CLIP embeddings.
//
// Modes
//   Smart     words or meaning; photos with both rank first
//   Words     every word must appear in one of the chosen fields; "quoted phrases", -excluded words
//   Meaning   CLIP: describe what is in the picture
//   Regex     ECMAScript regular expression, case-insensitive
//   Ask       natural language: an LLM extracts keywords, place and dates -> local search keeps candidates scoring
//             >= 50% -> a decider (LLM, or jev) rates them in batches -> ranked list of what was really asked for
//
// "Meaning" is a real match, not just a ranking. The query is scored exactly like a tag: it competes with the tags of
// its own category (the category of the nearest vocabulary tag), and that category must be present in the picture.
// confidence = share inside the category x presence; a photo matches at >= search_min_match. Searching "diagram"
// does not return a website screenshot where "diagram" was a 10% guess next to "website".
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "clip.h"
#include "config.h"
#include "db.h"

enum SearchMode { SM_SMART, SM_KEYWORD, SM_SEMANTIC, SM_REGEX, SM_ASK, SM_COUNT };
const char* search_mode_name(int m);

struct SearchQuery {
    std::string text;
    int mode = SM_SMART;
    bool in_name = true, in_exif = true, in_desc = true, in_prompt = true;  // in_prompt: AI generation prompts
    bool translate = false;  // may ask the LLM for equivalents in other languages (else only cached ones are used)
    std::string date_from, date_to;  // "YYYY-MM-DD" (inclusive), "" = open
    int limit = 1000;
    bool active() const { return !text.empty() || !date_from.empty() || !date_to.empty(); }
};

struct SearchHit {
    int64_t id;
    double score;
    std::string why;
};

struct SearchResult {
    std::vector<SearchHit> hits;
    std::string error, note;
    std::vector<Decision> decisions;  // jev verdicts made during an Ask search (the caller stores them)
};

// Query language (Smart and Words modes):
//   beach sunset          both (AND is implicit; "AND" / "&" also work)
//   beach OR sea, a | b   either
//   -snow, NOT snow, !snow  without
//   (beach | sea) -night  grouping
//   "new york"            phrase
//   tag:dog  name:IMG  desc:..  prompt:..  exif:canon   one field (tag: is an exact tag)
//   is:fav  is:ai  is:untagged
// A term also matches its equivalents in English, Chinese, Japanese and Korean (LLM translations, cached).
struct QNode {
    enum Kind { TERM, AND, OR, NOT } kind = TERM;
    std::string term, field;  // field: "", name, tag, desc, prompt, exif, is
    std::vector<QNode> kids;
};
bool query_has_logic(const std::string& q);
QNode parse_query(const std::string& q, std::string& err);
void query_terms(const QNode& n, std::vector<std::string>& out);  // the free-text leaves

// Does a photo's date (with its precision) overlap [from, to]? Undated photos never do when a range is set.
bool date_in_range(const std::string& date, const std::string& prec, const std::string& from, const std::string& to);

class Searcher {
public:
    void load(Db& db, const std::string& folder, int gen);
    // A term and its equivalents in other languages (cached in the catalog; the LLM only when allow_llm).
    std::vector<std::string> alternates(const Config& cfg, const std::string& term, bool allow_llm);
    void translate(const Config& cfg, const std::vector<std::string>& terms, bool allow_llm, std::string& note);
    SearchResult run(const Config& cfg, const SearchQuery& q);
    size_t size() const { return docs_.size(); }

    struct Query {  // a phrase prepared for matching
        Vec v;
        int cat = -1;       // category it competes in
        int same_tag = -1;  // the phrase is itself a vocabulary tag
        std::string text;
    };
    bool prepare(const Config& cfg, const std::string& phrase, Query& out, std::string& err);
    double match(const Query& q, size_t doc) const;  // confidence 0..1 that photo `doc` shows the phrase

private:
    bool ensure_clip(const Config& cfg, std::string& err);  // text encoder + vocabulary + per-photo normalisers
    SearchResult run_ask(const Config& cfg, const SearchQuery& q, const std::vector<size_t>& pool);
    SearchResult run_logic(const Config& cfg, const SearchQuery& q, const std::vector<size_t>& pool, const QNode& root);
    std::vector<SearchDoc> docs_;
    Db* db_ = nullptr;
    std::map<std::string, std::vector<std::string>> alt_cache_;
    std::vector<std::string> cats_;   // vocabulary categories
    std::vector<int> tag_cat_;        // per tag: index into cats_
    std::vector<double> lse_;         // per doc: log-sum-exp of 100*cos over the whole vocabulary (NaN = no embedding)
    std::vector<double> cat_lse_;     // per doc x category: the same inside each category
    std::string folder_;
    int gen_ = -1;
    std::unique_ptr<ClipModel> text_;
    std::string text_model_id_;
    TagIndex vocab_;
    int lse_gen_ = -2;
};
