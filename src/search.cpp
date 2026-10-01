#include "search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <regex>
#include <functional>
#include <set>

#include "http.h"
#include "jev.h"
#include "llm.h"
#include "nlohmann/json.hpp"
#include "util.h"
#include "vision.h"

using json = nlohmann::json;

const char* search_mode_name(int m) {
    switch (m) {
        case SM_SMART: return "smart";
        case SM_KEYWORD: return "keyword";
        case SM_SEMANTIC: return "semantic";
        case SM_REGEX: return "regex";
        case SM_ASK: return "ask";
        default: return "?";
    }
}

bool date_in_range(const std::string& date, const std::string& prec, const std::string& from, const std::string& to) {
    if (from.empty() && to.empty()) return true;
    if (date.size() < 10) return false;
    // The photo's own period: a year-only date covers the whole year, a month-only date the whole month.
    std::string a = date.substr(0, 10), b = a;
    if (prec == "year") { a = date.substr(0, 4) + "-01-01"; b = date.substr(0, 4) + "-12-31"; }
    else if (prec == "month") { a = date.substr(0, 7) + "-01"; b = date.substr(0, 7) + "-31"; }
    return (from.empty() || b >= from) && (to.empty() || a <= to);
}

void Searcher::load(Db& db, const std::string& folder, int gen) {
    db_ = &db;
    if (gen == gen_ && folder == folder_) return;
    docs_ = db.search_docs(folder);
    folder_ = folder;
    gen_ = gen;
    lse_gen_ = -2;  // normalisers must be recomputed for the new documents
}

bool Searcher::ensure_clip(const Config& cfg, std::string& err) {
    ClipInfo info = clip_choose(cfg);
    if (!text_ || text_model_id_ != info.id) {
        auto m = std::make_unique<ClipModel>();
        if (!m->load(info, cfg.clip_device, std::max(1, cfg.clip_threads / 2), false, true, err)) return false;
        text_ = std::move(m);
        text_model_id_ = info.id;
        vocab_ = TagIndex();
        lse_gen_ = -2;
    }
    if (vocab_.vecs.empty()) {
        if (!build_tag_index(*text_, vocab_, err)) return false;
        cats_.clear();
        tag_cat_.clear();
        for (auto& t : vocab_.tags) {
            auto it = std::find(cats_.begin(), cats_.end(), t.category);
            if (it == cats_.end()) { cats_.push_back(t.category); it = cats_.end() - 1; }
            tag_cat_.push_back(int(it - cats_.begin()));
        }
        lse_gen_ = -2;
    }
    if (lse_gen_ != gen_) {
        // Per photo: how strongly the whole vocabulary, and each category, matches it.
        size_t nc = cats_.size();
        lse_.assign(docs_.size(), NAN);
        cat_lse_.assign(docs_.size() * nc, -1e300);
        std::vector<double> l(vocab_.vecs.size()), sum(nc);
        for (size_t i = 0; i < docs_.size(); i++) {
            const SearchDoc& d = docs_[i];
            if (d.vec.empty() || d.clip_model != text_model_id_) continue;
            double mx = -1e9, all = 0;
            for (size_t k = 0; k < l.size(); k++) mx = std::max(mx, l[k] = 100.0 * dot(d.vec, vocab_.vecs[k]));
            std::fill(sum.begin(), sum.end(), 0.0);
            for (size_t k = 0; k < l.size(); k++) {
                double e = std::exp(l[k] - mx);
                all += e;
                sum[size_t(tag_cat_[k])] += e;
            }
            lse_[i] = mx + std::log(all);
            for (size_t c = 0; c < nc; c++)
                if (sum[c] > 0) cat_lse_[i * nc + c] = mx + std::log(sum[c]);
        }
        lse_gen_ = gen_;
    }
    return true;
}

bool Searcher::prepare(const Config& cfg, const std::string& phrase, Query& q, std::string& err) {
    q = Query();
    q.text = util::lower(util::trim(phrase));
    if (q.text.empty() || !ensure_clip(cfg, err)) return false;
    for (size_t k = 0; k < vocab_.tags.size(); k++)
        if (util::lower(vocab_.tags[k].tag) == q.text) q.same_tag = int(k);
    if (q.same_tag >= 0) {
        q.v = vocab_.vecs[size_t(q.same_tag)];
        q.cat = tag_cat_[size_t(q.same_tag)];
        return true;
    }
    if (!encode_query(*text_, q.text, q.v, err)) return false;
    // It competes where the nearest tag lives ("puppy" with the animals, "invoice" with the documents).
    double best = -2;
    for (size_t k = 0; k < vocab_.vecs.size(); k++) {
        double c = dot(q.v, vocab_.vecs[k]);
        if (c > best) { best = c; q.cat = tag_cat_[k]; }
    }
    return true;
}

double Searcher::match(const Query& q, size_t i) const {
    if (i >= lse_.size() || std::isnan(lse_[i]) || q.cat < 0) return 0;
    size_t nc = cats_.size();
    double all = lse_[i], cat = cat_lse_[i * nc + size_t(q.cat)];
    double lq = 100.0 * dot(q.v, docs_[i].vec);
    double share, mass;
    if (q.same_tag >= 0) {  // a tag already in the vocabulary: exactly its tag confidence
        share = std::exp(lq - cat);
        mass = std::exp(cat - all);
    } else {                // a new phrase joins its category
        double m = std::max({all, cat, lq});
        double e_all = std::exp(all - m), e_cat = std::exp(cat - m), e_q = std::exp(lq - m);
        share = e_q / (e_cat + e_q);
        mass = (e_cat + e_q) / (e_all + e_q);
    }
    if (cats_[size_t(q.cat)] == "medium") return std::clamp(share, 0.0, 1.0);  // every picture has a medium
    return std::clamp(share * std::min(1.0, mass / presence_scale(cats_[size_t(q.cat)])), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Query language

namespace {
const std::set<std::string> kFields = {"name", "tag", "desc", "prompt", "exif", "is", "has", "date", "year", "size", "width", "w", "height",
                                       "h", "mp", "ext", "type", "place", "loc", "camera", "like", "model", "lora"};

struct Tok {
    enum K { WORD, LP, RP, OR, AND, NOT } k;
    std::string text, field;
};

std::vector<Tok> tokenize(const std::string& q) {
    std::vector<Tok> out;
    size_t i = 0;
    while (i < q.size()) {
        char ch = q[i];
        if (isspace((unsigned char)ch)) { i++; continue; }
        if (ch == '(') { out.push_back({Tok::LP, "", ""}); i++; continue; }
        if (ch == ')') { out.push_back({Tok::RP, "", ""}); i++; continue; }
        if (ch == '|') { out.push_back({Tok::OR, "", ""}); i++; continue; }
        if (ch == '&') { out.push_back({Tok::AND, "", ""}); i++; continue; }
        if ((ch == '!' || ch == '-') && i + 1 < q.size() && !isspace((unsigned char)q[i + 1])) { out.push_back({Tok::NOT, "", ""}); i++; continue; }
        if (ch == '~' && i + 1 < q.size() && !isspace((unsigned char)q[i + 1])) {  // ~word: close spellings too
            size_t e = i + 1;
            while (e < q.size() && !isspace((unsigned char)q[e]) && q[e] != ')' && q[e] != '|') e++;
            out.push_back({Tok::WORD, util::lower(q.substr(i + 1, e - i - 1)), "fuzzy"});
            i = e;
            continue;
        }
        std::string field;
        // field:value
        size_t colon = q.find(':', i);
        if (colon != std::string::npos) {
            std::string f = util::lower(q.substr(i, colon - i));
            if (kFields.count(f) && colon + 1 < q.size() && !isspace((unsigned char)q[colon + 1])) { field = f; i = colon + 1; }
        }
        std::string w;
        if (i < q.size() && q[i] == '"') {
            size_t e = q.find('"', i + 1);
            w = q.substr(i + 1, e == std::string::npos ? std::string::npos : e - i - 1);
            i = e == std::string::npos ? q.size() : e + 1;
        } else {
            while (i < q.size() && !isspace((unsigned char)q[i]) && q[i] != '(' && q[i] != ')' && q[i] != '|' && q[i] != '&') w += q[i++];
        }
        if (field.empty() && (w == "OR" || w == "or")) out.push_back({Tok::OR, "", ""});
        else if (field.empty() && (w == "AND" || w == "and")) out.push_back({Tok::AND, "", ""});
        else if (field.empty() && (w == "NOT" || w == "not")) out.push_back({Tok::NOT, "", ""});
        else if (!w.empty()) out.push_back({Tok::WORD, util::lower(w), field});
    }
    return out;
}

struct Parser {
    std::vector<Tok> t;
    size_t i = 0;
    std::string err;
    bool at(Tok::K k) const { return i < t.size() && t[i].k == k; }
    QNode expr() {
        QNode a = conj();
        if (!at(Tok::OR)) return a;
        QNode n;
        n.kind = QNode::OR;
        n.kids.push_back(a);
        while (at(Tok::OR)) { i++; n.kids.push_back(conj()); }
        return n;
    }
    QNode conj() {
        QNode n;
        n.kind = QNode::AND;
        while (i < t.size() && !at(Tok::OR) && !at(Tok::RP)) {
            if (at(Tok::AND)) { i++; continue; }
            n.kids.push_back(unary());
        }
        if (n.kids.size() == 1) return n.kids[0];
        return n;
    }
    QNode unary() {
        if (at(Tok::NOT)) {
            i++;
            QNode n;
            n.kind = QNode::NOT;
            n.kids.push_back(unary());
            return n;
        }
        if (at(Tok::LP)) {
            i++;
            QNode n = expr();
            if (at(Tok::RP)) i++;
            else err = "missing )";
            return n;
        }
        QNode n;
        if (i < t.size() && t[i].k == Tok::WORD) { n.term = t[i].text; n.field = t[i].field; }
        i++;
        return n;
    }
};
}  // namespace

bool query_has_logic(const std::string& q) {
    for (auto& t : tokenize(q))
        if (t.k == Tok::LP || t.k == Tok::RP || t.k == Tok::OR || t.k == Tok::AND || !t.field.empty()) return true;
    for (auto& t : tokenize(q))  // NOT written as a word (the "-word" form is handled by plain search too)
        if (t.k == Tok::NOT) return q.find("NOT") != std::string::npos || q.find("not ") != std::string::npos || q.find('!') != std::string::npos;
    return false;
}

QNode parse_query(const std::string& q, std::string& err) {
    Parser p;
    p.t = tokenize(q);
    QNode n = p.expr();
    if (p.i < p.t.size() && p.err.empty()) p.err = "unexpected )";
    err = p.err;
    return n;
}

void query_terms(const QNode& n, std::vector<std::string>& out) {
    if (n.kind == QNode::TERM) {
        static const std::set<std::string> free_text = {"", "name", "tag", "desc", "prompt", "place", "loc"};
        if (!n.term.empty() && free_text.count(n.field)) out.push_back(n.term);
    } else {
        for (auto& k : n.kids) query_terms(k, out);
    }
}

// ---------------------------------------------------------------------------
// Other languages: a term's equivalents in English, Chinese, Japanese and Korean.

static bool ascii_only(const std::string& s) {
    for (unsigned char ch : s)
        if (ch >= 0x80) return false;
    return true;
}

std::vector<std::string> Searcher::alternates(const Config& cfg, const std::string& term, bool allow_llm) {
    std::string t = util::lower(util::trim(term));
    auto it = alt_cache_.find(t);
    if (it != alt_cache_.end()) return it->second;
    if (db_) {
        std::string j = db_->translation(t);
        if (!j.empty()) {
            std::vector<std::string> v{t};
            json a = json::parse(j, nullptr, false);
            if (a.is_array())
                for (auto& x : a)
                    if (x.is_string() && util::lower(x.get<std::string>()) != t) v.push_back(util::lower(x.get<std::string>()));
            return alt_cache_[t] = v;
        }
    }
    (void)cfg;
    (void)allow_llm;
    return {t};
}

void Searcher::translate(const Config& cfg, const std::vector<std::string>& terms, bool allow_llm, std::string& note) {
    std::vector<std::string> todo;
    for (auto& raw : terms) {
        std::string t = util::lower(util::trim(raw));
        bool digits = !t.empty() && std::all_of(t.begin(), t.end(), [](unsigned char ch) { return isdigit(ch) || ch == '-' || ch == '_'; });
        if (t.empty() || digits || util::utf8_len(t) < 2 || alternates(cfg, t, false).size() > 1) continue;
        if (db_ && !db_->translation(t).empty()) continue;  // known: no equivalents
        if (std::find(todo.begin(), todo.end(), t) == todo.end()) todo.push_back(t);
    }
    if (todo.empty() || !allow_llm || !cfg.llm_enabled) return;
    json list = todo;
    std::string err, reply = llm_ask(cfg,
                                     "Photo search terms: " + list.dump() +
                                         ". For each term give the words people would use for the same thing in English, Simplified Chinese, "
                                         "Traditional Chinese, Japanese and Korean (the translation, plus at most two common synonyms; "
                                         "lower case). Reply JSON only: {\"<term>\": [\"...\"], ...}",
                                     120 + 80 * int(todo.size()), true, err);
    json j = json::parse(extract_json_object(reply), nullptr, false);
    if (!j.is_object()) {
        note = "other languages: LLM unavailable" + (err.empty() ? std::string() : " (" + err + ")");
        return;
    }
    for (auto& t : todo) {
        json alts = json::array();
        if (j.contains(t) && j[t].is_array())
            for (auto& x : j[t])
                if (x.is_string() && !util::trim(x.get<std::string>()).empty() && alts.size() < 12) alts.push_back(util::lower(util::trim(x.get<std::string>())));
        if (db_) db_->save_translation(t, alts.dump());
        alt_cache_.erase(t);
    }
}

namespace {

struct Term {
    std::string text;
    bool exclude = false;
};

std::vector<Term> parse_terms(const std::string& q) {
    std::vector<Term> out;
    for (size_t i = 0; i < q.size();) {
        while (i < q.size() && isspace((unsigned char)q[i])) i++;
        if (i >= q.size()) break;
        Term t;
        if (q[i] == '-' && i + 1 < q.size() && !isspace((unsigned char)q[i + 1])) { t.exclude = true; i++; }
        if (q[i] == '"') {
            size_t e = q.find('"', i + 1);
            t.text = q.substr(i + 1, e == std::string::npos ? std::string::npos : e - i - 1);
            i = e == std::string::npos ? q.size() : e + 1;
        } else {
            size_t e = i;
            while (e < q.size() && !isspace((unsigned char)q[e])) e++;
            t.text = q.substr(i, e - i);
            i = e;
        }
        t.text = util::lower(util::trim(t.text));
        if (!t.text.empty()) out.push_back(t);
    }
    return out;
}

std::string excerpt(const std::string& text_lower, const std::string& raw, size_t pos, size_t len) {
    const std::string& src = raw.size() == text_lower.size() ? raw : text_lower;
    size_t a = pos > 24 ? pos - 24 : 0, b = std::min(src.size(), pos + len + 24);
    while (a > 0 && (static_cast<unsigned char>(src[a]) & 0xC0) == 0x80) a--;
    while (b < src.size() && (static_cast<unsigned char>(src[b]) & 0xC0) == 0x80) b++;
    std::string s = src.substr(a, b - a);
    for (auto& c : s)
        if (c == '\n') c = ' ';
    return (a > 0 ? "…" : "") + util::trim(s) + (b < src.size() ? "…" : "");
}

std::string exif_line(const std::string& exif, size_t pos) {
    size_t a = exif.rfind('\n', pos), b = exif.find('\n', pos);
    a = a == std::string::npos ? 0 : a + 1;
    std::string l = exif.substr(a, (b == std::string::npos ? exif.size() : b) - a);
    return l.size() > 80 ? l.substr(0, 80) + "…" : l;
}

// Where does term t appear (fields enabled by q)? Fills why; returns a weight (0 = nowhere).
double term_hit(const SearchDoc& d, const SearchQuery& q, const std::string& t, std::string* why) {
    size_t pn = q.in_name ? d.name.find(t) : std::string::npos;
    size_t pd = q.in_desc ? d.desc.find(t) : std::string::npos;
    size_t pe = q.in_exif ? d.exif.find(t) : std::string::npos;
    size_t pp = q.in_prompt ? d.prompt.find(t) : std::string::npos;
    if (why && why->size() < 90) {
        std::string w = pd != std::string::npos ? "desc: " + excerpt(d.desc, d.desc_raw, pd, t.size())
                      : pp != std::string::npos ? "prompt: " + excerpt(d.prompt, d.prompt_raw, pp, t.size())
                      : pn != std::string::npos ? "name: " + util::basename(d.name_raw.substr(0, d.name_raw.find('\n')))
                      : pe != std::string::npos ? "exif: " + exif_line(d.exif, pe) : "";
        if (!w.empty() && why->find(w) == std::string::npos) *why += (why->empty() ? "" : " · ") + w;
    }
    // Descriptions count most (they describe the picture), then the prompt the picture was made from, names, metadata.
    return (pd != std::string::npos ? 1.5 : 0) + (pp != std::string::npos ? 1.2 : 0) + (pn != std::string::npos ? 1.0 : 0) +
           (pe != std::string::npos ? 0.8 : 0);
}

// Compact description of a photo for the decider.
std::string describe(const SearchDoc& d) {
    std::string name = util::basename(d.name_raw.substr(0, d.name_raw.find('\n')));
    std::string desc = d.desc_raw.size() > 220 ? d.desc_raw.substr(0, 220) : d.desc_raw;
    for (auto& c : desc)
        if (c == '\n') c = ' ';
    if (!d.prompt_raw.empty()) {  // what the picture was generated from says a lot about it
        std::string pr = d.prompt_raw.substr(0, d.prompt_raw.find("\nNegative: "));
        if (pr.size() > 240) {
            size_t cut = 240;
            while (cut > 0 && (static_cast<unsigned char>(pr[cut]) & 0xC0) == 0x80) cut--;
            pr = pr.substr(0, cut) + "…";
        }
        for (auto& c : pr)
            if (c == '\n') c = ' ';
        desc += " | prompt: " + pr;
    }
    return desc + " | date " + (d.date.empty() ? "unknown" : d.date.substr(0, d.prec == "year" ? 4 : d.prec == "month" ? 7 : 10)) +
           " | file " + name;
}

}  // namespace

SearchResult Searcher::run(const Config& cfg, const SearchQuery& q) {
    SearchResult res;
    std::string text = util::trim(q.text);
    std::vector<size_t> pool;
    for (size_t i = 0; i < docs_.size(); i++)
        if (date_in_range(docs_[i].date, docs_[i].prec, q.date_from, q.date_to)) pool.push_back(i);

    if (text.empty()) {  // date range only
        for (size_t i : pool) res.hits.push_back({docs_[i].id, 0, docs_[i].date.substr(0, 10)});
        return res;
    }
    if (q.mode == SM_ASK) return run_ask(cfg, q, pool);

    // ---- regex
    if (q.mode == SM_REGEX) {
        std::regex re;
        try {
            re = std::regex(text, std::regex::ECMAScript | std::regex::icase | std::regex::optimize);
        } catch (const std::regex_error& e) {
            res.error = std::string("invalid regular expression: ") + e.what();
            return res;
        }
        for (size_t i : pool) {
            const SearchDoc& d = docs_[i];
            std::smatch m;
            std::string why;
            if (q.in_name && std::regex_search(d.name_raw, m, re)) why = "name: " + m.str(0).substr(0, 80);
            else if (q.in_desc && std::regex_search(d.desc_raw, m, re)) why = "desc: " + excerpt(d.desc, d.desc_raw, size_t(m.position(0)), size_t(m.length(0)));
            else if (q.in_prompt && std::regex_search(d.prompt_raw, m, re)) why = "prompt: " + excerpt(d.prompt, d.prompt_raw, size_t(m.position(0)), size_t(m.length(0)));
            else if (q.in_exif && std::regex_search(d.exif, m, re)) why = "exif: " + exif_line(d.exif, size_t(m.position(0)));
            if (!why.empty()) res.hits.push_back({d.id, 1.0, why});
            if (int(res.hits.size()) >= q.limit) break;
        }
        return res;
    }

    // ---- other languages: equivalents of every term (cached; the LLM only once typing has paused)
    bool logic = q.mode != SM_SEMANTIC && query_has_logic(text);
    QNode root;
    std::vector<std::string> leaves;
    if (logic) {
        std::string perr;
        root = parse_query(text, perr);
        if (!perr.empty()) res.note = "query: " + perr;
        query_terms(root, leaves);
    } else {
        for (auto& t : parse_terms(text)) leaves.push_back(t.text);
        if (leaves.size() > 1) leaves.push_back(util::lower(text));  // the whole phrase too ("海边日落")
    }
    std::string tnote;
    translate(cfg, leaves, q.translate, tnote);
    if (!tnote.empty()) res.note += (res.note.empty() ? "" : " · ") + tnote;
    if (logic) {
        SearchResult r = run_logic(cfg, q, pool, root);
        if (!res.note.empty()) r.note = res.note + (r.note.empty() ? "" : " · " + r.note);
        return r;
    }
    // A term matches if it or one of its equivalents appears.
    auto hit_any = [&](const SearchDoc& d, const std::string& term, std::string* why) {
        double best = 0;
        for (auto& a : alternates(cfg, term, false)) best = std::max(best, term_hit(d, q, a, why));
        return best;
    };

    // ---- words
    std::vector<Term> terms = parse_terms(text);
    int positives = 0;
    for (auto& t : terms) positives += !t.exclude;
    std::vector<double> kw(docs_.size(), 0.0);
    std::vector<std::string> kw_why(docs_.size());
    std::vector<bool> excluded(docs_.size(), false), all_terms(docs_.size(), false);
    for (size_t i : pool) {
        double score = 0;
        int matched = 0;
        for (auto& t : terms) {
            double w = hit_any(docs_[i], t.text, t.exclude ? nullptr : &kw_why[i]);
            if (t.exclude) { if (w > 0) excluded[i] = true; continue; }
            if (w > 0) { matched++; score += w; }
        }
        all_terms[i] = positives > 0 && matched == positives;
        kw[i] = positives ? score / (3.3 * positives) : 0;
    }
    if (q.mode == SM_KEYWORD) {
        for (size_t i : pool)
            if (all_terms[i] && !excluded[i]) res.hits.push_back({docs_[i].id, kw[i], kw_why[i]});
    } else {
        // ---- meaning: a calibrated match (query vs the tag vocabulary for that photo), not just "the closest ones"
        std::string err, phrase;
        for (auto& t : terms)
            if (!t.exclude) phrase += (phrase.empty() ? "" : " ") + t.text;
        // CLIP reads English: a query in another language is matched through its English equivalent.
        if (!ascii_only(phrase)) {
            std::string en;
            for (auto& a : alternates(cfg, util::lower(phrase), false))
                if (ascii_only(a) && a != util::lower(phrase)) { en = a; break; }
            if (en.empty()) {
                for (auto& t : terms) {
                    if (t.exclude) continue;
                    std::string pick = ascii_only(t.text) ? t.text : "";
                    for (auto& a : alternates(cfg, t.text, false))
                        if (pick.empty() && ascii_only(a)) pick = a;
                    if (!pick.empty()) en += (en.empty() ? "" : " ") + pick;
                }
            }
            if (!en.empty()) phrase = en;
        }
        Query qv;
        bool sem = cfg.clip_enabled && !phrase.empty() && prepare(cfg, phrase, qv, err);
        if (!sem) {
            res.note = "meaning search unavailable" + (err.empty() ? std::string() : ": " + err) + (q.mode == SM_SMART ? "; words only" : "");
            if (q.mode == SM_SEMANTIC) return res;
        }
        for (size_t i : pool) {
            if (excluded[i]) continue;
            double p = sem ? match(qv, i) : 0;
            bool sem_ok = p >= cfg.search_min_match;
            double sem_score = p;
            if (q.mode == SM_SEMANTIC) {
                if (sem_ok) res.hits.push_back({docs_[i].id, sem_score, util::fmt("shows \"%s\" (%.0f%%)", phrase.c_str(), p * 100)});
                continue;
            }
            bool kw_ok = all_terms[i];
            if (!kw_ok && !sem_ok) continue;
            double score = 0.6 * sem_score + 0.4 * (kw_ok ? std::max(kw[i], 0.5) : 0.0) + (kw_ok && sem_ok ? 0.2 : 0.0);
            std::string why = kw_ok ? kw_why[i] : "";
            if (sem_ok) why += (why.empty() ? "" : " · ") + util::fmt("shows \"%s\" (%.0f%%)", phrase.c_str(), p * 100);
            res.hits.push_back({docs_[i].id, score, why});
        }
    }
    std::stable_sort(res.hits.begin(), res.hits.end(), [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    if (int(res.hits.size()) > q.limit) res.hits.resize(size_t(q.limit));
    return res;
}

// ---------------------------------------------------------------------------
// Query-language search: every leaf is a word (or its equivalents) in the chosen fields, or (Smart mode, no field)
// something the picture shows.

namespace {
// "2mb", "500kb", "12mp", "1024" -> number (bytes / pixels / plain)
double parse_amount(std::string v) {
    v = util::lower(v);
    double mul = 1;
    for (auto [suf, m] : std::initializer_list<std::pair<const char*, double>>{{"gb", 1e9}, {"mb", 1e6}, {"kb", 1e3}, {"mp", 1e6}, {"k", 1e3}, {"m", 1e6}, {"b", 1}})
        if (v.size() > strlen(suf) && v.compare(v.size() - strlen(suf), strlen(suf), suf) == 0) { mul = m; v = v.substr(0, v.size() - strlen(suf)); break; }
    return atof(v.c_str()) * mul;
}
// ">2mb", "<=1000", "2..5mb", plain = equal within 10%
bool compare_amount(double x, const std::string& spec) {
    if (spec.find("..") != std::string::npos) {
        std::string a = spec.substr(0, spec.find("..")), b = spec.substr(spec.find("..") + 2);
        std::string unit;
        for (size_t i = b.size(); i > 0 && isalpha((unsigned char)b[i - 1]); i--) unit = b[i - 1] + unit;
        bool a_unit = !a.empty() && isalpha((unsigned char)a.back());
        return (a.empty() || x >= parse_amount(a_unit ? a : a + unit)) && (b.empty() || x <= parse_amount(b));
    }
    if (util::starts_with(spec, ">=")) return x >= parse_amount(spec.substr(2));
    if (util::starts_with(spec, "<=")) return x <= parse_amount(spec.substr(2));
    if (util::starts_with(spec, ">")) return x > parse_amount(spec.substr(1));
    if (util::starts_with(spec, "<")) return x < parse_amount(spec.substr(1));
    double v = parse_amount(spec);
    return std::fabs(x - v) <= 0.1 * v;
}
// date:2019, date:2019-05, date:2019-05-01..2019-08-31, date:>2020, year:2018..2020
bool match_date(const SearchDoc& d, const std::string& spec) {
    if (d.date.size() < 4) return spec == "none" || spec == "undated";
    auto lo_of = [](std::string v) { return v.size() == 4 ? v + "-01-01" : v.size() == 7 ? v + "-01" : v; };
    auto hi_of = [](std::string v) { return v.size() == 4 ? v + "-12-31" : v.size() == 7 ? v + "-31" : v; };
    std::string from, to;
    if (spec.find("..") != std::string::npos) { from = spec.substr(0, spec.find("..")); to = spec.substr(spec.find("..") + 2); }
    else if (util::starts_with(spec, ">=")) from = spec.substr(2);
    else if (util::starts_with(spec, "<=")) to = spec.substr(2);
    else if (util::starts_with(spec, ">")) { from = spec.substr(1); if (from.size() == 4) from = std::to_string(atoi(from.c_str()) + 1); }
    else if (util::starts_with(spec, "<")) { to = spec.substr(1); if (to.size() == 4) to = std::to_string(atoi(to.c_str()) - 1); }
    else { from = spec; to = spec; }
    return date_in_range(d.date, d.prec, from.empty() ? "" : lo_of(from), to.empty() ? "" : hi_of(to));
}
// Edit distance with an early exit (bytes; fine for typos).
int edit_distance(const std::string& a, const std::string& b, int limit) {
    if (std::abs(int(a.size()) - int(b.size())) > limit) return limit + 1;
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); j++) prev[j] = int(j);
    for (size_t i = 1; i <= a.size(); i++) {
        cur[0] = int(i);
        int best = cur[0];
        for (size_t j = 1; j <= b.size(); j++) {
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
            best = std::min(best, cur[j]);
        }
        if (best > limit) return limit + 1;
        std::swap(prev, cur);
    }
    return prev[b.size()];
}
// The word of `text` closest to `w` within the allowed typos, or "".
std::string fuzzy_word(const std::string& text, const std::string& w) {
    int limit = w.size() <= 4 ? 1 : 2;
    std::string word, best;
    int best_d = limit + 1;
    auto flush = [&] {
        if (word.size() + 2 >= w.size()) {
            int dd = edit_distance(word, w, limit);
            if (dd < best_d) { best_d = dd; best = word; }
        }
        word.clear();
    };
    for (unsigned char ch : text) {
        if (isalnum(ch) || ch >= 0x80) word += char(ch);
        else flush();
    }
    flush();
    return best_d <= limit ? best : "";
}
}  // namespace

SearchResult Searcher::run_logic(const Config& cfg, const SearchQuery& q, const std::vector<size_t>& pool, const QNode& root) {
    SearchResult res;
    std::map<std::string, Query> clip_q;
    bool sem = q.mode == SM_SMART && cfg.clip_enabled;
    std::string cerr;
    std::function<void(const QNode&)> prep = [&](const QNode& n) {
        if (n.kind != QNode::TERM) { for (auto& k : n.kids) prep(k); return; }
        if (!sem || !n.field.empty() || n.term.empty() || clip_q.count(n.term)) return;
        std::string en = n.term;
        if (!ascii_only(en))
            for (auto& a : alternates(cfg, n.term, false))
                if (ascii_only(a)) { en = a; break; }
        Query qq;
        if (ascii_only(en) && prepare(cfg, en, qq, cerr)) clip_q[n.term] = qq;
    };
    prep(root);
    // like:<id> (or like:<part of a file name>): the embedding of that photo
    std::map<std::string, Vec> like_target;
    std::function<void(const QNode&)> likes = [&](const QNode& n) {
        if (n.kind != QNode::TERM) { for (auto& k : n.kids) likes(k); return; }
        if (n.field != "like" || n.term.empty()) return;
        std::string t = n.term[0] == '#' ? n.term.substr(1) : n.term;
        for (auto& d : docs_)
            if (!d.vec.empty() && (std::to_string(d.id) == t || d.name.find(t) != std::string::npos)) { like_target[n.term] = d.vec; break; }
    };
    likes(root);
    // value > 0 = matches (the score adds up); NOT inverts.
    std::function<double(const QNode&, size_t, std::string*)> eval = [&](const QNode& n, size_t i, std::string* why) -> double {
        const SearchDoc& d = docs_[i];
        switch (n.kind) {
            case QNode::AND: {
                double s = 0;
                for (auto& k : n.kids) {
                    double v = eval(k, i, why);
                    if (v <= 0) return 0;
                    s += v;
                }
                return s;
            }
            case QNode::OR: {
                double best = 0;
                for (auto& k : n.kids) best = std::max(best, eval(k, i, best > 0 ? nullptr : why));
                return best;
            }
            case QNode::NOT: return n.kids.empty() || eval(n.kids[0], i, nullptr) > 0 ? 0 : 0.01;
            case QNode::TERM: break;
        }
        if (n.term.empty()) return 0.01;
        auto note = [&](const std::string& w) {
            if (why && why->size() < 90) *why += (why->empty() ? "" : " · ") + w;
        };
        if (n.field == "is") {
            bool on = (n.term == "fav" || n.term == "favorite" || n.term == "favourite" || n.term == "star") ? d.favorite
                    : n.term == "ai" || n.term == "generated" ? !d.prompt_raw.empty()
                    : n.term == "untagged" ? d.tags.empty()
                    : n.term == "uncertain" || n.term == "review" ? d.uncertain
                    : n.term == "undated" ? d.date.empty()
                    : n.term == "portrait" ? d.height > d.width && d.width > 0
                    : n.term == "landscape" ? d.width > d.height && d.height > 0
                    : n.term == "square" ? d.width == d.height && d.width > 0 : false;
            return on ? 0.5 : 0;
        }
        if (n.field == "has") {
            bool on = n.term == "gps" ? d.has_gps
                    : n.term == "prompt" ? !d.prompt_raw.empty()
                    : n.term == "place" || n.term == "location" ? !d.location.empty() || d.has_gps
                    : n.term == "tags" ? !d.tags.empty()
                    : n.term == "desc" || n.term == "description" ? d.exif.find("xmp.dc.description") != std::string::npos
                    : n.term == "keywords" ? d.exif.find("xmp.dc.subject") != std::string::npos || d.exif.find("keywords") != std::string::npos
                    : n.term == "camera" ? d.exif.find("exif.image.model") != std::string::npos
                    : n.term == "date" ? !d.date.empty() : false;
            return on ? 0.5 : 0;
        }
        if (n.field == "date" || n.field == "year") {
            if (!match_date(d, n.term)) return 0;
            note("date " + d.date.substr(0, 10));
            return 0.5;
        }
        if (n.field == "size") {
            if (!compare_amount(double(d.size), n.term)) return 0;
            note(util::human_size(d.size));
            return 0.5;
        }
        if (n.field == "width" || n.field == "w" || n.field == "height" || n.field == "h") {
            int v = n.field[0] == 'w' ? d.width : d.height;
            if (!v || !compare_amount(v, n.term)) return 0;
            note(util::fmt("%d x %d", d.width, d.height));
            return 0.5;
        }
        if (n.field == "mp")
            return d.width && compare_amount(double(d.width) * d.height, n.term.find_first_of("mk") == std::string::npos ? n.term + "mp" : n.term) ? 0.5 : 0;
        if (n.field == "ext" || n.field == "type") {
            std::string e = n.term == "jpeg" ? "jpg" : n.term;
            bool raw = e == "raw" && std::string(" dng cr2 cr3 nef arw raf orf rw2 ").find(" " + d.ext + " ") != std::string::npos;
            return d.ext == e || (e == "jpg" && d.ext == "jpeg") || raw ? 0.5 : 0;
        }
        if (n.field == "camera") {  // Make / Model / lens lines of the EXIF
            for (auto& line : util::split(d.exif, '\n'))
                if ((util::starts_with(line, "exif.image.make") || util::starts_with(line, "exif.image.model") || line.find("lensmodel") != std::string::npos) &&
                    line.find(n.term) != std::string::npos) {
                    note("camera: " + line.substr(std::min(line.size(), line.find('=') + 2)));
                    return 1.0;
                }
            return 0;
        }
        if (n.field == "model" || n.field == "lora") {  // the generator's checkpoint / LoRA names (last line of the prompt field)
            size_t nl = d.prompt.rfind('\n');
            std::string tail = nl == std::string::npos ? "" : d.prompt.substr(nl + 1);
            if (tail.find(n.term) == std::string::npos) return 0;
            note(n.field + ": " + n.term);
            return 1.0;
        }
        if (n.field == "like") {  // looks like photo #id: CLIP image-to-image similarity
            auto it = like_target.find(n.term);
            if (it == like_target.end() || d.vec.empty()) return 0;
            double c = dot(it->second, d.vec);
            if (c < 0.80) return 0;
            note(util::fmt("looks like it (%.0f%%)", c * 100));
            return 3.0 * c;
        }
        if (n.field == "fuzzy") {
            for (auto* field : {&d.desc, &d.name, &d.prompt}) {
                std::string w = fuzzy_word(*field, n.term);
                if (!w.empty()) {
                    note("~" + n.term + ": " + w);
                    return 1.0;
                }
            }
            return 0;
        }
        if (n.field == "place" || n.field == "loc") {
            for (auto& a : alternates(cfg, n.term, false))
                if (d.location.find(a) != std::string::npos) {
                    note("place: " + d.location);
                    return 1.0;
                }
            return 0;
        }
        auto alts = alternates(cfg, n.term, false);
        if (n.field == "tag") {
            for (auto& a : alts)
                if (std::find(d.tags.begin(), d.tags.end(), a) != d.tags.end()) {
                    if (why && why->size() < 90) *why += (why->empty() ? "" : " · ") + std::string("tag: ") + a;
                    return 1.5;
                }
            return 0;
        }
        SearchQuery f = q;
        if (!n.field.empty()) {
            f.in_name = n.field == "name";
            f.in_desc = n.field == "desc";
            f.in_prompt = n.field == "prompt";
            f.in_exif = n.field == "exif";
        }
        double best = 0;
        for (auto& a : alts) best = std::max(best, term_hit(d, f, a, why));
        if (best > 0) return best;
        auto it = clip_q.find(n.term);
        if (it != clip_q.end()) {
            double p = match(it->second, i);
            if (p >= cfg.search_min_match) {
                if (why && why->size() < 90) *why += (why->empty() ? "" : " · ") + util::fmt("shows \"%s\" (%.0f%%)", it->second.text.c_str(), p * 100);
                return p * 1.5;
            }
        }
        return 0;
    };
    for (size_t i : pool) {
        std::string why;
        double v = eval(root, i, &why);
        if (v > 0) res.hits.push_back({docs_[i].id, v, why.empty() ? "matches the filter" : why});
    }
    std::stable_sort(res.hits.begin(), res.hits.end(), [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    if (int(res.hits.size()) > q.limit) res.hits.resize(size_t(q.limit));
    if (sem && clip_q.empty() && !cerr.empty()) res.note = "meaning search unavailable: " + cerr;
    return res;
}

// ---------------------------------------------------------------------------
// Ask: natural language -> keywords/conditions (LLM) -> local candidates >= 50% -> decisions in batches -> ranking

SearchResult Searcher::run_ask(const Config& cfg, const SearchQuery& q, const std::vector<size_t>& pool_in) {
    SearchResult res;
    std::string text = util::trim(q.text), err;
    // 1. understand the request
    std::vector<std::string> keywords, excludes;
    std::string visual = text, place, from = q.date_from, to = q.date_to;
    bool understood = false;
    if (cfg.llm_enabled) {
        std::string today = util::now_iso().substr(0, 10);
        std::string prompt =
            "Today is " + today + ". A user searches their personal photo library with: \"" + text + "\".\n"
            "Extract search conditions. Reply JSON only:\n"
            "{\"keywords\": [\"single words or short phrases likely to appear in photo tags, captions, places or file names; "
            "at most 12, include synonyms\"], \"visual\": \"short description of what the picture shows, in English\", "
            "\"place\": \"place name or empty\", \"date_from\": \"YYYY-MM-DD or empty\", \"date_to\": \"YYYY-MM-DD or empty\", "
            "\"exclude\": [\"things the user does not want\"]}";
        std::string reply = llm_ask(cfg, prompt, 600, true, err);
        if (err.empty() && extract_json_object(reply).empty()) err = "could not read the LLM's reply";
        json j = json::parse(extract_json_object(reply), nullptr, false);
        if (j.is_object()) {
            understood = true;
            auto list = [&](const char* k, std::vector<std::string>& out) {
                if (j.contains(k) && j[k].is_array())
                    for (auto& x : j[k])
                        if (x.is_string() && !util::trim(x.get<std::string>()).empty()) out.push_back(util::lower(util::trim(x.get<std::string>())));
            };
            list("keywords", keywords);
            list("exclude", excludes);
            if (j.contains("visual") && j["visual"].is_string() && !j["visual"].get<std::string>().empty()) visual = j["visual"].get<std::string>();
            if (j.contains("place") && j["place"].is_string()) place = util::lower(util::trim(j["place"].get<std::string>()));
            auto date = [&](const char* k, std::string& out) {
                std::string v = j.contains(k) && j[k].is_string() ? util::trim(j[k].get<std::string>()) : "";
                util::Civil c = util::parse_exif_datetime(v);
                if (c.valid() && c.prec >= util::P_DAY && out.empty()) out = c.pretty().substr(0, 10);
            };
            date("date_from", from);
            date("date_to", to);
        }
    }
    if (!understood) {  // no LLM: the words themselves, minus filler words
        static const std::set<std::string> filler = {"a", "an", "the", "of", "in", "on", "at", "to", "my", "our", "and", "or", "with", "from",
                                                     "for", "photo", "photos", "picture", "pictures", "image", "images", "some", "me", "show"};
        for (auto& t : parse_terms(text))
            if (!filler.count(t.text)) (t.exclude ? excludes : keywords).push_back(t.text);
        res.note = "LLM unavailable" + (err.empty() ? std::string() : " (" + err + ")") + "; searching the words directly. ";
    }
    if (!place.empty() && std::find(keywords.begin(), keywords.end(), place) == keywords.end()) keywords.push_back(place);

    // 2. local candidates: keywords in any field, or a real visual match; keep the ones scoring >= 50%
    Query qv;
    std::string cerr;
    bool sem = cfg.clip_enabled && prepare(cfg, visual, qv, cerr);
    SearchQuery all_fields = q;
    all_fields.in_name = all_fields.in_exif = all_fields.in_desc = all_fields.in_prompt = true;
    struct Cand {
        size_t i;
        double local;
        std::string why;
    };
    std::vector<Cand> cands;
    for (size_t i : pool_in) {
        const SearchDoc& d = docs_[i];
        if (!date_in_range(d.date, d.prec, from, to)) continue;
        bool excluded = false;
        for (auto& x : excludes) excluded |= term_hit(d, all_fields, x, nullptr) > 0;
        if (excluded) continue;
        int hit = 0;
        std::string why;
        for (auto& k : keywords) hit += term_hit(d, all_fields, k, &why) > 0;
        double kwf = keywords.empty() ? 0 : std::min(1.0, 2.0 * hit / double(keywords.size()));  // half the keywords = full marks
        double p = sem ? match(qv, i) : 0;
        double semf = std::clamp(p / 0.6, 0.0, 1.0);
        double local = std::min(1.0, std::max(kwf, semf) + (kwf > 0 && semf > 0.3 ? 0.15 : 0.0));
        if (local < 0.5) continue;
        if (p >= cfg.search_min_match) why += (why.empty() ? "" : " · ") + util::fmt("shows it (%.0f%%)", p * 100);
        cands.push_back({i, local, why});
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.local > b.local; });
    size_t found = cands.size();
    if (int(cands.size()) > cfg.ask_max_candidates) cands.resize(size_t(cfg.ask_max_candidates));

    // 3. decisions, in batches. The LLM rates; jev (Julia) judges one self-contained statement pair per photo
    //    ("... is / is not what someone searching X wants"), which is the form it answers well.
    std::vector<double> llm(cands.size(), -1), jev(cands.size(), -1);
    bool use_llm = cfg.ask_decider == JUDGE_LLM || cfg.ask_decider == JUDGE_LLM_JEV;
    bool use_jev = cfg.ask_decider == JUDGE_JEV || cfg.ask_decider == JUDGE_LLM_JEV;
    std::string llm_err, jev_err;
    for (size_t at = 0; at < cands.size() && use_llm; at += size_t(cfg.ask_batch)) {
        size_t n = std::min(size_t(cfg.ask_batch), cands.size() - at);
        std::string lines;
        for (size_t k = 0; k < n; k++) lines += std::to_string(k) + ": " + describe(docs_[cands[at + k].i]) + "\n";
        std::string prompt = "Today is " + util::now_iso().substr(0, 10) + ". A user searches their photo library for: \"" + text +
                             "\".\nCandidate photos (id: tags, scene, place, caption | date | file):\n" + lines +
                             "Rate how likely each photo is what the user is looking for, 0 to 100. "
                             "Reply JSON only: {\"scores\": {\"<id>\": <0-100>, ...}}";
        std::string reply = llm_ask(cfg, prompt, 40 + 12 * int(n), true, llm_err);
        json j = json::parse(extract_json_object(reply), nullptr, false);
        if (!j.is_object() || !j.contains("scores")) {
            if (llm_err.empty()) llm_err = "unreadable reply";
            use_llm = false;
            break;
        }
        for (auto& [k, v] : j["scores"].items()) {
            size_t idx = size_t(atoi(k.c_str()));
            if (idx < n && v.is_number()) llm[at + idx] = std::clamp(v.get<double>(), 0.0, 100.0);
        }
    }
    // Which photos jev judges: all of them (jev alone), or the LLM's close calls (LLM + jev).
    std::vector<size_t> ask_jev;
    for (size_t k = 0; k < cands.size(); k++)
        if (cfg.ask_decider == JUDGE_JEV || (use_llm && llm[k] >= 30 && llm[k] <= 70) || (!use_llm && cfg.ask_decider == JUDGE_LLM_JEV))
            ask_jev.push_back(k);
    std::string quoted = util::replace_all(text, "\"", "'");
    auto yes_text = [&](const SearchDoc& d) { return "A photo with " + describe(d) + " is what someone searching for \"" + quoted + "\" wants."; };
    auto no_text = [&](const SearchDoc& d) { return "A photo with " + describe(d) + " is not what someone searching for \"" + quoted + "\" wants."; };
    for (size_t at = 0; at < ask_jev.size() && use_jev; at += size_t(cfg.ask_batch)) {
        size_t n = std::min(size_t(cfg.ask_batch), ask_jev.size() - at);
        json qs = json::object();
        for (size_t k = 0; k < n; k++) {
            const SearchDoc& d = docs_[cands[ask_jev[at + k]].i];
            qs["c" + std::to_string(k)] = {{"type", "choice"},
                                           {"instructions", "Which statement is true?"},
                                           {"criteria", {{"yes", yes_text(d)}, {"no", no_text(d)}}}};
        }
        json req = {{"model", cfg.jev_model}, {"state", json::object()}, {"questions", qs}};
        HttpResponse h = http_post_json(url_join(cfg.jev_url, "/v1/systemone"), req.dump(-1, ' ', false, json::error_handler_t::replace),
                                        cfg.jev_key, cfg.jev_timeout);
        json j = json::parse(h.body, nullptr, false);
        if (!h.ok() || !j.is_object() || !j.contains("answers")) {
            jev_err = h.error.empty() ? "unexpected reply" : h.error;
            use_jev = false;
            break;
        }
        for (size_t k = 0; k < n; k++) {
            const json& a = j["answers"]["c" + std::to_string(k)];
            if (a.is_object() && a.contains("probabilities") && a["probabilities"].contains("yes") && a["probabilities"]["yes"].is_number())
                jev[ask_jev[at + k]] = 100.0 * a["probabilities"]["yes"].get<double>();
        }
    }
    int kept = 0, jev_asked = 0, jev_changed = 0;
    for (size_t k = 0; k < cands.size(); k++) {
        const Cand& c = cands[k];
        const SearchDoc& d = docs_[c.i];
        double decision = -1;
        std::string why;
        if (llm[k] >= 0 && jev[k] >= 0) {
            decision = 0.5 * llm[k] + 0.5 * jev[k];
            why = util::fmt("LLM %.0f%% · jev %.0f%%", llm[k], jev[k]);
        } else if (llm[k] >= 0) {
            decision = llm[k];
            why = util::fmt("LLM %.0f%%", llm[k]);
        } else if (jev[k] >= 0) {
            decision = jev[k];
            why = util::fmt("jev %.0f%%", jev[k]);
        }
        if (jev[k] >= 0) {
            jev_asked++;
            bool before = llm[k] < 0 ? true : llm[k] >= cfg.ask_keep, after = decision >= cfg.ask_keep;
            jev_changed += before != after;
            Decision rec;
            rec.kind = "search";
            rec.photo_id = d.id;
            rec.subject = text;
            rec.question = util::basename(d.name_raw.substr(0, d.name_raw.find('\n')));
            rec.options = json::array({json::array({yes_text(d), std::round(jev[k] * 10) / 1000}),
                                       json::array({no_text(d), std::round((100 - jev[k]) * 10) / 1000})})
                              .dump();
            rec.rules_pick = llm[k] < 0 ? "-" : util::fmt("LLM %.0f%%: %s", llm[k], before ? "keep" : "drop");
            rec.jev_pick = util::fmt("%s (%.0f%%)", jev[k] >= 50 ? "yes" : "no", jev[k] >= 50 ? jev[k] : 100 - jev[k]);
            rec.final_pick = after ? "keep" : "drop";
            rec.changed = llm[k] >= 0 && before != after;
            res.decisions.push_back(rec);
        }
        double score = c.local;
        if (decision >= 0) {
            if (decision < cfg.ask_keep) continue;
            score = 0.3 * c.local + 0.7 * decision / 100.0;
        }
        if (!c.why.empty()) why += (why.empty() ? "" : " · ") + c.why;
        res.hits.push_back({d.id, score, why});
        kept++;
    }
    std::stable_sort(res.hits.begin(), res.hits.end(), [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
    std::string conds;
    for (auto& k : keywords) conds += (conds.empty() ? "" : ", ") + k;
    res.note += util::fmt("looked for: %s", conds.empty() ? "-" : conds.c_str());
    if (visual != text) res.note += " · picture: \"" + visual + "\"";
    if (!from.empty() || !to.empty()) res.note += " · dates " + (from.empty() ? std::string("…") : from) + " to " + (to.empty() ? std::string("…") : to);
    res.note += util::fmt(" · %zu candidates", found);
    if (use_llm) res.note += " · LLM rated " + std::to_string(cands.size());
    if (jev_asked) res.note += util::fmt(" · jev judged %d%s", jev_asked, cfg.ask_decider == JUDGE_LLM_JEV ? " close calls" : "") +
                               (jev_changed ? util::fmt(", changed %d", jev_changed) : "");
    res.note += util::fmt(" · kept %d", kept);
    if (!llm_err.empty() && cfg.ask_decider != JUDGE_JEV && cfg.ask_decider != JUDGE_NONE) res.note += " · LLM unavailable (" + llm_err + ")";
    if (!jev_err.empty()) res.note += " · jev unavailable (" + jev_err + ")";
    if (cfg.ask_decider == JUDGE_NONE || (!use_llm && !use_jev)) res.note += " · local ranking only";
    return res;
}
