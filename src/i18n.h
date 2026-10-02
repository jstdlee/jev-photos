// UI translations keyed by the English text (English, 简体中文, 日本語, 한국어).
#pragma once

enum Lang { L_EN, L_ZH_CN, L_JA, L_KO, L_COUNT };
extern const char* const kLangCodes[L_COUNT];
extern const char* const kLangNames[L_COUNT];

const char* tr(const char* en);
void set_lang(Lang l);
Lang get_lang();
Lang lang_from_code(const char* code);
