/* The GPT-4-style split regex, hand-compiled (profile 1):
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)          A1 contraction
 *   [^\r\n\p{L}\p{N}]?\p{L}+              A2 word (optional one-char prefix)
 *   \p{N}                                 A3 single digit
 *    ?[^\s\p{L}\p{N}]+[\r\n]*             A4 punctuation run
 *   \s*[\r\n]+                            A5 whitespace through last newline
 *   \s+(?!\S)                             A6 trailing whitespace (give-back)
 *   \s+                                   A7 whitespace fallback
 *
 * Ordered alternation on a backtracking engine; every character matches one
 * alternative, so the spans partition the input and each pretoken feeds the
 * BPE engine directly. docs/FORMAT.md derives the consequences implemented
 * here; the differential suite is the proof.
 *
 * Each position is decoded once. ASCII runs are extended by the AVX2
 * kernels when simd is set; the scalar paths are the reference the kernels
 * must equal byte for byte. Malformed UTF-8 decodes as class 0 ("other"),
 * one byte at a time, which lands in A4 like any symbol.
 */
#include "bp_internal.h"
#include "tables/bp_uctables.h"

typedef struct {
    uint32_t cp;
    uint32_t adv;
    uint8_t  cl;
} bp_cp;

static inline bp_cp peek(const uint8_t *t, size_t i, size_t len)
{
    bp_cp r;
    uint8_t b = t[i];
    if (b < 0x80) {
        r.cp = b;
        r.adv = 1;
        r.cl = bp_char_class(b);
        return r;
    }
    size_t adv;
    r.cp = bp_utf8_next(t + i, t + len, &adv);
    r.adv = (uint32_t)adv;
    r.cl = (r.cp == 0xFFFFFFFFu) ? 0 : bp_char_class(r.cp);
    return r;
}

/* A1: at an ASCII apostrophe, match 's 't 're 've 'm 'll 'd, case-insensitive
 * under Unicode simple case folding, which admits exactly one non-ASCII
 * character: U+017F LATIN SMALL LETTER LONG S folds to 's'. That set was
 * established by sweeping every letter codepoint through every contraction
 * position against the reference implementation (2026-08-05); all other
 * positions are strictly [a-zA-Z]. Returns match length in bytes. */
static size_t try_contraction(const uint8_t *t, size_t i, size_t len)
{
    size_t n = len - i;
    if (n < 2) return 0;
    uint8_t a = t[i + 1] | 0x20;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (t[i + 1] == 0xC5 && n >= 3 && t[i + 2] == 0xBF) return 3; /* U+017F */
    if (n < 3) return 0;
    uint8_t b = t[i + 2] | 0x20;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
        (a == 'l' && b == 'l'))
        return 3;
    return 0;
}

/* Extend a letter run starting at j. */
static size_t run_letters(const uint8_t *t, size_t j, size_t len, int simd)
{
    while (j < len) {
        uint8_t b = t[j];
        if (b < 0x80) {
            if (simd) {
                j = bp_ascii_letter_run_avx2(t, j, len);
                if (j >= len || t[j] < 0x80) return j;
                continue; /* non-ASCII byte: decode below */
            }
            if (!(bp_char_class(b) & BP_CC_L)) return j;
            j++;
            continue;
        }
        size_t adv;
        uint32_t cp = bp_utf8_next(t + j, t + len, &adv);
        if (cp == 0xFFFFFFFFu || !(bp_char_class(cp) & BP_CC_L)) return j;
        j += adv;
    }
    return j;
}

/* Extend a symbol run (not \s, not letter, not number) starting at j. */
static size_t run_symbols(const uint8_t *t, size_t j, size_t len, int simd)
{
    while (j < len) {
        uint8_t b = t[j];
        if (b < 0x80) {
            if (simd) {
                j = bp_ascii_symbol_run_avx2(t, j, len);
                if (j >= len || t[j] < 0x80) return j;
                continue;
            }
            if (bp_char_class(b) & (BP_CC_L | BP_CC_N | BP_CC_S)) return j;
            j++;
            continue;
        }
        size_t adv;
        uint32_t cp = bp_utf8_next(t + j, t + len, &adv);
        if (cp != 0xFFFFFFFFu &&
            (bp_char_class(cp) & (BP_CC_L | BP_CC_N | BP_CC_S)))
            return j;
        j += adv;
    }
    return j;
}

#define EMIT(a, b)                                                \
    do {                                                          \
        int rc_ = bp_bpe_pretoken(c, t + (a), (b) - (a), sink);   \
        if (rc_) return rc_;                                      \
    } while (0)

int bp_scan_gpt4(bp_ctx *c, const uint8_t *t, size_t len,
                 bp_sink *sink, int simd)
{
    size_t i = 0;
    while (i < len) {
        bp_cp cur = peek(t, i, len);

        /* A1 */
        if (t[i] == '\'') {
            size_t m = try_contraction(t, i, len);
            if (m) {
                EMIT(i, i + m);
                i += m;
                continue;
            }
        }

        /* A2: letter run */
        if (cur.cl & BP_CC_L) {
            size_t j = run_letters(t, i + cur.adv, len, simd);
            EMIT(i, j);
            i = j;
            continue;
        }

        /* A2 with one-char prefix: cur not CR/LF/L/N, next is a letter */
        bp_cp nx = { 0, 0, 0 };
        int have_nx = 0;
        if (!(cur.cl & (BP_CC_L | BP_CC_N | BP_CC_NL)) &&
            i + cur.adv < len) {
            nx = peek(t, i + cur.adv, len);
            have_nx = 1;
            if (nx.cl & BP_CC_L) {
                size_t j = run_letters(t, i + cur.adv + nx.adv, len, simd);
                EMIT(i, j);
                i = j;
                continue;
            }
        }

        /* A3: single number */
        if (cur.cl & BP_CC_N) {
            EMIT(i, i + cur.adv);
            i += cur.adv;
            continue;
        }

        /* A4: optional single space, >=1 symbol, then a CR/LF run */
        if (t[i] == ' ' && have_nx &&
            !(nx.cl & (BP_CC_L | BP_CC_N | BP_CC_S))) {
            size_t j = run_symbols(t, i + 1 + nx.adv, len, simd);
            while (j < len && (t[j] == '\r' || t[j] == '\n')) j++;
            EMIT(i, j);
            i = j;
            continue;
        }
        if (!(cur.cl & (BP_CC_L | BP_CC_N | BP_CC_S))) {
            size_t j = run_symbols(t, i + cur.adv, len, simd);
            while (j < len && (t[j] == '\r' || t[j] == '\n')) j++;
            EMIT(i, j);
            i = j;
            continue;
        }

        /* A5/A6/A7: cur is whitespace. Walk the maximal \s run, tracking
         * the end of the last CR/LF and the start of the last codepoint. */
        {
            size_t j = i, last_nl_end = 0, last_cp_start = i, ncp = 0;
            while (j < len) {
                bp_cp w = (j == i) ? cur : peek(t, j, len);
                if (!(w.cl & BP_CC_S)) break;
                last_cp_start = j;
                ncp++;
                j += w.adv;
                if (w.cl & BP_CC_NL) last_nl_end = j;
            }
            size_t end;
            if (last_nl_end)          end = last_nl_end;      /* A5 */
            else if (j == len)        end = len;              /* A6 at EOS */
            else if (ncp >= 2)        end = last_cp_start;    /* A6 give-back */
            else                      end = j;                /* A7 */
            EMIT(i, end);
            i = end;
        }
    }
    return 0;
}
