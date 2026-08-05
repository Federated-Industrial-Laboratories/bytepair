/* Vocabulary reachability census.
 *
 * For every token id, decide with evidence whether any valid UTF-8 input
 * can make the encoder emit it. Verdicts:
 *
 *   added        matched literally by the added-token pass (witness: itself)
 *   self         encode(bytes(t)) contains t (witness: the bytes alone)
 *   context      a bounded affix search found w with t in encode(w)
 *   impossible   a sound proof exists that no input can produce t
 *   unresolved   no witness within the search bound, no proof found
 *
 * The exactness backbone is the boundary lemma (docs/CENSUS.md): within one
 * pretoken, surrounding bytes either cross the token's byte range in some
 * merge (after which the range can never reassemble into t, since merged
 * bytes never separate) or they never cross it, in which case the range's
 * internal merges fire in exactly the isolated relative order (ranks are
 * global and unaffected by neighbors). Hence if pure BPE over bytes(t)
 * does not yield [t], no context can produce t: merge-order failure is a
 * PROOF of unreachability, not merely a failed test. The remaining
 * questions are pretokenizer questions, and those are decided by the
 * profile-1 pretoken language plus a bounded search whose witnesses are
 * re-verified through the full public encode path.
 *
 * Impossibility reasons:
 *   merge-order      pure BPE over bytes(t) yields a different partition
 *   invalid-utf8     bytes(t) cannot occur inside any valid UTF-8 string
 *   pretoken-shape   no pretoken of the profile-1 language can contain
 *                    bytes(t) (letter/digit mixes, interior spaces, ...)
 *   contraction      bytes(t) would need an apostrophe at pretoken start
 *                    followed by a contraction suffix plus more letters,
 *                    which alternative A1 always claims first
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bp_internal.h"
#include "tables/bp_uctables.h"

enum { CV_ADDED, CV_SELF, CV_CONTEXT, CV_IMPOSSIBLE, CV_UNRESOLVED };
enum { CR_NONE, CR_MERGE_ORDER, CR_INVALID_UTF8, CR_PRETOKEN_SHAPE,
       CR_CONTRACTION, CR_NFC_EXCLUDED, CR_COUNT };

static const char *verdict_name[] = { "added", "self", "context",
                                      "impossible", "unresolved" };
static const char *reason_name[] = { "", "merge-order", "invalid-utf8",
                                     "pretoken-shape", "contraction",
                                     "nfc-excluded" };

/* ---- class bits for the pretoken-language prover ---------------------- */
enum { U_L = 1, U_N = 2, U_SP = 4, U_SNL = 8, U_SO = 16, U_O = 32 };

static int cp_unit(uint32_t cp)
{
    if (cp == ' ') return U_SP;
    if (cp == '\r' || cp == '\n') return U_SNL;
    uint8_t c = bp_char_class(cp);
    if (c & BP_CC_L) return U_L;
    if (c & BP_CC_N) return U_N;
    if (c & BP_CC_S) return U_SO;
    return U_O;
}

/* Union of unit bits over every codepoint whose UTF-8 encoding ends with
 * the k bytes at p (a partial leading character) or starts with the bytes
 * at p (a partial trailing character). Exact: candidates are iterated by
 * codepoint arithmetic (fixed low or high bit groups), never capped. A
 * companion flag reports whether EVERY candidate is NFC quick-check No,
 * in which case that partial character can never occur in normalized
 * input at all. */
static int cp_unit_qc(uint32_t cp, int *any_qcyes)
{
    if (bp_nfc_qc(cp) != 0) *any_qcyes = 1;
    return cp_unit(cp);
}

static int partial_prefix_bits(const uint8_t *p, size_t k, int *all_qcno)
{
    int bits = 0, any_qcyes = 0;
    /* the k continuation bytes fix the low 6k bits of the codepoint */
    uint32_t low = 0;
    for (size_t i = 0; i < k; i++)
        low = (low << 6) | (p[i] & 63);
    for (int m = (int)k + 1; m <= 4; m++) {
        uint32_t lo = m == 2 ? 0x80 : m == 3 ? 0x800 : 0x10000;
        uint32_t hi = m == 2 ? 0x7FF : m == 3 ? 0xFFFF : 0x10FFFF;
        uint32_t step = 1u << (6 * k);
        for (uint32_t high = 0;; high++) {
            uint32_t cp = (high << (6 * k)) | low;
            if (cp > hi) break;
            if (cp < lo || (cp >= 0xD800 && cp <= 0xDFFF)) continue;
            (void)step;
            bits |= cp_unit_qc(cp, &any_qcyes);
        }
    }
    *all_qcno = bits != 0 && !any_qcyes;
    return bits;
}

static int utf8_len_from_lead(uint8_t b)
{
    if (b < 0x80) return 1;
    if (b >= 0xC2 && b <= 0xDF) return 2;
    if (b >= 0xE0 && b <= 0xEF) return 3;
    if (b >= 0xF0 && b <= 0xF4) return 4;
    return 0;
}

static int partial_suffix_bits(const uint8_t *p, size_t have, int *all_qcno)
{
    int need = utf8_len_from_lead(p[0]);
    *all_qcno = 0;
    if (need == 0 || (size_t)need <= have) return 0; /* not a valid partial */
    for (size_t i = 1; i < have; i++)
        if ((p[i] & 0xC0) != 0x80) return 0;
    long combos = 1;
    for (int i = 0; i < need - (int)have; i++) combos *= 64;
    int bits = 0, any_qcyes = 0;
    for (long c = 0; c < combos; c++) {
        uint8_t b[4];
        memcpy(b, p, have);
        long v = c;
        for (int i = need - 1; i >= (int)have; i--) { b[i] = 0x80 | (v & 63); v >>= 6; }
        size_t adv;
        uint32_t cp = bp_utf8_next(b, b + need, &adv);
        if (cp != 0xFFFFFFFFu && adv == (size_t)need)
            bits |= cp_unit_qc(cp, &any_qcyes);
    }
    *all_qcno = bits != 0 && !any_qcyes;
    return bits;
}

/* One UTF-8 alignment of the token bytes: unit bit-sets, in order. */
#define MAX_UNITS 260

static int alignment_units(const uint8_t *t, size_t n, size_t lead_k,
                           int *units, int *nu, int *has_qcno)
{
    int m = 0;
    *has_qcno = 0;
    if (lead_k) {
        for (size_t i = 0; i < lead_k; i++)
            if ((t[i] & 0xC0) != 0x80) return 0;
        int aq;
        int bits = partial_prefix_bits(t, lead_k, &aq);
        if (!bits) return 0;
        if (aq) *has_qcno = 1;
        units[m++] = bits;
    }
    size_t i = lead_k;
    while (i < n) {
        int need = utf8_len_from_lead(t[i]);
        if (need == 0) return 0;
        if (i + (size_t)need <= n) {
            size_t adv;
            uint32_t cp = bp_utf8_next(t + i, t + n, &adv);
            if (cp == 0xFFFFFFFFu || adv != (size_t)need) return 0;
            if (m >= MAX_UNITS) return 0;
            /* NFC quick-check No: this codepoint never occurs in NFC
             * output, so an alignment containing it as a complete
             * character cannot occur in post-normalization text */
            if (bp_nfc_qc(cp) == 0) *has_qcno = 1;
            units[m++] = cp_unit(cp);
            i += adv;
        } else {
            int aq;
            int bits = partial_suffix_bits(t + i, n - i, &aq);
            if (!bits) return 0;
            if (m >= MAX_UNITS) return 0;
            if (aq) *has_qcno = 1;
            units[m++] = bits;
            i = n;
        }
    }
    *nu = m;
    return m > 0;
}

/* Can this unit sequence sit inside some pretoken of the relaxed profile-1
 * language?  Shapes (each a superset of what the scanner can emit):
 *   ii   P? L+        P = one unit that could be SP, SO or O
 *   iii  N            a single number unit
 *   iv   SP? O+ NL*
 *   v    S+           (SP, SO or SNL)
 */
static int fits_pretoken_language(const int *u, int nu)
{
    int all_L = 1, all_S = 1;
    for (int i = 0; i < nu; i++) {
        if (!(u[i] & U_L)) all_L = 0;
        if (!(u[i] & (U_SP | U_SO | U_SNL))) all_S = 0;
    }
    if (all_L || all_S) return 1;
    if (nu == 1 && (u[0] & (U_N | U_SP | U_SO | U_O))) return 1;
    /* ii with prefix */
    if (u[0] & (U_SP | U_SO | U_O)) {
        int rest_L = 1;
        for (int i = 1; i < nu; i++)
            if (!(u[i] & U_L)) rest_L = 0;
        if (rest_L && nu >= 2) return 1;
    }
    /* iv */
    {
        int i = 0;
        if (u[0] & U_SP) i = 1;
        int syms = 0;
        while (i < nu && (u[i] & U_O)) { i++; syms++; }
        while (i < nu && (u[i] & U_SNL)) i++;
        if (i == nu && (syms > 0 || nu == 1)) return 1;
    }
    return 0;
}

/* A1 priority: an apostrophe that begins its pretoken, followed by a
 * contraction suffix and then MORE letters, is always claimed by A1 first,
 * so the P?L+ reading never happens. Exact check on the raw bytes. */
static int blocked_by_contraction(const uint8_t *t, size_t n)
{
    if (n < 2 || t[0] != '\'') return 0;
    size_t m = 0;
    uint8_t a = t[1] | 0x20;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') m = 2;
    else if (n >= 3 && t[1] == 0xC5 && t[2] == 0xBF) m = 3;
    else if (n >= 3) {
        uint8_t b = t[2] | 0x20;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l'))
            m = 3;
    }
    if (m == 0 || m >= n) return 0; /* no match, or t IS the contraction */
    /* something follows the contraction: if it is all letters, the only
     * possible reading was P?L+ from the apostrophe, which A1 pre-empts */
    const uint8_t *p = t + m, *end = t + n;
    while (p < end) {
        size_t adv;
        uint32_t cp = bp_utf8_next(p, end, &adv);
        if (cp == 0xFFFFFFFFu) return 0;
        if (p + adv > end) return 0;
        if (!(bp_char_class(cp) & BP_CC_L)) return 0;
        p += adv;
    }
    return 1;
}

static int prove_impossible(const uint8_t *t, size_t n)
{
    if (blocked_by_contraction(t, n)) return CR_CONTRACTION;
    int any_alignment = 0, fits_only_with_qcno = 0;
    for (size_t k = 0; k <= 3 && k < n; k++) {
        int units[MAX_UNITS], nu, qcno;
        if (!alignment_units(t, n, k, units, &nu, &qcno)) continue;
        any_alignment = 1;
        if (fits_pretoken_language(units, nu)) {
            if (!qcno) return CR_NONE;
            fits_only_with_qcno = 1;
        }
    }
    if (fits_only_with_qcno) return CR_NFC_EXCLUDED;
    /* the whole token may sit INSIDE one character (a middle fragment of a
     * 3- or 4-byte encoding): all bytes are continuations, and one unit of
     * every class is the sound loosening - it always fits, so such tokens
     * go to the witness search rather than being called impossible */
    if (n <= 3) {
        int all_cont = 1;
        for (size_t i = 0; i < n; i++)
            if ((t[i] & 0xC0) != 0x80) all_cont = 0;
        if (all_cont) return CR_NONE;
    }
    return any_alignment ? CR_PRETOKEN_SHAPE : CR_INVALID_UTF8;
}

/* ---- encode helpers --------------------------------------------------- */

#define WMAX 320

static int contains_id(const uint32_t *ids, int64_t n, uint32_t t)
{
    for (int64_t i = 0; i < n; i++)
        if (ids[i] == t) return 1;
    return 0;
}

/* A witness must be valid UTF-8: reachability is defined against the
 * reference implementation, which only accepts well-formed strings. (Our
 * own encoder tolerates malformed bytes, so without this check the census
 * would "witness" tokens with inputs the reference can never receive.) */
static int utf8_valid(const uint8_t *p, size_t n)
{
    const uint8_t *end = p + n;
    while (p < end) {
        size_t adv;
        if (bp_utf8_next(p, end, &adv) == 0xFFFFFFFFu) return 0;
        p += adv;
    }
    return 1;
}

static int try_witness(bp_ctx *c, uint32_t t, const char *pre,
                       const uint8_t *tb, size_t tn, const char *suf,
                       uint8_t *w, size_t *wn)
{
    size_t pl = strlen(pre), sl = strlen(suf);
    size_t len = pl + tn + sl;
    if (len > WMAX) return 0;
    memcpy(w, pre, pl);
    memcpy(w + pl, tb, tn);
    memcpy(w + pl + tn, suf, sl);
    if (!utf8_valid(w, len)) return 0;
    uint32_t out[WMAX + 8];
    int64_t n = bp_encode(c, (const char *)w, len, out, WMAX + 8, 0);
    if (n > 0 && n <= WMAX + 8 && contains_id(out, n, t)) {
        *wn = len; /* only a successful, valid witness is ever reported */
        return 1;
    }
    return 0;
}

/* pure BPE over the bytes, no pretokenizer: the boundary-lemma test */
typedef struct { uint32_t only; int count; int match; } pure_probe;

static int pure_is_single(bp_ctx *c, const uint8_t *tb, size_t tn, uint32_t t)
{
    uint32_t out[WMAX + 8];
    bp_sink s = { out, WMAX + 8, 0 };
    if (tn > WMAX || bp_bpe_pretoken(c, tb, tn, &s) != 0) return 0;
    return s.total == 1 && out[0] == t;
}

/* Constructive gluing: a token whose bytes begin inside one character or
 * end inside another is witnessed by BUILDING the completing characters
 * from the alignment, rather than hoping a pool affix happens to fit.
 * Prefix candidates complete the leading k continuation bytes (a lead byte
 * plus padding continuations placed before them); suffix candidates finish
 * a trailing partial character with padding continuations. Every candidate
 * still passes the UTF-8 validity gate and the full-pipeline encode. */
static int try_glue(bp_ctx *c, uint32_t t, const uint8_t *tb, size_t tn,
                    uint8_t *w, size_t *wn)
{
    static const uint8_t pad[] = { 0x80, 0x90, 0xA0, 0xB1, 0x8F, 0x9F };
    static const uint8_t leads[] = { 0xC2, 0xC3, 0xD0, 0xE0, 0xE1, 0xE4,
                                     0xEA, 0xED, 0xF0, 0xF1, 0xF3, 0xF4 };
    for (size_t k = 0; k <= 3 && k <= tn; k++) {
        for (size_t i = 0; i < k; i++)
            if ((tb[i] & 0xC0) != 0x80) goto next_k;
        {
        /* trailing partial: bytes after the leading k form complete chars
         * then possibly a lead byte + some continuations */
        size_t i = k;
        while (i < tn) {
            int need = utf8_len_from_lead(tb[i]);
            if (need == 0) break;
            if (i + (size_t)need > tn) break;
            i += need;
        }
        size_t tail_at = i;
        int tail_need = 0;
        if (i < tn) {
            tail_need = utf8_len_from_lead(tb[i]);
            if (tail_need == 0 && k == tn) { /* pure continuation run */
                tail_at = tn;
            } else if (tail_need == 0) {
                goto next_k;
            } else {
                for (size_t j = i + 1; j < tn; j++)
                    if ((tb[j] & 0xC0) != 0x80) goto next_k;
            }
        }
        size_t tail_have = tn - tail_at;
        size_t tail_pad = tail_need ? (size_t)tail_need - 1 -
                                      (tail_have ? tail_have - 1 : 0) : 0;
        if (tail_need && tail_have == 0) goto next_k; /* cannot happen */
        if (tail_pad > 3) goto next_k;

        for (size_t li = 0; k == 0 ? li < 1 : li < sizeof(leads); li++)
            for (size_t p1 = 0; p1 < (k ? sizeof(pad) : 1); p1++)
                for (size_t p2 = 0; p2 < (tail_pad ? sizeof(pad) : 1); p2++) {
                    uint8_t pre[8], suf[8];
                    size_t pl = 0, sl = 0;
                    if (k) {
                        int m1 = utf8_len_from_lead(leads[li]);
                        if (m1 == 0 || (size_t)m1 < k + 1) continue;
                        pre[pl++] = leads[li];
                        for (int q = 0; q < m1 - 1 - (int)k; q++)
                            pre[pl++] = pad[p1];
                    }
                    for (size_t q = 0; q < tail_pad; q++)
                        suf[sl++] = pad[p2];
                    if (pl + tn + sl > WMAX || pl + tn + sl == tn) continue;
                    char pres[9], sufs[9];
                    memcpy(pres, pre, pl); pres[pl] = 0;
                    memcpy(sufs, suf, sl); sufs[sl] = 0;
                    /* NUL bytes cannot occur in pad/lead sets, so the
                     * string-based try_witness carries them faithfully */
                    if (try_witness(c, t, pres, tb, tn, sufs, w, wn))
                        return 1;
                }
        }
next_k:;
    }
    /* the whole token inside ONE character: place the run at each interior
     * offset of a constructed 3- or 4-byte character */
    if (tn <= 2) {
        int all_cont = 1;
        for (size_t i = 0; i < tn; i++)
            if ((tb[i] & 0xC0) != 0x80) all_cont = 0;
        if (all_cont && tn > 0) {
            static const uint8_t pads2[] = { 0x80, 0x90, 0xA0, 0xB1 };
            static const uint8_t leads2[] = { 0xE0, 0xE1, 0xE4, 0xED, 0xF0,
                                              0xF1, 0xF3, 0xF4 };
            for (size_t li = 0; li < sizeof(leads2); li++)
                for (size_t off = 1; off + tn <= 4; off++)
                    for (size_t pa = 0; pa < sizeof(pads2); pa++) {
                        int m = utf8_len_from_lead(leads2[li]);
                        if ((size_t)m < off + tn) continue;
                        uint8_t pre[8], suf[8];
                        size_t pl = 0, sl = 0;
                        pre[pl++] = leads2[li];
                        for (size_t q = 1; q < off; q++) pre[pl++] = pads2[pa];
                        for (size_t q = off + tn; q < (size_t)m; q++)
                            suf[sl++] = pads2[pa];
                        char pres[9], sufs[9];
                        memcpy(pres, pre, pl); pres[pl] = 0;
                        memcpy(sufs, suf, sl); sufs[sl] = 0;
                        if (try_witness(c, t, pres, tb, tn, sufs, w, wn))
                            return 1;
                    }
        }
    }
    return 0;
}

/* Exhaustive fragment resolution for short tokens (deep mode, last
 * resort). A byte-fragment token whose every single-character completion
 * is absorbed by an existing merge can still surface when a SECOND
 * character follows: if the pair (absorbing byte, next lead) outranks the
 * absorbing merge, the neighbor is stolen rightward and the fragment
 * survives alone. Enumerate one- and two-character completions over full
 * continuation ranges; the validity gate and the encode check judge every
 * candidate. Bounded: 64 completions x ~200 second characters. */
static int try_fragment_exhaustive(bp_ctx *c, uint32_t t, const uint8_t *tb,
                                   size_t tn, uint8_t *w, size_t *wn)
{
    if (tn > 3) return 0;
    int need = utf8_len_from_lead(tb[0]);
    int cont = 0;
    for (size_t i = (tb[0] & 0xC0) != 0x80 ? 1 : 0; i < tn; i++)
        if ((tb[i] & 0xC0) == 0x80) cont++;
    if (need == 0 && (tb[0] & 0xC0) != 0x80) return 0;

    /* one-character completions, full range */
    if (need > 0 && (size_t)need > tn) {
        size_t missing = (size_t)need - tn;
        if (missing <= 2) {
            uint32_t combos = missing == 1 ? 64 : 64 * 64;
            for (uint32_t v = 0; v < combos; v++) {
                char suf[4];
                suf[0] = (char)(0x80 | (v & 63));
                if (missing == 2) suf[1] = (char)(0x80 | ((v >> 6) & 63));
                suf[missing] = 0;
                if (try_witness(c, t, "", tb, tn, suf, w, wn)) return 1;
            }
            /* two-character completions: finish the char with each byte,
             * then follow with a second character whose leading byte can
             * steal the completing byte's merge partner. Brute over all
             * completions and a broad follower set; the encode check is
             * the judge. */
            for (uint32_t b1 = 0x80; b1 <= 0xBF && missing == 1; b1++) {
                for (uint32_t yy = 0x21; yy <= 0xF4; yy++) {
                    int m2;
                    if (yy <= 0x7E) m2 = 1;
                    else {
                        m2 = utf8_len_from_lead((uint8_t)yy);
                        if (m2 == 0) continue;
                    }
                    char suf[8];
                    size_t sl = 0;
                    suf[sl++] = (char)b1;
                    suf[sl++] = (char)yy;
                    uint8_t padb = yy == 0xE0 ? 0xA0 : yy == 0xF0 ? 0x90
                                                                  : 0x80;
                    for (int q = 1; q < m2; q++) suf[sl++] = (char)padb;
                    suf[sl] = 0;
                    if (try_witness(c, t, "", tb, tn, suf, w, wn))
                        return 1;
                }
            }
        }
    }

    /* targeted rank steal for a bare 2-byte lead: each completion XX is
     * absorbed by the merge (t, XX); a follower YY whose pair (XX, YY)
     * outranks it pulls XX rightward, leaving t emitted alone */
    if (tn == 1 && need == 2) {
        const bp_vocab *v = c->v;
        for (uint32_t xx = 0x80; xx <= 0xBF; xx++) {
            uint64_t absorb = bp_pair_lookup(v, v->byte_id[tb[0]],
                                             v->byte_id[xx]);
            if (absorb == BPV_PAIR_EMPTY) continue; /* one-char case above */
            uint32_t r1 = (uint32_t)(absorb >> 32);
            for (uint32_t yy = 0x20; yy <= 0xF4; yy++) {
                if (yy > 0x7E && utf8_len_from_lead((uint8_t)yy) == 0)
                    continue;
                uint64_t steal = bp_pair_lookup(v, v->byte_id[xx],
                                                v->byte_id[yy]);
                if (steal == BPV_PAIR_EMPTY ||
                    (uint32_t)(steal >> 32) >= r1)
                    continue;
                char suf[8];
                size_t sl = 0;
                suf[sl++] = (char)xx;
                suf[sl++] = (char)yy;
                int m2 = yy > 0x7E ? utf8_len_from_lead((uint8_t)yy) : 1;
                uint8_t padb = yy == 0xE0 ? 0xA0 : yy == 0xF0 ? 0x90 : 0x80;
                for (int q = 1; q < m2; q++) suf[sl++] = (char)padb;
                suf[sl] = 0;
                if (try_witness(c, t, "", tb, tn, suf, w, wn)) return 1;
            }
        }
    }

    /* pure continuation runs: exhaustive lead(+pad) prefixes and pad
     * suffixes across full ranges */
    int all_cont = 1;
    for (size_t i = 0; i < tn; i++)
        if ((tb[i] & 0xC0) != 0x80) all_cont = 0;
    if (all_cont) {
        for (uint32_t lead = 0xC2; lead <= 0xF4; lead++) {
            int m = utf8_len_from_lead((uint8_t)lead);
            if (m == 0) continue;
            for (int off = 1; off + (int)tn <= m; off++) {
                uint32_t padn = (uint32_t)(off - 1 + (m - off - (int)tn));
                if (padn > 2) continue;
                uint32_t combos = 1;
                for (uint32_t q = 0; q < padn; q++) combos *= 64;
                for (uint32_t v = 0; v < combos; v++) {
                    uint8_t pre[8], suf[8];
                    size_t pl = 0, sl = 0;
                    uint32_t vv = v;
                    pre[pl++] = (uint8_t)lead;
                    for (int q = 1; q < off; q++) {
                        pre[pl++] = (uint8_t)(0x80 | (vv & 63));
                        vv >>= 6;
                    }
                    for (int q = off + (int)tn; q < m; q++) {
                        suf[sl++] = (uint8_t)(0x80 | (vv & 63));
                        vv >>= 6;
                    }
                    char pres[9], sufs[9];
                    memcpy(pres, pre, pl); pres[pl] = 0;
                    memcpy(sufs, suf, sl); sufs[sl] = 0;
                    if (try_witness(c, t, pres, tb, tn, sufs, w, wn))
                        return 1;
                }
            }
        }
    }
    return 0;
}

static const char *AFFIX_QUICK[] = {
    "", "a", "A", "z", "Q", "0", " ", ".", ",", "!", "\n", "'", "\"",
    "\xC3\xA9" /* e-acute */, "\xE4\xB8\xAD" /* CJK */, "_", "-", "(",
    "\xC3", "\xE4\xB8", "\xE6\x97", "\xA9", "\x80", "\xB8\xAD", "\xA5",
    "\x80\x80", "\x80\x80\x80", "\x90\x80\x80",
};
static const char *AFFIX_DEEP[] = {
    "", "a", "b", "e", "s", "t", "x", "A", "T", "0", "9", " ", "  ", ".",
    ",", ";", ":", "!", "?", "\n", "\r\n", "\n ", " \n", "'", "\"", "`",
    "\xC3\xA9", "\xC3\x9F" /* sharp s */, "\xE4\xB8\xAD", "\xE6\x97\xA5",
    "\xD0\xB0" /* cyrillic a */, "\xCE\xB1" /* alpha */, "_", "-", "(",
    ")", "[", "/", "#", "$", "\t", "\xC2\xA0" /* NBSP */,
    /* partial characters: tokens holding continuation bytes or bare lead
     * bytes only become witnessable when an affix completes the character;
     * the UTF-8 validity gate discards every combination that fails to
     * glue into a well-formed string */
    "\xC3", "\xD0", "\xE4\xB8", "\xE6", "\xE6\x97",
    "\xF0\x9F\x98" /* emoji lead */, "\xF0\x9F",
    "\xA9", "\xAD", "\x80", "\x80\x80", "\xB8\xAD", "\x97\xA5",
    "\x98\x80" /* grinning-face tail */, "\x80\x80\x80",
    "\x90\x80\x80", "\x9F\x98\x80",
};

/* ---- escaping for the JSON report ------------------------------------- */

static void put_hex(FILE *f, const uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) fprintf(f, "%02x", p[i]);
}

static void put_json_text(FILE *f, const uint8_t *p, size_t n)
{
    /* lossy preview: printable ASCII kept, everything else escaped */
    for (size_t i = 0; i < n; i++) {
        uint8_t b = p[i];
        if (b == '"' || b == '\\') fprintf(f, "\\%c", b);
        else if (b >= 0x20 && b < 0x7F) fputc(b, f);
        else fprintf(f, "\\u%04x", b);
    }
}

/* ---- one token -------------------------------------------------------- */

typedef struct {
    int verdict;
    int reason;
    uint8_t witness[WMAX];
    size_t wlen;
    int searched;
} cres;

static void census_token(const bp_vocab *v, bp_ctx *c, uint32_t t, int deep,
                         cres *r)
{
    memset(r, 0, sizeof(*r));
    const uint8_t *tb = v->blob + v->tok_off[t];
    size_t tn = v->tok_off[t + 1] - v->tok_off[t];

    for (uint32_t i = 0; i < v->hdr.added_count; i++)
        if (v->added[i].id == t) {
            r->verdict = CV_ADDED;
            memcpy(r->witness, tb, tn < WMAX ? tn : WMAX);
            r->wlen = tn < WMAX ? tn : WMAX;
            return;
        }

    if (tn > WMAX) { /* no Qwen3 token approaches this; recorded honestly */
        r->verdict = CV_UNRESOLVED;
        return;
    }

    if (!pure_is_single(c, tb, tn, t)) {
        r->verdict = CV_IMPOSSIBLE;
        r->reason = CR_MERGE_ORDER;
        return;
    }

    if (try_witness(c, t, "", tb, tn, "", r->witness, &r->wlen)) {
        r->verdict = CV_SELF;
        return;
    }

    int reason = prove_impossible(tb, tn);
    if (reason != CR_NONE) {
        r->verdict = CV_IMPOSSIBLE;
        r->reason = reason;
        return;
    }

    if (try_glue(c, t, tb, tn, r->witness, &r->wlen)) {
        r->verdict = CV_CONTEXT;
        return;
    }

    const char **pool = deep ? AFFIX_DEEP : AFFIX_QUICK;
    size_t np = deep ? sizeof(AFFIX_DEEP) / sizeof(*AFFIX_DEEP)
                     : sizeof(AFFIX_QUICK) / sizeof(*AFFIX_QUICK);
    for (size_t pi = 0; pi < np; pi++)
        for (size_t si = 0; si < np; si++) {
            if (!pi && !si) continue; /* the self case, already tried */
            r->searched++;
            if (try_witness(c, t, pool[pi], tb, tn, pool[si],
                            r->witness, &r->wlen)) {
                r->verdict = CV_CONTEXT;
                return;
            }
        }
    if (deep && try_fragment_exhaustive(c, t, tb, tn, r->witness,
                                        &r->wlen)) {
        r->verdict = CV_CONTEXT;
        return;
    }
    r->verdict = CV_UNRESOLVED;
    r->wlen = 0;
}

/* ---- entry points ----------------------------------------------------- */

int bp_census_run(const bp_vocab *v, bp_ctx *c, FILE *json, int deep,
                  uint32_t limit, FILE *out)
{
    uint32_t n = v->hdr.vocab_count;
    if (limit && limit < n) n = limit;
    long counts[5] = { 0 };
    long reasons[CR_COUNT] = { 0 };
    long reverified = 0, reverify_failed = 0;

    if (json)
        fprintf(json, "{\n \"vocab_count\": %u,\n \"censused\": %u,\n"
                      " \"deep\": %s,\n \"tokens\": [\n",
                v->hdr.vocab_count, n, deep ? "true" : "false");

    cres r;
    for (uint32_t t = 0; t < n; t++) {
        const uint8_t *tb = v->blob + v->tok_off[t];
        size_t tn = v->tok_off[t + 1] - v->tok_off[t];
        if (tn == 0) continue; /* unused id in a sparse vocabulary */
        census_token(v, c, t, deep, &r);
        counts[r.verdict]++;
        if (r.verdict == CV_IMPOSSIBLE) reasons[r.reason]++;

        /* independent re-verification of every witness through the full
         * public path; a failure here is a census bug, and fatal */
        if (r.verdict == CV_SELF || r.verdict == CV_CONTEXT) {
            uint32_t ids[WMAX + 8];
            int64_t m = bp_encode(c, (const char *)r.witness, r.wlen, ids,
                                  WMAX + 8, 0);
            reverified++;
            if (m <= 0 || !contains_id(ids, m, t)) {
                reverify_failed++;
                fprintf(stderr, "census: witness re-verification FAILED "
                                "for id %u\n", t);
            }
        }

        if (json) {
            fprintf(json, "  {\"id\":%u,\"hex\":\"", t);
            put_hex(json, tb, tn);
            fprintf(json, "\",\"text\":\"");
            put_json_text(json, tb, tn);
            fprintf(json, "\",\"verdict\":\"%s\"", verdict_name[r.verdict]);
            if (r.verdict == CV_IMPOSSIBLE)
                fprintf(json, ",\"reason\":\"%s\"", reason_name[r.reason]);
            if (r.wlen) {
                fprintf(json, ",\"witness_hex\":\"");
                put_hex(json, r.witness, r.wlen);
                fprintf(json, "\",\"witness\":\"");
                put_json_text(json, r.witness, r.wlen);
                fprintf(json, "\"");
            }
            if (r.verdict == CV_UNRESOLVED)
                fprintf(json, ",\"search_bound\":%d", r.searched);
            fprintf(json, "}%s\n", t + 1 < n ? "," : "");
        }
    }
    if (json) fprintf(json, " ]\n}\n");

    fprintf(out, "census of %u tokens (%s search):\n", n,
            deep ? "deep" : "quick");
    fprintf(out, "  added        %8ld\n", counts[CV_ADDED]);
    fprintf(out, "  self         %8ld\n", counts[CV_SELF]);
    fprintf(out, "  context      %8ld\n", counts[CV_CONTEXT]);
    fprintf(out, "  impossible   %8ld\n", counts[CV_IMPOSSIBLE]);
    fprintf(out, "    merge-order     %8ld\n", reasons[CR_MERGE_ORDER]);
    fprintf(out, "    invalid-utf8    %8ld\n", reasons[CR_INVALID_UTF8]);
    fprintf(out, "    pretoken-shape  %8ld\n", reasons[CR_PRETOKEN_SHAPE]);
    fprintf(out, "    contraction     %8ld\n", reasons[CR_CONTRACTION]);
    fprintf(out, "    nfc-excluded    %8ld\n", reasons[CR_NFC_EXCLUDED]);
    fprintf(out, "  unresolved   %8ld\n", counts[CV_UNRESOLVED]);
    fprintf(out, "  witnesses re-verified: %ld, failures: %ld\n",
            reverified, reverify_failed);
    return reverify_failed ? 1 : 0;
}

int bp_census_witness(const bp_vocab *v, bp_ctx *c, uint32_t t, FILE *out)
{
    if (t >= v->hdr.vocab_count) {
        fprintf(stderr, "witness: id %u out of range\n", t);
        return 1;
    }
    cres r;
    census_token(v, c, t, 1, &r);
    const uint8_t *tb = v->blob + v->tok_off[t];
    size_t tn = v->tok_off[t + 1] - v->tok_off[t];
    fprintf(out, "id %u  bytes ", t);
    put_hex(out, tb, tn);
    fprintf(out, "  verdict %s", verdict_name[r.verdict]);
    if (r.verdict == CV_IMPOSSIBLE)
        fprintf(out, " (%s)", reason_name[r.reason]);
    fprintf(out, "\n");
    if (r.wlen) {
        fprintf(out, "witness bytes ");
        put_hex(out, r.witness, r.wlen);
        fprintf(out, "\nencodes to:");
        uint32_t ids[WMAX + 8];
        int64_t m = bp_encode(c, (const char *)r.witness, r.wlen, ids,
                              WMAX + 8, 0);
        for (int64_t i = 0; i < m; i++)
            fprintf(out, " %u%s", ids[i], ids[i] == t ? "*" : "");
        fprintf(out, "\n");
    }
    return 0;
}
