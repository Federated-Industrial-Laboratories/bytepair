/* NFC normalization (UAX #15, canonical only), with two fast paths:
 * pure-ASCII input and quick-check-clean input are returned as-is.
 * Tables come from src/tables/bp_uctables.* (generated, pinned UCD). */
#include <string.h>
#include "bp_internal.h"
#include "tables/bp_uctables.h"

/* Hangul (algorithmic, UAX #15 section 3.12) */
#define SBASE 0xAC00u
#define LBASE 0x1100u
#define VBASE 0x1161u
#define TBASE 0x11A7u
#define LCOUNT 19
#define VCOUNT 21
#define TCOUNT 28
#define NCOUNT (VCOUNT * TCOUNT)
#define SCOUNT (LCOUNT * NCOUNT)

static int is_hangul_s(uint32_t cp) { return cp >= SBASE && cp < SBASE + SCOUNT; }

/* Returns 1 if seg is pure ASCII. */
static int all_ascii(const uint8_t *p, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        if (w & 0x8080808080808080ull) return 0;
    }
    for (; i < n; i++)
        if (p[i] & 0x80) return 0;
    return 1;
}

/* NFC quick check over the segment: 1 = definitely NFC already. */
static int nfc_quick(const uint8_t *p, size_t n)
{
    const uint8_t *end = p + n;
    uint8_t prev_ccc = 0;
    while (p < end) {
        size_t adv;
        uint32_t cp = bp_utf8_next(p, end, &adv);
        p += adv;
        if (cp == 0xFFFFFFFFu) { prev_ccc = 0; continue; } /* opaque byte */
        if (cp < 0x80) { prev_ccc = 0; continue; }
        uint8_t ccc = bp_ccc(cp);
        if (ccc && prev_ccc > ccc) return 0;
        if (bp_nfc_qc(cp) != 1) return 0;
        prev_ccc = ccc;
    }
    return 1;
}

typedef struct { uint32_t cp; uint8_t ccc; } cpe;

static size_t decompose_one(uint32_t cp, cpe *out)
{
    if (is_hangul_s(cp)) {
        uint32_t s = cp - SBASE;
        size_t k = 0;
        out[k++] = (cpe){ LBASE + s / NCOUNT, 0 };
        out[k++] = (cpe){ VBASE + (s % NCOUNT) / TCOUNT, 0 };
        if (s % TCOUNT)
            out[k++] = (cpe){ TBASE + s % TCOUNT, 0 };
        return k;
    }
    uint32_t d[4];
    int n = bp_decomp(cp, d);
    if (n == 0) {
        out[0] = (cpe){ cp, bp_ccc(cp) };
        return 1;
    }
    for (int i = 0; i < n; i++)
        out[i] = (cpe){ d[i], bp_ccc(d[i]) };
    return (size_t)n;
}

static uint32_t compose_pair(uint32_t a, uint32_t b)
{
    /* Hangul first (algorithmic), then the table. */
    if (a >= LBASE && a < LBASE + LCOUNT && b >= VBASE && b < VBASE + VCOUNT)
        return SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT;
    if (is_hangul_s(a) && !((a - SBASE) % TCOUNT) &&
        b > TBASE && b < TBASE + TCOUNT)
        return a + (b - TBASE);
    return bp_compose(a, b);
}

int bp_nfc(bp_ctx *c, const uint8_t *seg, size_t len,
           const uint8_t **out, size_t *out_len)
{
    if (!(c->v->hdr.flags & BPV_FLAG_NFC) || len == 0 ||
        all_ascii(seg, len) || nfc_quick(seg, len)) {
        *out = seg;
        *out_len = len;
        return 0;
    }

    /* Full path. Worst case each codepoint decomposes to 4. Malformed bytes
     * are carried through as sentinel entries re-emitted verbatim; to keep
     * their byte values we tag them cp = 0x110000 + byte. */
    size_t max_cp = len * 4 + 1;
    if (bp_buf_reserve(&c->work, max_cp * sizeof(cpe)) != 0)
        return BP_E_NOMEM;
    cpe *buf = (cpe *)c->work.p;
    size_t n = 0;

    const uint8_t *p = seg, *end = seg + len;
    while (p < end) {
        size_t adv;
        uint32_t cp = bp_utf8_next(p, end, &adv);
        if (cp == 0xFFFFFFFFu)
            buf[n++] = (cpe){ 0x110000u + p[0], 0 };
        else
            n += decompose_one(cp, buf + n);
        p += adv;
    }

    /* Canonical ordering: stable insertion sort within nonzero-ccc runs. */
    for (size_t i = 1; i < n; i++) {
        if (buf[i].ccc == 0) continue;
        size_t j = i;
        cpe t = buf[i];
        while (j > 0 && buf[j - 1].ccc > t.ccc) {
            buf[j] = buf[j - 1];
            j--;
        }
        buf[j] = t;
    }

    /* Canonical composition (UAX #15 algorithm). */
    if (n) {
        size_t w = 0;                  /* write index                  */
        size_t starter = (size_t)-1;   /* index in [0,w) of last starter */
        for (size_t i = 0; i < n; i++) {
            cpe cur = buf[i];
            if (starter != (size_t)-1 && cur.cp < 0x110000u &&
                (w == starter + 1 || buf[w - 1].ccc < cur.ccc)) {
                uint32_t comp = compose_pair(buf[starter].cp, cur.cp);
                if (comp) {
                    buf[starter].cp = comp; /* ccc stays 0 */
                    continue;
                }
            }
            if (cur.ccc == 0)
                starter = (cur.cp >= 0x110000u) ? (size_t)-1 : w;
            buf[w++] = cur;
        }
        n = w;
    }

    /* Encode back to UTF-8 (sentinels emit their raw byte). */
    if (bp_buf_reserve(&c->norm, n * 4 + 1) != 0)
        return BP_E_NOMEM;
    uint8_t *o = c->norm.p;
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t cp = buf[i].cp;
        if (cp >= 0x110000u) {
            o[m++] = (uint8_t)(cp - 0x110000u);
        } else if (cp < 0x80) {
            o[m++] = (uint8_t)cp;
        } else if (cp < 0x800) {
            o[m++] = 0xC0 | (uint8_t)(cp >> 6);
            o[m++] = 0x80 | (cp & 0x3F);
        } else if (cp < 0x10000) {
            o[m++] = 0xE0 | (uint8_t)(cp >> 12);
            o[m++] = 0x80 | ((cp >> 6) & 0x3F);
            o[m++] = 0x80 | (cp & 0x3F);
        } else {
            o[m++] = 0xF0 | (uint8_t)(cp >> 18);
            o[m++] = 0x80 | ((cp >> 12) & 0x3F);
            o[m++] = 0x80 | ((cp >> 6) & 0x3F);
            o[m++] = 0x80 | (cp & 0x3F);
        }
    }
    *out = o;
    *out_len = m;
    return 0;
}
