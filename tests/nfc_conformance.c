/* Unicode normalization conformance for the C normalizer (bp_nfc.c).
 *
 * Runs the pinned NormalizationTest.txt (Unicode 9.0.0 - the version the
 * reference engine's NFC data corresponds to; see docs/FORMAT.md) against
 * the shipped implementation, not against the tables it was generated
 * from. Asserts, per test row with columns c1..c5:
 *
 *     NFC(c1) == c2   NFC(c2) == c2   NFC(c3) == c2
 *     NFC(c4) == c4   NFC(c5) == c4      (UAX #15 conformance clause)
 *
 * and, for every codepoint not listed in Part 1, NFC(cp) == cp.
 *
 *     nfc_conformance <NormalizationTest.txt>       exit 0 pass, 1 fail
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/bp_internal.h"

static size_t enc_utf8(const uint32_t *cps, size_t n, uint8_t *out)
{
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t cp = cps[i];
        if (cp < 0x80) out[m++] = (uint8_t)cp;
        else if (cp < 0x800) {
            out[m++] = 0xC0 | (uint8_t)(cp >> 6);
            out[m++] = 0x80 | (cp & 0x3F);
        } else if (cp < 0x10000) {
            out[m++] = 0xE0 | (uint8_t)(cp >> 12);
            out[m++] = 0x80 | ((cp >> 6) & 0x3F);
            out[m++] = 0x80 | (cp & 0x3F);
        } else {
            out[m++] = 0xF0 | (uint8_t)(cp >> 18);
            out[m++] = 0x80 | ((cp >> 12) & 0x3F);
            out[m++] = 0x80 | ((cp >> 6) & 0x3F);
            out[m++] = 0x80 | (cp & 0x3F);
        }
    }
    return m;
}

/* parse one semicolon field of space-separated hex codepoints */
static size_t parse_field(const char **pp, uint32_t *cps)
{
    const char *p = *pp;
    size_t n = 0;
    while (*p && *p != ';') {
        char *end;
        unsigned long v = strtoul(p, &end, 16);
        if (end == p) break;
        cps[n++] = (uint32_t)v;
        p = end;
        while (*p == ' ') p++;
    }
    if (*p == ';') p++;
    *pp = p;
    return n;
}

static bp_ctx *g_ctx;
static long g_assert, g_fail;

static void check(const uint32_t *in, size_t nin,
                  const uint32_t *want, size_t nwant, int line, int which)
{
    uint8_t inb[512], wantb[512];
    size_t inl = enc_utf8(in, nin, inb);
    size_t wantl = enc_utf8(want, nwant, wantb);
    const uint8_t *out;
    size_t outl;
    g_assert++;
    if (bp_nfc(g_ctx, inb, inl, &out, &outl) != 0 ||
        outl != wantl || memcmp(out, wantb, wantl) != 0) {
        if (g_fail < 10)
            fprintf(stderr, "FAIL line %d column c%d\n", line, which);
        g_fail++;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: nfc_conformance <NormalizationTest.txt>\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 2; }

    /* bp_nfc needs only the NFC flag from the vocabulary handle */
    bp_vocab fake;
    memset(&fake, 0, sizeof(fake));
    fake.hdr.flags = BPV_FLAG_NFC;
    g_ctx = bp_ctx_new(&fake, 0);
    if (!g_ctx) return 2;

    /* Part 1 listing: those codepoints are exempt from the identity rule */
    static uint8_t part1[0x110000 / 8];
    int in_part1 = 0;

    char linebuf[2048];
    int lineno = 0, rows = 0;
    while (fgets(linebuf, sizeof(linebuf), f)) {
        lineno++;
        if (linebuf[0] == '@') {
            in_part1 = strncmp(linebuf, "@Part1", 6) == 0;
            continue;
        }
        if (linebuf[0] == '#' || linebuf[0] == '\n') continue;
        uint32_t c[6][64];
        size_t n[6];
        const char *p = linebuf;
        int ok = 1;
        for (int i = 1; i <= 5; i++) {
            n[i] = parse_field(&p, c[i]);
            if (n[i] == 0) { ok = 0; break; }
        }
        if (!ok) continue;
        rows++;
        if (in_part1 && n[1] == 1)
            part1[c[1][0] >> 3] |= (uint8_t)(1u << (c[1][0] & 7));
        check(c[1], n[1], c[2], n[2], lineno, 1);
        check(c[2], n[2], c[2], n[2], lineno, 2);
        check(c[3], n[3], c[2], n[2], lineno, 3);
        check(c[4], n[4], c[4], n[4], lineno, 4);
        check(c[5], n[5], c[4], n[4], lineno, 5);
    }
    fclose(f);

    /* fixture guard: an empty or truncated test file proves nothing */
    if (rows < 10000) {
        fprintf(stderr, "conformance file suspiciously small: %d rows\n",
                rows);
        return 2;
    }

    /* identity for every codepoint not listed in Part 1 */
    for (uint32_t cp = 0; cp < 0x110000; cp++) {
        if (cp >= 0xD800 && cp <= 0xDFFF) continue;
        if (part1[cp >> 3] & (1u << (cp & 7))) continue;
        check(&cp, 1, &cp, 1, 0, 0);
    }

    printf("nfc conformance: %d rows, %ld assertions, %ld failures\n",
           rows, g_assert, g_fail);
    bp_ctx_free(g_ctx);
    return g_fail ? 1 : 0;
}
