/* Shared small utilities: UTF-8 decode, growable buffers, error strings. */
#include <stdlib.h>
#include <string.h>
#include "bp_internal.h"

uint32_t bp_utf8_next(const uint8_t *p, const uint8_t *end, size_t *adv)
{
    uint8_t b0 = p[0];
    if (b0 < 0x80) { *adv = 1; return b0; }
    if (b0 < 0xC2) goto bad; /* continuation or overlong lead */
    if (b0 < 0xE0) {
        if (end - p < 2 || (p[1] & 0xC0) != 0x80) goto bad;
        *adv = 2;
        return ((uint32_t)(b0 & 0x1F) << 6) | (p[1] & 0x3F);
    }
    if (b0 < 0xF0) {
        if (end - p < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80)
            goto bad;
        uint32_t cp = ((uint32_t)(b0 & 0x0F) << 12) |
                      ((uint32_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) goto bad;
        *adv = 3;
        return cp;
    }
    if (b0 < 0xF5) {
        if (end - p < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 ||
            (p[3] & 0xC0) != 0x80)
            goto bad;
        uint32_t cp = ((uint32_t)(b0 & 0x07) << 18) |
                      ((uint32_t)(p[1] & 0x3F) << 12) |
                      ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) goto bad;
        *adv = 4;
        return cp;
    }
bad:
    *adv = 1;
    return 0xFFFFFFFFu;
}

int bp_buf_reserve(bp_buf *b, size_t need)
{
    if (b->cap >= need) return 0;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < need) cap *= 2;
    uint8_t *p = realloc(b->p, cap);
    if (!p) return -1;
    b->p = p;
    b->cap = cap;
    return 0;
}

int bp_ids_reserve(bp_ctx *c, size_t need)
{
    if (c->ids_cap >= need) return 0;
    size_t cap = c->ids_cap ? c->ids_cap : 1024;
    while (cap < need) cap *= 2;
    uint32_t *p = realloc(c->ids, cap * sizeof(uint32_t));
    if (!p) return -1;
    c->ids = p;
    c->ids_cap = cap;
    return 0;
}

const char *bp_strerror(int err)
{
    switch (err) {
    case BP_OK:       return "ok";
    case BP_E_IO:     return "file open, stat, or mmap failed";
    case BP_E_FORMAT: return "not a valid .bpv vocabulary file";
    case BP_E_RANGE:  return "token id out of range";
    case BP_E_NOMEM:  return "out of memory";
    case BP_E_ARG:    return "invalid argument";
    default:          return "unknown error";
    }
}

const char *bp_version(void) { return "0.1.0"; }
