/* bp_encode / bp_count: the full pipeline.
 *
 *   [added-token scan over raw bytes]   (default; BP_RAW disables)
 *      -> per segment: NFC -> split scanner -> per-pretoken BPE
 *
 * Added tokens are matched on raw input before normalization, longest match
 * first at each position, because that is what HuggingFace does (verified
 * empirically; see docs/FORMAT.md). */
#include <string.h>
#include "bp_internal.h"

static int encode_segment(bp_ctx *c, const uint8_t *seg, size_t len,
                          bp_sink *sink, int simd)
{
    if (len == 0) return 0;
    const uint8_t *norm;
    size_t norm_len;
    int rc = bp_nfc(c, seg, len, &norm, &norm_len);
    if (rc) return rc;
    return bp_scan_gpt4(c, norm, norm_len, sink, simd);
}

/* Longest added token matching at p, or NULL. */
static const bpv_added *match_added(const bp_vocab *v, const uint8_t *p,
                                    size_t avail)
{
    for (uint32_t k = 0; k < v->hdr.added_count; k++) {
        const bpv_added *a = &v->added[v->added_order[k]];
        if (a->len <= avail && !memcmp(p, v->blob + a->offset, a->len))
            return a;
    }
    return NULL;
}

int64_t bp_encode(bp_ctx *c, const char *utf8, size_t len,
                  uint32_t *out, size_t out_cap, uint32_t flags)
{
    if (!c || (!utf8 && len) || (!out && out_cap)) return BP_E_ARG;
    const bp_vocab *v = c->v;
    const uint8_t *t = (const uint8_t *)utf8;
    bp_sink s = { out, out_cap, 0 };
    int simd = v->use_avx2 && !(flags & BP_FORCE_SCALAR) &&
               !c->env_force_scalar;
    int rc;

    if ((flags & BP_RAW) || v->hdr.added_count == 0) {
        rc = encode_segment(c, t, len, &s, simd);
        return rc ? rc : s.total;
    }

    size_t seg_start = 0, i = 0;
    while (i < len) {
        if (v->added_single_first) {
            const uint8_t *hit = memchr(t + i, v->added_first, len - i);
            if (!hit) break;
            i = (size_t)(hit - t);
        } else {
            while (i < len &&
                   !(v->added_first_map[t[i] >> 3] & (1u << (t[i] & 7))))
                i++;
            if (i >= len) break;
        }
        const bpv_added *a = match_added(v, t + i, len - i);
        if (!a) { i++; continue; }
        if ((rc = encode_segment(c, t + seg_start, i - seg_start, &s, simd)))
            return rc;
        bp_sink_put1(&s, a->id);
        i += a->len;
        seg_start = i;
    }
    if ((rc = encode_segment(c, t + seg_start, len - seg_start, &s, simd)))
        return rc;
    return s.total;
}

int64_t bp_count(bp_ctx *c, const char *utf8, size_t len, uint32_t flags)
{
    return bp_encode(c, utf8, len, NULL, 0, flags);
}

int64_t bp_decode(bp_ctx *c, const uint32_t *ids, size_t n,
                  char *out, size_t out_cap, uint32_t flags)
{
    if (!c || (!ids && n) || (!out && out_cap)) return BP_E_ARG;
    const bp_vocab *v = c->v;
    int64_t total = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i];
        if (id >= v->hdr.vocab_count) return BP_E_RANGE;
        if (flags & BP_SKIP_SPECIAL) {
            /* added array is sorted by id */
            uint32_t lo = 0, hi = v->hdr.added_count;
            int skip = 0;
            while (lo < hi) {
                uint32_t mid = (lo + hi) / 2;
                if (v->added[mid].id < id) lo = mid + 1;
                else if (v->added[mid].id > id) hi = mid;
                else { skip = v->added[mid].flags & 1; break; }
            }
            if (skip) continue;
        }
        uint32_t off = v->tok_off[id], end = v->tok_off[id + 1];
        size_t tl = end - off;
        if (tl == 0) return BP_E_RANGE; /* unused id in a sparse vocabulary */
        if ((size_t)total < out_cap) {
            size_t room = out_cap - (size_t)total;
            memcpy(out + total, v->blob + off, tl < room ? tl : room);
        }
        total += (int64_t)tl;
    }
    return total;
}
