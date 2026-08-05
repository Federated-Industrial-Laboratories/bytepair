/* BPE merge over one pretoken, plus the per-context pretoken cache.
 *
 * Short pretokens (the overwhelming case) use a flat-array rescan: find the
 * lowest-rank adjacent pair, merge, repeat. Long pretokens switch to a
 * linked-list + binary min-heap so pathological runs stay O(n log n).
 * Ties on rank break toward the leftmost pair, matching HF. */
#include <string.h>
#include "bp_internal.h"

#define SHORT_MAX 64
#define CACHE_MAX_BYTES 38
#define CACHE_MAX_IDS 16

/* ---- short path: flat array rescan ------------------------------------ */

static size_t merge_short(const bp_vocab *v, uint32_t *ids, size_t n)
{
    while (n >= 2) {
        uint32_t best_rank = 0xFFFFFFFFu, best_id = 0;
        size_t best_at = n;
        for (size_t i = 0; i + 1 < n; i++) {
            uint64_t val = bp_pair_lookup(v, ids[i], ids[i + 1]);
            if (val == BPV_PAIR_EMPTY) continue;
            uint32_t rank = (uint32_t)(val >> 32);
            if (rank < best_rank) {
                best_rank = rank;
                best_id = (uint32_t)val;
                best_at = i;
            }
        }
        if (best_at == n) break;
        ids[best_at] = best_id;
        memmove(ids + best_at + 1, ids + best_at + 2,
                (n - best_at - 2) * sizeof(uint32_t));
        n--;
    }
    return n;
}

/* ---- long path: linked list + lazy min-heap --------------------------- */

typedef struct {
    uint32_t rank;
    uint32_t pos;  /* left element index at push time */
    uint32_t gen;  /* left element generation at push time */
} heap_ent;

typedef struct {
    uint32_t id;
    int32_t  prev, next;
    uint32_t gen;
} node;

static inline int heap_less(const heap_ent *a, const heap_ent *b)
{
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->pos < b->pos; /* leftmost tie-break */
}

static void heap_push(heap_ent *h, size_t *hn, heap_ent e)
{
    size_t i = (*hn)++;
    h[i] = e;
    while (i) {
        size_t p = (i - 1) / 2;
        if (!heap_less(&h[i], &h[p])) break;
        heap_ent t = h[i]; h[i] = h[p]; h[p] = t;
        i = p;
    }
}

static heap_ent heap_pop(heap_ent *h, size_t *hn)
{
    heap_ent top = h[0];
    h[0] = h[--(*hn)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *hn && heap_less(&h[l], &h[m])) m = l;
        if (r < *hn && heap_less(&h[r], &h[m])) m = r;
        if (m == i) break;
        heap_ent t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
    return top;
}

static int merge_long(bp_ctx *c, const uint8_t *p, size_t n, bp_sink *sink)
{
    const bp_vocab *v = c->v;
    /* work buffer: n nodes + up to (heap of ~2n live entries, grown) */
    size_t need = n * sizeof(node) + 2 * n * sizeof(heap_ent);
    if (bp_buf_reserve(&c->work, need) != 0) return BP_E_NOMEM;
    node *nd = (node *)c->work.p;
    heap_ent *heap = (heap_ent *)(c->work.p + n * sizeof(node));
    size_t heap_cap = 2 * n, hn = 0;

    for (size_t i = 0; i < n; i++)
        nd[i] = (node){ v->byte_id[p[i]], (int32_t)i - 1,
                        i + 1 < n ? (int32_t)(i + 1) : -1, 0 };

    for (size_t i = 0; i + 1 < n; i++) {
        uint64_t val = bp_pair_lookup(v, nd[i].id, nd[i + 1].id);
        if (val != BPV_PAIR_EMPTY)
            heap_push(heap, &hn, (heap_ent){ (uint32_t)(val >> 32),
                                             (uint32_t)i, 0 });
    }

    while (hn) {
        heap_ent e = heap_pop(heap, &hn);
        uint32_t i = e.pos;
        if (nd[i].gen != e.gen || nd[i].next < 0) continue; /* stale */
        uint32_t j = (uint32_t)nd[i].next;
        uint64_t val = bp_pair_lookup(v, nd[i].id, nd[j].id);
        if (val == BPV_PAIR_EMPTY || (uint32_t)(val >> 32) != e.rank)
            continue; /* pair changed since push */

        nd[i].id = (uint32_t)val;
        nd[i].gen++;
        nd[i].next = nd[j].next;
        if (nd[j].next >= 0) nd[nd[j].next].prev = (int32_t)i;
        nd[j].next = nd[j].prev = -1; /* dead */

        if (hn + 2 > heap_cap) {
            /* compact: drop stale entries in place */
            size_t w = 0;
            for (size_t k = 0; k < hn; k++) {
                heap_ent x = heap[k];
                if (nd[x.pos].gen == x.gen && nd[x.pos].next >= 0)
                    heap[w++] = x;
            }
            hn = 0;
            for (size_t k = 0; k < w; k++) heap_push(heap, &hn, heap[k]);
            if (hn + 2 > heap_cap) return BP_E_NOMEM; /* cannot happen: live
                pairs <= n-1 <= heap_cap - 2 for n >= 2 */
        }
        if (nd[i].prev >= 0) {
            uint32_t l = (uint32_t)nd[i].prev;
            uint64_t lv = bp_pair_lookup(v, nd[l].id, nd[i].id);
            if (lv != BPV_PAIR_EMPTY)
                heap_push(heap, &hn, (heap_ent){ (uint32_t)(lv >> 32), l,
                                                 nd[l].gen });
        }
        if (nd[i].next >= 0) {
            uint64_t rv = bp_pair_lookup(v, nd[i].id,
                                         nd[(uint32_t)nd[i].next].id);
            if (rv != BPV_PAIR_EMPTY)
                heap_push(heap, &hn, (heap_ent){ (uint32_t)(rv >> 32), i,
                                                 nd[i].gen });
        }
    }

    if (bp_ids_reserve(c, n) != 0) return BP_E_NOMEM;
    size_t count = 0;
    for (int32_t i = 0; i >= 0; i = nd[i].next)
        c->ids[count++] = nd[i].id;
    bp_sink_put(sink, c->ids, count);
    return 0;
}

/* ---- entry point with cache ------------------------------------------- */

int bp_bpe_pretoken(bp_ctx *c, const uint8_t *p, size_t n, bp_sink *sink)
{
    const bp_vocab *v = c->v;
    if (n == 0) return 0;

    /* Tiny pretokens dominate code-like text; they need no cache. */
    if (n == 1) {
        bp_sink_put1(sink, v->byte_id[p[0]]);
        return 0;
    }
    if (n == 2) {
        uint32_t a = v->byte_id[p[0]], b = v->byte_id[p[1]];
        uint64_t val = bp_pair_lookup(v, a, b);
        if (val != BPV_PAIR_EMPTY) {
            bp_sink_put1(sink, (uint32_t)val);
        } else {
            bp_sink_put1(sink, a);
            bp_sink_put1(sink, b);
        }
        return 0;
    }

    bp_cache_ent *ent = NULL;
    uint64_t h = 0;
    if (c->cache && n <= CACHE_MAX_BYTES) {
        h = bp_hash_bytes(p, n);
        ent = &c->cache[h & c->cache_mask];
        if (ent->len == n && ent->hash == h && !memcmp(ent->bytes, p, n)) {
            bp_sink_put(sink, ent->ids, ent->nids);
            return 0;
        }
    }

    if (n > SHORT_MAX)
        return merge_long(c, p, n, sink);

    uint32_t ids[SHORT_MAX];
    for (size_t i = 0; i < n; i++)
        ids[i] = v->byte_id[p[i]];
    size_t m = merge_short(v, ids, n);

    if (ent && m <= CACHE_MAX_IDS) {
        ent->hash = h;
        ent->len = (uint8_t)n;
        ent->nids = (uint8_t)m;
        memcpy(ent->bytes, p, n);
        memcpy(ent->ids, ids, m * sizeof(uint32_t));
    }
    bp_sink_put(sink, ids, m);
    return 0;
}
