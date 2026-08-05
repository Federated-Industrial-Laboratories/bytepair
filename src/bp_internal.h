/* Internal definitions shared across bytepair translation units. */
#ifndef BP_INTERNAL_H
#define BP_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "../include/bytepair.h"

/* ---------- .bpv on-disk format (little-endian, see docs/DESIGN.md) ---- */

#define BPV_MAGIC 0x31565042u /* "BPV1" */

enum { BPV_FLAG_NFC = 1u << 0 };
enum { BPV_PROFILE_NONE = 0, BPV_PROFILE_GPT4 = 1 };

typedef struct {
    uint64_t off;
    uint64_t size;
} bpv_section;

/* 64-byte fixed part, then the section table. Struct layout matches the
 * file: keep fields 8-byte packed by construction (no implicit padding). */
typedef struct {
    uint32_t magic;
    uint32_t format_version;
    uint32_t flags;
    uint32_t profile_id;
    uint32_t vocab_count;   /* max id + 1, added tokens included */
    uint32_t added_count;
    uint32_t pair_slots_log2;
    uint32_t reserved;
    bpv_section token_offsets; /* u32[vocab_count+1] into token_blob */
    bpv_section token_blob;
    bpv_section pair_table;    /* u64 key, u64 value pairs           */
    bpv_section byte_to_id;    /* u32[256]                           */
    bpv_section added;         /* bpv_added[added_count]             */
    bpv_section meta;
    uint64_t source_hash;      /* FNV-1a-64 of source tokenizer.json */
} bpv_header;

typedef struct {
    uint32_t id;
    uint32_t offset; /* into token_blob */
    uint32_t len;
    uint32_t flags;  /* bit0: special */
} bpv_added;

#define BPV_PAIR_EMPTY 0xFFFFFFFFFFFFFFFFull

/* ---------- in-memory handles ------------------------------------------ */

struct bp_vocab {
    const uint8_t  *map;      /* whole file */
    size_t          map_size;
    bpv_header      hdr;      /* copied out of the map, validated */
    const uint32_t *tok_off;  /* vocab_count+1 */
    const uint8_t  *blob;
    const uint64_t *pairs;    /* 2 u64 per slot: key, value */
    uint64_t        pair_mask;/* slots-1 */
    const uint32_t *byte_id;  /* 256 */
    const bpv_added *added;   /* added_count, sorted by id in file */
    const char     *meta_name;/* NUL-terminated copy */
    /* added-token matcher: candidate indexes sorted by length desc */
    uint16_t        added_order[64];
    uint8_t         added_first;   /* first byte shared by all added tokens */
    int             use_avx2;      /* CPU supports the AVX2+BMI2 kernels */
};

/* Growable scratch owned by a context. */
typedef struct {
    uint8_t *p;
    size_t   cap;
} bp_buf;

typedef struct {
    uint64_t hash;
    uint8_t  len;    /* pretoken byte length, 0 = empty slot */
    uint8_t  nids;
    uint8_t  bytes[38];
    uint32_t ids[16];
} bp_cache_ent; /* 112 bytes */

struct bp_ctx {
    const bp_vocab *v;
    bp_buf norm;      /* NFC output */
    bp_buf work;      /* codepoint scratch for NFC (u32 cps + u8 ccc)     */
    uint32_t *ids;    /* merge scratch */
    size_t    ids_cap;
    bp_cache_ent *cache;
    uint32_t      cache_mask; /* entries-1, or 0 with cache NULL */
    int           env_force_scalar; /* BYTEPAIR_FORCE_SCALAR=1 at ctx_new */
};

/* ---------- shared helpers --------------------------------------------- */

static inline uint64_t bp_fmix64(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

/* Cache hash for short pretokens: wide loads instead of a byte loop. Only
 * used for keys the cache itself verifies with memcmp, so the only
 * requirement is good dispersion. */
static inline uint64_t bp_hash_bytes(const uint8_t *p, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull ^ (n * 0x9e3779b97f4a7c15ull);
    while (n >= 8) {
        uint64_t w;
        __builtin_memcpy(&w, p, 8);
        h = bp_fmix64(h ^ w);
        p += 8;
        n -= 8;
    }
    if (n) {
        uint64_t w = 0;
        __builtin_memcpy(&w, p, n);
        h = bp_fmix64(h ^ w);
    }
    return h;
}

/* Output sink: counts everything, writes what fits. No function pointers on
 * the hot path. */
typedef struct {
    uint32_t *out;
    size_t    cap;
    int64_t   total;
} bp_sink;

static inline void bp_sink_put(bp_sink *s, const uint32_t *ids, size_t n)
{
    if ((size_t)s->total < s->cap) {
        size_t room = s->cap - (size_t)s->total;
        __builtin_memcpy(s->out + s->total, ids,
                         (n < room ? n : room) * sizeof(uint32_t));
    }
    s->total += (int64_t)n;
}

static inline void bp_sink_put1(bp_sink *s, uint32_t id)
{
    if ((size_t)s->total < s->cap) s->out[s->total] = id;
    s->total++;
}

/* Pair table probe: returns value or BPV_PAIR_EMPTY. */
static inline uint64_t bp_pair_lookup(const bp_vocab *v, uint32_t a, uint32_t b)
{
    uint64_t key = ((uint64_t)a << 32) | b;
    uint64_t slot = bp_fmix64(key) & v->pair_mask;
    for (;;) {
        uint64_t k = v->pairs[slot * 2];
        if (k == key) return v->pairs[slot * 2 + 1];
        if (k == BPV_PAIR_EMPTY) return BPV_PAIR_EMPTY;
        slot = (slot + 1) & v->pair_mask;
    }
}

/* UTF-8: decode one codepoint at p (limit end). Returns codepoint and sets
 * *adv. Malformed input: returns 0xFFFFFFFF with *adv = 1 (caller treats the
 * byte as an opaque "other"-class unit; it still tokenizes as raw bytes). */
uint32_t bp_utf8_next(const uint8_t *p, const uint8_t *end, size_t *adv);

/* Buffer helpers (bp_util.c). Return 0 on success, -1 on OOM. */
int bp_buf_reserve(bp_buf *b, size_t need);
int bp_ids_reserve(bp_ctx *c, size_t need);

/* NFC (bp_nfc.c): normalize seg[0..len) if needed. On return, *out points
 * either at seg (already NFC) or at c->norm.p, with *out_len set. Returns
 * 0 on success, BP_E_NOMEM on allocation failure. */
int bp_nfc(bp_ctx *c, const uint8_t *seg, size_t len,
           const uint8_t **out, size_t *out_len);

/* Scanner (bp_scan.c): pretokenize text[0..len) (already normalized) and
 * BPE-encode each pretoken into the sink via the context. simd enables the
 * AVX2 run kernels (caller has checked CPU support and flags). Returns 0 or
 * BP_E_NOMEM. */
int bp_scan_gpt4(bp_ctx *c, const uint8_t *text, size_t len,
                 bp_sink *sink, int simd);

/* BPE (bp_bpe.c): encode one pretoken (raw bytes) into the sink. Uses the
 * context cache. Returns 0 or BP_E_NOMEM. */
int bp_bpe_pretoken(bp_ctx *c, const uint8_t *p, size_t n, bp_sink *sink);

/* AVX2 run kernels (bp_scan_avx2.c, compiled with -mavx2 -mbmi2; call only
 * when the CPU reports support). Each extends a run starting at i over
 * ASCII bytes of the given kind and returns the first index not consumed;
 * the caller continues scalar from there (non-ASCII or end of run). */
size_t bp_ascii_letter_run_avx2(const uint8_t *t, size_t i, size_t len);
size_t bp_ascii_symbol_run_avx2(const uint8_t *t, size_t i, size_t len);

#endif /* BP_INTERNAL_H */
