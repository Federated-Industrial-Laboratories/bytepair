/* .bpv loader: mmap, validate, index. Validation is total — a file that
 * fails any check produces a clean BP_E_FORMAT and no usable handle. */
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "bp_internal.h"

#define BPV_HDR_BYTES 136
#define BPV_FIRST_SECTION 192

static int section_ok(const bpv_section *s, size_t file_size, uint64_t elem)
{
    if (s->off < BPV_FIRST_SECTION || (s->off & 63)) return 0;
    if (s->off > file_size || s->size > file_size - s->off) return 0;
    if (elem && (s->size % elem)) return 0;
    return 1;
}

static int cpu_has_avx2_bmi2(void)
{
#if defined(__x86_64__)
    uint32_t a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                     : "a"(7u), "c"(0u));
    return (b >> 5) & (b >> 8) & 1; /* AVX2 (EBX bit 5) and BMI2 (EBX bit 8) */
#else
    return 0;
#endif
}

bp_vocab *bp_vocab_open(const char *path, int *err)
{
    int e = BP_E_IO;
    bp_vocab *v = NULL;
    uint8_t *map = MAP_FAILED;
    size_t size = 0;

    if (!path) { e = BP_E_ARG; goto fail; }

    int fd = open(path, O_RDONLY);
    if (fd < 0) goto fail;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < BPV_FIRST_SECTION) {
        close(fd);
        goto fail;
    }
    size = (size_t)st.st_size;
    map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) goto fail;

    e = BP_E_FORMAT;
    bpv_header h;
    memcpy(&h, map, BPV_HDR_BYTES);
    if (h.magic != BPV_MAGIC || h.format_version != 1) goto fail;
    if (h.profile_id != BPV_PROFILE_NONE && h.profile_id != BPV_PROFILE_GPT4)
        goto fail;
    if (h.vocab_count == 0 || h.vocab_count > (1u << 21)) goto fail;
    if (h.added_count > 64) goto fail;
    if (h.pair_slots_log2 > 32) goto fail;

    uint64_t slots = 1ull << h.pair_slots_log2;
    if (!section_ok(&h.token_offsets, size, 4) ||
        h.token_offsets.size != 4ull * (h.vocab_count + 1))
        goto fail;
    if (!section_ok(&h.token_blob, size, 0)) goto fail;
    if (!section_ok(&h.pair_table, size, 16) ||
        h.pair_table.size != 16ull * slots)
        goto fail;
    if (!section_ok(&h.byte_to_id, size, 4) || h.byte_to_id.size != 1024)
        goto fail;
    if (!section_ok(&h.added, size, 16) ||
        h.added.size != 16ull * h.added_count)
        goto fail;
    if (!section_ok(&h.meta, size, 0)) goto fail;

    const uint32_t *tok_off = (const uint32_t *)(map + h.token_offsets.off);
    for (uint32_t i = 0; i < h.vocab_count; i++)
        if (tok_off[i] > tok_off[i + 1]) goto fail;
    if (tok_off[h.vocab_count] > h.token_blob.size) goto fail;

    const uint32_t *byte_id = (const uint32_t *)(map + h.byte_to_id.off);
    for (int i = 0; i < 256; i++) {
        uint32_t id = byte_id[i];
        if (id >= h.vocab_count) goto fail;
        if (tok_off[id + 1] - tok_off[id] != 1) goto fail; /* must be 1 byte */
    }

    const bpv_added *added = (const bpv_added *)(map + h.added.off);
    for (uint32_t i = 0; i < h.added_count; i++) {
        if (added[i].id >= h.vocab_count) goto fail;
        if (added[i].offset > h.token_blob.size ||
            added[i].len > h.token_blob.size - added[i].offset)
            goto fail;
        if (added[i].len == 0 || added[i].len > 250) goto fail;
        if (i && added[i].id <= added[i - 1].id) goto fail; /* sorted, unique */
    }

    /* meta: two length-prefixed strings minimum (name, converter version) */
    if (h.meta.size < 4) goto fail;

    v = calloc(1, sizeof(*v));
    if (!v) { e = BP_E_NOMEM; goto fail; }
    v->map = map;
    v->map_size = size;
    v->hdr = h;
    v->tok_off = tok_off;
    v->blob = map + h.token_blob.off;
    v->pairs = (const uint64_t *)(map + h.pair_table.off);
    v->pair_mask = slots - 1;
    v->byte_id = byte_id;
    v->added = added;

    /* meta name: first length-prefixed string, copied NUL-terminated */
    {
        const uint8_t *m = map + h.meta.off;
        uint32_t nlen;
        memcpy(&nlen, m, 4);
        if (nlen > h.meta.size - 4) goto fail_v;
        char *name = malloc((size_t)nlen + 1);
        if (!name) { e = BP_E_NOMEM; goto fail_v; }
        memcpy(name, m + 4, nlen);
        name[nlen] = 0;
        v->meta_name = name;
    }

    /* added-token matcher order: longest first, and all share a first byte */
    if (h.added_count) {
        v->added_first = v->blob[added[0].offset];
        for (uint32_t i = 0; i < h.added_count; i++) {
            if (v->blob[added[i].offset] != v->added_first) goto fail_v;
            v->added_order[i] = (uint16_t)i;
        }
        for (uint32_t i = 1; i < h.added_count; i++) { /* insertion sort */
            uint16_t k = v->added_order[i];
            uint32_t j = i;
            while (j && added[v->added_order[j - 1]].len < added[k].len) {
                v->added_order[j] = v->added_order[j - 1];
                j--;
            }
            v->added_order[j] = k;
        }
    }

    v->use_avx2 = cpu_has_avx2_bmi2();
    if (err) *err = BP_OK;
    return v;

fail_v:
    free((void *)v->meta_name);
    free(v);
    v = NULL;
fail:
    if (map != MAP_FAILED) munmap(map, size);
    if (err) *err = e;
    return NULL;
}

void bp_vocab_close(bp_vocab *v)
{
    if (!v) return;
    munmap((void *)v->map, v->map_size);
    free((void *)v->meta_name);
    free(v);
}

uint32_t bp_vocab_size(const bp_vocab *v) { return v->hdr.vocab_count; }
const char *bp_vocab_name(const bp_vocab *v) { return v->meta_name; }

bp_ctx *bp_ctx_new(const bp_vocab *v, int cache_log2)
{
    if (!v || cache_log2 > 24) return NULL;
    bp_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->v = v;
    {
        const char *e = getenv("BYTEPAIR_FORCE_SCALAR");
        c->env_force_scalar = e && e[0] == '1' && e[1] == 0;
    }
    int lg = (cache_log2 == BP_CTX_DEFAULT_CACHE) ? 15 : cache_log2;
    if (lg > 0) {
        c->cache = calloc((size_t)1 << lg, sizeof(bp_cache_ent));
        if (!c->cache) { free(c); return NULL; }
        c->cache_mask = ((uint32_t)1 << lg) - 1;
    }
    return c;
}

void bp_ctx_free(bp_ctx *c)
{
    if (!c) return;
    free(c->norm.p);
    free(c->work.p);
    free(c->ids);
    free(c->cache);
    free(c);
}
