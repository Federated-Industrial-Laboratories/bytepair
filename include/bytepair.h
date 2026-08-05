/* bytepair - dependency-free byte-level BPE tokenizer library.
 *
 * Exact-match guarantee: with a .bpv built from the Qwen3 tokenizer.json,
 * bp_encode/bp_decode reproduce HuggingFace `tokenizers` output token for
 * token on valid UTF-8 input (see docs/FORMAT.md for the pipeline specification).
 *
 * Thread model: bp_vocab is immutable and shareable across threads after
 * bp_vocab_open; every thread uses its own bp_ctx. The library takes no
 * locks and has no mutable global state.
 *
 * License: MIT.
 */
#ifndef BYTEPAIR_H
#define BYTEPAIR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BYTEPAIR_VERSION_MAJOR 0
#define BYTEPAIR_VERSION_MINOR 1
#define BYTEPAIR_VERSION_PATCH 0

typedef struct bp_vocab bp_vocab; /* immutable, mmap-backed vocabulary */
typedef struct bp_ctx   bp_ctx;   /* per-thread scratch, cache, kernels */

/* Error codes: negative returns from the functions below. */
enum {
    BP_OK        = 0,
    BP_E_IO      = -1, /* open/stat/mmap failure (see errno for detail)   */
    BP_E_FORMAT  = -2, /* file is not a valid .bpv                        */
    BP_E_RANGE   = -3, /* token id out of range in bp_decode              */
    BP_E_NOMEM   = -4, /* allocation failure                              */
    BP_E_ARG     = -5  /* null/invalid argument                           */
};

/* Flags for bp_encode / bp_count / bp_decode. Default 0 is HF parity:
 * added tokens (special or not) appearing literally in the input are
 * matched and emitted as single ids, exactly as HuggingFace does.
 *
 * SECURITY: with the default, untrusted input containing a literal
 * control-token string (for example "<|im_start|>") becomes a control
 * token. Pass BP_RAW when tokenizing untrusted text. */
enum {
    BP_RAW          = 1u << 0, /* do not match added tokens              */
    BP_FORCE_SCALAR = 1u << 1, /* disable SIMD kernels (testing)         */
    BP_SKIP_SPECIAL = 1u << 2  /* bp_decode: omit special added tokens   */
};

/* Open a .bpv vocabulary file (memory-mapped, fully validated: structure,
 * section bounds and overlap, pair-table occupancy and content ranges,
 * byte mappings, added-token records). Returns NULL on failure; if err is
 * non-NULL it receives a BP_E_* code. The file is mapped for the handle's
 * lifetime: truncating it concurrently from outside the process is
 * undefined, as with any memory-mapped file. */
bp_vocab *bp_vocab_open(const char *path, int *err);
void      bp_vocab_close(bp_vocab *v);

uint32_t    bp_vocab_size(const bp_vocab *v);  /* max id + 1, added included */
const char *bp_vocab_name(const bp_vocab *v);  /* source name from meta      */

/* Per-thread context. Cache_log2 in [0,24] sets the pretoken cache to
 * 2^cache_log2 entries (0 disables it); pass BP_CTX_DEFAULT_CACHE for the
 * default (2^15). Any other value is rejected with NULL. */
#define BP_CTX_DEFAULT_CACHE (-1)
bp_ctx *bp_ctx_new(const bp_vocab *v, int cache_log2);
void    bp_ctx_free(bp_ctx *c);

/* Encode utf8[0..len) into token ids. Writes at most out_cap ids to out and
 * returns the total token count of the full input (which may exceed out_cap;
 * call with out_cap 0 to size). Returns a negative BP_E_* on error. Invalid
 * UTF-8 never crashes: malformed bytes are tokenized as opaque bytes, but
 * the exact-match guarantee covers valid UTF-8 only. */
int64_t bp_encode(bp_ctx *c, const char *utf8, size_t len,
                  uint32_t *out, size_t out_cap, uint32_t flags);

/* Token count only (same pipeline, no id output). */
int64_t bp_count(bp_ctx *c, const char *utf8, size_t len, uint32_t flags);

/* Decode ids[0..n) into UTF-8 bytes. Writes at most out_cap bytes and
 * returns the total byte count of the full output (call with out_cap 0 to
 * size). Any id >= bp_vocab_size fails with BP_E_RANGE. Output is not
 * NUL-terminated. */
int64_t bp_decode(bp_ctx *c, const uint32_t *ids, size_t n,
                  char *out, size_t out_cap, uint32_t flags);

const char *bp_strerror(int err);
const char *bp_version(void); /* "0.1.0" */

/* ---- vocabulary reachability census ----------------------------------
 * Classifies every token id: reachable with a verifying witness string
 * (added / self / context), provably impossible with a stated reason, or
 * unresolved within the bounded search. Every witness is independently
 * re-encoded through bp_encode before being reported; bp_census_run
 * returns nonzero only if such a re-verification fails, which would be a
 * defect in the census itself. json (optional) receives a full
 * machine-readable report; out receives the human summary. */
#include <stdio.h>
int bp_census_run(const bp_vocab *v, bp_ctx *c, FILE *json, int deep,
                  uint32_t limit, FILE *out);
int bp_census_witness(const bp_vocab *v, bp_ctx *c, uint32_t id, FILE *out);

#ifdef __cplusplus
}
#endif
#endif /* BYTEPAIR_H */
