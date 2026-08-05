/* bytepair CLI: encode, decode, count, bench.
 *
 * Exit codes (asserted by the test suite; keep stable):
 *   0 success
 *   1 usage error
 *   2 input file I/O error
 *   3 vocabulary open/validation error
 *   4 encode/decode error (message names the BP_E_* cause)
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/bytepair.h"

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "bytepair: out of memory\n"); exit(4); }
    return p;
}

/* Reads the whole stream; the buffer is always NUL-terminated one byte
 * past *len so text parsers (strtoull in decode) cannot run off the end. */
static uint8_t *read_all(FILE *f, size_t *len)
{
    size_t cap = 1 << 20, n = 0;
    uint8_t *buf = xmalloc(cap);
    for (;;) {
        if (n + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); if (!buf) exit(4); }
        size_t r = fread(buf + n, 1, cap - n - 1, f);
        n += r;
        if (r == 0) break;
    }
    if (ferror(f)) { free(buf); return NULL; }
    buf[n] = 0;
    *len = n;
    return buf;
}

/* The vocabulary name comes from the file, which may be hostile: never let
 * it drive the terminal. Everything below 0x20 and 0x7F prints escaped. */
static void print_sanitized(const char *s)
{
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p < 0x20 || *p == 0x7F)
            printf("\\x%02X", *p);
        else
            putchar(*p);
    }
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static bp_vocab *open_vocab(const char *path)
{
    int err;
    bp_vocab *v = bp_vocab_open(path, &err);
    if (!v) {
        fprintf(stderr, "bytepair: %s: %s\n", path, bp_strerror(err));
        exit(3);
    }
    return v;
}

static uint8_t *load_input(const char *path, size_t *len)
{
    FILE *f = stdin;
    if (path) {
        f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "bytepair: %s: %s\n", path, strerror(errno));
            exit(2);
        }
    }
    uint8_t *buf = read_all(f, len);
    if (path) fclose(f);
    if (!buf) {
        fprintf(stderr, "bytepair: read failed: %s\n", strerror(errno));
        exit(2);
    }
    return buf;
}

/* ---- bench ------------------------------------------------------------ */

typedef struct {
    const bp_vocab *v;
    const uint8_t *text;
    const size_t *doc_off; /* ndocs+1 */
    size_t ndocs;
    size_t first, step;
    uint32_t flags;
    int64_t tokens;
} bench_job;

static void *bench_worker(void *arg)
{
    bench_job *j = arg;
    bp_ctx *c = bp_ctx_new(j->v, BP_CTX_DEFAULT_CACHE);
    if (!c) return (void *)1;
    for (size_t d = j->first; d < j->ndocs; d += j->step) {
        int64_t n = bp_count(c, (const char *)(j->text + j->doc_off[d]),
                             j->doc_off[d + 1] - j->doc_off[d], j->flags);
        if (n < 0) { bp_ctx_free(c); return (void *)1; }
        j->tokens += n;
    }
    bp_ctx_free(c);
    return NULL;
}

static int cmd_bench(bp_vocab *v, const char *file, int threads,
                     uint32_t flags)
{
    size_t len;
    uint8_t *text = load_input(file, &len);
    if (len == 0) {
        fprintf(stderr, "bytepair: bench: empty input\n");
        free(text);
        return 2;
    }

    /* split into ~8 KB documents on line boundaries, like the recon bench */
    size_t cap = len / 4096 + 2, ndocs = 0;
    size_t *off = xmalloc(cap * sizeof(size_t));
    off[0] = 0;
    size_t last = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == '\n' && i - last >= 8192) {
            off[++ndocs] = i + 1;
            last = i + 1;
        }
    }
    if (last < len) off[++ndocs] = len;

    double best = 0;
    int64_t tokens = 0;
    for (int round = 0; round < 3; round++) {
        pthread_t tid[256];
        bench_job job[256];
        if (threads > 256) threads = 256;
        double t0 = now_s();
        for (int t = 0; t < threads; t++) {
            job[t] = (bench_job){ v, text, off, ndocs, (size_t)t,
                                  (size_t)threads, flags, 0 };
            if (pthread_create(&tid[t], NULL, bench_worker, &job[t]))
                return 4;
        }
        int64_t sum = 0;
        int failed = 0;
        for (int t = 0; t < threads; t++) {
            void *ret;
            pthread_join(tid[t], &ret);
            if (ret) failed = 1;
            sum += job[t].tokens;
        }
        if (failed) { fprintf(stderr, "bytepair: bench worker failed\n"); return 4; }
        double dt = now_s() - t0;
        double mbs = (double)len / 1e6 / dt;
        if (mbs > best) { best = mbs; tokens = sum; }
    }
    printf("%zu bytes, %zu docs, %d thread(s): %.2f MB/s, %.2f Mtok/s "
           "(%lld tokens, best of 3)\n",
           len, ndocs, threads, best,
           (double)tokens / 1e6 * best * 1e6 / (double)len, (long long)tokens);
    free(text);
    free(off);
    return 0;
}

/* ----------------------------------------------------------------------- */

static uint32_t parse_flags(int *argc, char **argv)
{
    uint32_t flags = 0;
    int w = 1;
    for (int i = 1; i < *argc; i++) {
        if (!strcmp(argv[i], "--raw")) flags |= BP_RAW;
        else if (!strcmp(argv[i], "--scalar")) flags |= BP_FORCE_SCALAR;
        else if (!strcmp(argv[i], "--skip-special")) flags |= BP_SKIP_SPECIAL;
        else argv[w++] = argv[i];
    }
    *argc = w;
    return flags;
}

static int usage(void)
{
    fprintf(stderr,
        "usage: bytepair [--raw] [--scalar] [--skip-special] <command> ...\n"
        "  encode <vocab.bpv> [file]         ids to stdout, one per line\n"
        "  count  <vocab.bpv> [file]         token count to stdout\n"
        "  decode <vocab.bpv> [file]         ids (whitespace-separated) to text\n"
        "  bench  <vocab.bpv> <file> [N]     throughput with N threads (default 1)\n"
        "  info   <vocab.bpv>                vocabulary summary\n");
    return 1;
}

int main(int argc, char **argv)
{
    uint32_t flags = parse_flags(&argc, argv);
    if (argc < 3) return usage();
    const char *cmd = argv[1];
    if (strcmp(cmd, "info") && strcmp(cmd, "encode") && strcmp(cmd, "count") &&
        strcmp(cmd, "decode") && strcmp(cmd, "bench"))
        return usage(); /* before any file is touched: exit 1, always */
    double t_open = now_s();
    bp_vocab *v = open_vocab(argv[2]);
    t_open = now_s() - t_open;
    const char *file = argc > 3 ? argv[3] : NULL;
    int rc = 0;

    if (!strcmp(cmd, "info")) {
        printf("name: ");
        print_sanitized(bp_vocab_name(v));
        printf("\nvocab: %u\nversion: %s\nopen: %.3f ms\n",
               bp_vocab_size(v), bp_version(), t_open * 1e3);
    } else if (!strcmp(cmd, "encode") || !strcmp(cmd, "count")) {
        size_t len;
        uint8_t *text = load_input(file, &len);
        bp_ctx *c = bp_ctx_new(v, BP_CTX_DEFAULT_CACHE);
        if (!c) { rc = 4; goto done_text; }
        int64_t n = bp_count(c, (const char *)text, len, flags);
        if (n < 0) {
            fprintf(stderr, "bytepair: %s\n", bp_strerror((int)n));
            bp_ctx_free(c);
            rc = 4;
            goto done_text;
        }
        if (!strcmp(cmd, "count")) {
            printf("%lld\n", (long long)n);
        } else {
            uint32_t *ids = xmalloc((size_t)n * sizeof(uint32_t));
            int64_t n2 = bp_encode(c, (const char *)text, len, ids,
                                   (size_t)n, flags);
            if (n2 != n) {
                fprintf(stderr, "bytepair: %s\n",
                        bp_strerror(n2 < 0 ? (int)n2 : BP_E_ARG));
                free(ids);
                bp_ctx_free(c);
                rc = 4;
                goto done_text;
            }
            for (int64_t i = 0; i < n; i++)
                printf("%u\n", ids[i]);
            free(ids);
        }
        bp_ctx_free(c);
done_text:
        free(text);
    } else if (!strcmp(cmd, "decode")) {
        size_t len;
        uint8_t *text = load_input(file, &len);
        size_t cap = 1024, n = 0;
        uint32_t *ids = xmalloc(cap * sizeof(uint32_t));
        char *s = (char *)text, *end = s + len;
        while (s < end) {
            char *next;
            unsigned long long val = strtoull(s, &next, 10);
            if (next == s) { s++; continue; }
            if (val > 0xFFFFFFFFull) {
                fprintf(stderr, "bytepair: %s\n", bp_strerror(BP_E_RANGE));
                free(ids);
                free(text);
                bp_vocab_close(v);
                return 4;
            }
            if (n == cap) {
                cap *= 2;
                ids = realloc(ids, cap * sizeof(uint32_t));
                if (!ids) exit(4);
            }
            ids[n++] = (uint32_t)val;
            s = next;
        }
        bp_ctx *c = bp_ctx_new(v, 0);
        if (!c) { rc = 4; goto done_dec; }
        int64_t out_len = bp_decode(c, ids, n, NULL, 0, flags);
        if (out_len < 0) {
            fprintf(stderr, "bytepair: %s\n", bp_strerror((int)out_len));
            bp_ctx_free(c);
            rc = 4;
            goto done_dec;
        }
        char *out = xmalloc((size_t)out_len);
        bp_decode(c, ids, n, out, (size_t)out_len, flags);
        fwrite(out, 1, (size_t)out_len, stdout);
        free(out);
        bp_ctx_free(c);
done_dec:
        free(ids);
        free(text);
    } else if (!strcmp(cmd, "bench")) {
        if (!file) { bp_vocab_close(v); return usage(); }
        int threads = argc > 4 ? atoi(argv[4]) : 1;
        if (threads < 1) threads = 1;
        rc = cmd_bench(v, file, threads, flags);
    } else {
        bp_vocab_close(v);
        return usage();
    }
    bp_vocab_close(v);
    return rc;
}
