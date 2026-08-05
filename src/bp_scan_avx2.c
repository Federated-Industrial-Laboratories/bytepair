/* AVX2 run kernels. This translation unit is compiled with -mavx2 -mbmi2;
 * its functions are called only after CPUID reports both (bp_vocab.use_avx2)
 * and the caller has not forced scalar. Each kernel consumes a run of ASCII
 * bytes of its kind and stops at the first byte that is not one - including
 * any non-ASCII byte, which the scalar caller then decodes properly. The
 * kernels must equal the scalar paths byte for byte; the differential suite
 * runs both.
 */
#include "bp_internal.h"

#if defined(__x86_64__)
#include <immintrin.h>

/* mask of bytes equal within [lo..hi] treating bytes as unsigned */
static inline __m256i in_range(__m256i v, uint8_t lo, uint8_t span)
{
    __m256i x = _mm256_sub_epi8(v, _mm256_set1_epi8((char)lo));
    return _mm256_cmpeq_epi8(_mm256_min_epu8(x, _mm256_set1_epi8((char)span)),
                             x);
}

size_t bp_ascii_letter_run_avx2(const uint8_t *t, size_t i, size_t len)
{
    while (i + 32 <= len) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(t + i));
        __m256i lower = _mm256_or_si256(v, _mm256_set1_epi8(0x20));
        uint32_t m = (uint32_t)_mm256_movemask_epi8(in_range(lower, 'a', 25));
        if (m != 0xFFFFFFFFu) return i + (size_t)__builtin_ctz(~m);
        i += 32;
    }
    while (i < len) {
        uint8_t b = t[i] | 0x20;
        if (b < 'a' || b > 'z') break;
        i++;
    }
    return i;
}

size_t bp_ascii_symbol_run_avx2(const uint8_t *t, size_t i, size_t len)
{
    const __m256i sp = _mm256_set1_epi8(' ');
    while (i + 32 <= len) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(t + i));
        __m256i lower = _mm256_or_si256(v, _mm256_set1_epi8(0x20));
        __m256i letter = in_range(lower, 'a', 25);
        __m256i digit = in_range(v, '0', 9);
        /* \t \n \v \f \r are 0x09..0x0D, plus space */
        __m256i ws = _mm256_or_si256(in_range(v, 0x09, 4),
                                     _mm256_cmpeq_epi8(v, sp));
        uint32_t ascii = ~(uint32_t)_mm256_movemask_epi8(v);
        uint32_t bad = (uint32_t)_mm256_movemask_epi8(
            _mm256_or_si256(_mm256_or_si256(letter, digit), ws));
        uint32_t ok = ascii & ~bad;
        if (ok != 0xFFFFFFFFu) return i + (size_t)__builtin_ctz(~ok);
        i += 32;
    }
    while (i < len) {
        uint8_t b = t[i];
        if (b >= 0x80) break;
        uint8_t lower = b | 0x20;
        if (lower >= 'a' && lower <= 'z') break;
        if (b >= '0' && b <= '9') break;
        if (b == ' ' || (b >= 0x09 && b <= 0x0D)) break;
        i++;
    }
    return i;
}

#else /* non-x86-64: never selected at runtime; keep the linker happy */

size_t bp_ascii_letter_run_avx2(const uint8_t *t, size_t i, size_t len)
{
    (void)t; (void)len;
    return i;
}

size_t bp_ascii_symbol_run_avx2(const uint8_t *t, size_t i, size_t len)
{
    (void)t; (void)len;
    return i;
}

#endif
