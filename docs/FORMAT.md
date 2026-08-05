# bytepair technical reference

This document specifies the `.bpv` vocabulary format, the encoding pipeline, and the
validation contract. It is the maintenance reference; the README covers usage. If code
and this document disagree, the disagreement is a defect.

## The encoding pipeline

```
added-token scan (raw bytes, longest match; disabled by BP_RAW)
  -> per segment: NFC normalize -> split scanner (profile 1) -> byte-level BPE
decode: concatenate token byte spans (BP_SKIP_SPECIAL omits special added tokens)
```

### Added tokens

The reference implementation matches every added token in the input unconditionally,
special or not, before normalization runs on the remaining segments. bytepair does the
same by default. Matching is leftmost, longest-first at each position, on raw bytes.
When every added token begins with the same byte the scan uses `memchr` on that byte;
otherwise a 256-bit first-byte map drives a per-byte scan. `BP_RAW` disables the pass
entirely; the security consequence is documented in the README.

Added-token records with `single_word`, `lstrip`, `rstrip`, or `normalized` set are not
implemented; the converter refuses such vocabularies rather than approximating them.

### NFC normalization

Input segments are normalized to NFC before scanning. Two fast paths skip the work when
it is provably unnecessary: pure-ASCII segments, and segments that pass the NFC
quick-check with in-order combining classes. The full path is canonical decomposition
(Hangul algorithmically), canonical reordering (stable), and canonical composition per
UAX #15.

The Unicode data carries two deliberate version pins, both derived by probing the
reference implementation across every codepoint and both gated in the table generator:

- character classes for the split regex behave as **Unicode 16.0.0**;
- NFC data (combining classes, decompositions, compositions, quick-check) behaves as
  **Unicode 9.0.0**, which is what the reference engine ships.

Codepoints that gained a combining class or a canonical decomposition after 9.0.0 do
not normalize, because they do not normalize in the reference. The generator refuses to
emit tables unless the recorded probe artifacts match the UCD files it reads; the
conformance test (`tests/nfc_conformance.c`) runs the shipped C implementation against
the pinned Unicode 9.0.0 NormalizationTest file.

### The split scanner (profile 1)

The pretokenizer implements this regex under leftmost-first alternation, as a hand
compiled scanner (no regex engine):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)          A1 contraction
[^\r\n\p{L}\p{N}]?\p{L}+              A2 word (optional one-char prefix)
\p{N}                                 A3 single number
 ?[^\s\p{L}\p{N}]+[\r\n]*             A4 punctuation run
\s*[\r\n]+                            A5 whitespace through the last newline
\s+(?!\S)                             A6 trailing whitespace (give-back)
\s+                                   A7 whitespace fallback
```

Every character matches one alternative, so the spans partition the input. Derived
behavior the tests pin:

- A1 is case-insensitive under Unicode simple case folding. The participating
  characters are exactly `[a-zA-Z]` per position plus U+017F (long s), which folds to
  `s` at the single-letter position. Established by sweeping every letter codepoint
  through every contraction position against the reference.
- A5 matches a whitespace run through the **last** CR or LF it contains.
- A6 matches a whitespace run reaching end of input whole; a run followed by a
  non-space character loses its final codepoint to the next match.
- `"a   b"` splits `["a", "  ", " b"]`; `" 5"` splits `[" ", "5"]`; `"\tword"` is one
  span; `"it's"` splits `["it", "'s"]`.

The scalar scanner is the reference; the AVX2 kernels (selected at runtime only after
a full CPUID/XGETBV check) must equal it byte for byte, and both paths run the full
differential suite.

### BPE

Within each pretoken, bytes map to initial ids through `byte_to_id`, then the adjacent
pair with the lowest merge rank is merged repeatedly; rank ties break leftmost. Short
pretokens use a flat-array rescan; longer ones a linked list with a lazy min-heap.
A per-context cache keyed by pretoken bytes (verified by full hash, length, and
`memcmp` on every hit) accelerates repeated pretokens.

## The .bpv format

Little-endian, memory-mapped, one file. Fixed fields occupy bytes 0..135 in the order
below with no padding; bytes 136..191 are zero; every section offset is an absolute,
64-byte-aligned file offset, the first at 192.

```
u32 magic "BPV1" (0x31565042) | u32 format_version = 1
u32 flags (bit0: nfc)         | u32 profile_id
u32 vocab_count               | u32 added_count
u32 pair_slots_log2           | u32 reserved = 0
6 x (u64 offset, u64 size): token_offsets, token_blob, pair_table, byte_to_id,
                            added, meta
u64 source_hash (FNV-1a-64 of the source tokenizer.json bytes)
```

Sections:

- `token_offsets`: `u32[vocab_count + 1]` into `token_blob`; monotonic. Ids with a
  zero-length span are unused ids of a sparse vocabulary: encode never produces them
  and decode refuses them as out of range.
- `token_blob`: concatenated token byte strings, added tokens at their real ids.
- `pair_table`: open-addressing hash, one slot = `u64 key, u64 value`, where key is
  `(left << 32) | right` and value is `(rank << 32) | merged_id`. Power-of-two slots,
  load factor at most 0.65, linear probing with step 1. Empty slots are 16 bytes of
  0xFF; lookups test the key word alone. The slot function is murmur3 fmix64
  (`h ^= h >> 33; h *= 0xff51afd7ed558ccd; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33`), slot = `h & (slots - 1)`.
- `byte_to_id`: `u32[256]`, the id of each single-byte token.
- `added`: `added_count` records of `u32 id, u32 offset, u32 len, u32 flags`
  (bit 0: special), sorted by id.
- `meta`: length-prefixed UTF-8 strings: source name, converter version, the profile
  regex for human audit.

`profile_id` 1 is the scanner above and the only profile this version loads; other
values, including 0, are reserved and refused.

## Validation contract

The loader validates the entire image before returning a handle; any failure is a
clean `BP_E_FORMAT` with no partial open. A `.bpv` is an untrusted input: the checks
exist so that no file, however crafted, can make the library loop, crash, read out of
bounds, or emit an id at or above `vocab_count`.

- magic, version, profile id; `vocab_count` in (0, 2^21]; `added_count` at most 64;
  `pair_slots_log2` at most 32
- every section 64-byte aligned, inside the file, and non-overlapping; element sizes
  divide section sizes; `token_offsets` and `byte_to_id` and `added` sizes exactly as
  implied by the counts
- `token_offsets` monotonic and bounded by the blob
- pair table: every occupied slot's left, right, and merged ids below `vocab_count`;
  occupancy at most 65 percent, so a linear probe always reaches an empty slot and
  lookups terminate
- `byte_to_id`: every id in range, resolving to a one-byte token whose blob byte equals
  the index
- added records: ids in range, sorted, unique; each record's offset and length exactly
  equal to the token table's span for that id, so a record can never present a prefix
  or an alias of its token

Semantic coherence beyond these bounds (for example, whether the merge table encodes a
meaningful BPE) is the vocabulary author's responsibility; the contract here is memory
safety, termination, and id-range safety.

## Error and threading contracts

See `include/bytepair.h`. All errors are negative `BP_E_*` enumerators; `bp_vocab` is
immutable and shareable across threads after open; each thread uses its own `bp_ctx`;
the library takes no locks and has no mutable global state.
