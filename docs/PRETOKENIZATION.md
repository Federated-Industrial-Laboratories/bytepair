# Pretokenization carries linguistic priors, and it does not carry them equally

A position, grounded in the measurements in this repository. It argues one claim: the
split pattern used by the cl100k family of tokenizers, including Qwen3's, applies
linguistic structure unevenly across writing systems, and the unevenness is an accident
of inheritance with measurable costs. Everything quantitative below is reproducible
from this tree (`bytepair census`, `docs/CENSUS.md`); everything about training
consequences is explicitly out of scope, because it would require experiments this
project has not run.

## The pattern, and what it assumes

Qwen3 splits text with a pattern in the cl100k lineage:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

It differs from cl100k_base in details (single digits rather than `\p{N}{1,3}`, an
optional leading space on the punctuation run), but the structure is the same, and the
structure encodes assumptions:

- **English gets grammar.** The first alternative carves out seven English clitic
  contractions, case-insensitively. No other language's morphology appears anywhere.
- **Space-delimited scripts get word boundaries.** `[^\r\n\p{L}\p{N}]?\p{L}+` ensures
  merges never cross a word boundary, a strong and useful prior for Latin, Cyrillic,
  Greek, Hangul, and similar scripts.
- **Chinese, Japanese, and undelimited scripts get nothing.** Every han character is a
  letter, so entire sentences arrive at BPE as one pretoken and the merge table must
  learn all segmentation statistically. Whether that is good or bad is genuinely open;
  what matters here is that it is a different philosophy than the one applied to
  English, chosen implicitly.
- **Scripts that build syllables from letters plus combining marks get an active
  harm.** `\p{L}` excludes the mark categories. A Thai vowel or tone mark, an Arabic
  diacritic, a Hebrew point: each terminates the letter run and lands in the
  punctuation alternative. The scanner therefore cuts inside every marked syllable,
  and no merge can ever cross that cut.

## What the census measures

The reachability census (`docs/CENSUS.md`) makes the third asymmetry concrete for
Qwen3. Classifying every vocabulary token by dominant script and intersecting with the
census verdicts:

| script | tokens in vocabulary | provably unreachable | share |
| --- | --- | --- | --- |
| Thai | 2,570 | 1,578 | 61.4% |
| CJK compatibility ideographs | 195 | 194 | 99.5% |
| Arabic | 3,642 | 75 | 2.1% |
| Hebrew | 3,160 | 3 | 0.1% |
| CJK unified | 25,475 | 1 | 0.0% |
| Hangul | 3,473 | 0 | 0.0% |
| Latin | 95,704 | 14 | 0.0% |

Three of every five Thai tokens in the vocabulary cannot be produced by any input. The
dead entries are whole syllables: `ที่`, `เป็น`, `ได้`, syllable tokens the vocabulary
plainly considered worth their embedding rows. Encoding each dead token's own text
through the tokenizer, the stored single token is delivered as 3.13 tokens on average
(minimum two, 185 of them five or more). Thai text still tokenizes and the model still
processes it; the cost is paid in sequence length, in 1,578 embedding rows that can
never activate, and in whatever representation quality difference follows from
syllables arriving in fragments. That last consequence is training-dependent and this
repository makes no claim about its size.

The Arabic dead tokens are the diacritized forms (`مْ`, `هُ`, `لّ`); the harm is
smaller only because most Arabic text is written unvocalized. Hangul escapes entirely,
by an accident of a different kind: NFC composes jamo into precomposed syllable blocks
that are single letters before the scanner ever runs. The CJK compatibility ideographs
die for an unrelated inherited reason, normalization, documented in the census.

How did unreachable syllables enter the vocabulary at all? This repository cannot see
the training pipeline, so the honest statement is: the shipped vocabulary and the
shipped pretokenizer disagree about which strings can be pretokens, and the census can
only date the disagreement to before release. Whatever produced the vocabulary treated
letter-plus-mark sequences as units; whatever ships in the tokenizer does not.

## The field has already located the repair

OpenAI's o200k_base pattern, the successor to cl100k_base, revises exactly the two
asymmetries that are repairable inside a regex. Its word alternative reads:

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
```

`\p{M}` appears in both cluster groups: marks travel with their letters, and a Thai
syllable survives pretokenization whole. The contractions remain, but as an optional
suffix bound to a preceding word rather than a free-floating alternative. Sander Land's
self-encoding analysis of o200k finds twelve unreachable tokens in a 200,000-entry
vocabulary, against the 1,932 this census proves for Qwen3's 151,669: the mark repair
works at scale, and the residue is edge cases rather than a writing system.

## The position

1. **Uniformity is the principle.** A pretokenizer should apply one philosophy across
   scripts: if word structure is worth enforcing for English, syllable structure is
   worth preserving for Thai. The minimal concrete form is o200k's: include `\p{M}` in
   letter clusters, so the unit of pretokenization approximates the grapheme cluster
   everywhere rather than the ASCII word.
2. **A vocabulary should be validated against its own pretokenizer before release.**
   The disagreement measured here is mechanically detectable in half a second; a
   release gate of zero unreachable entries (or documented exceptions) costs nothing
   and would have surfaced the Thai situation immediately. `bytepair census` exists
   and is generic over the format.
3. **Behavior should ship pinned.** The Unicode version mix, the fold set, and the
   class table documented in this repository were all discoverable only by probing.
   A behavioral fingerprint published with a model (`tools/fingerprint.py`) turns
   silent drift into a named diff.
4. **What this position does not claim.** No statement is made here about downstream
   Thai task quality, because that is a training question. The measurable facts are
   the dead vocabulary share, the fragment delivery of stored syllables, and the
   existence of a deployed repair in a successor pattern; the position is that these
   facts are sufficient to motivate the uniform treatment on engineering grounds
   alone.

## Sources

- This repository: `docs/CENSUS.md` (method, proofs, per-token results), reproducible
  via `bytepair census`.
- OpenAI tiktoken, `tiktoken_ext/openai_public.py`: the cl100k_base and o200k_base
  pattern strings quoted above.
- Sander Land, "Unreachable tokens in GPT-4o", Token Contributions, for the o200k
  self-encoding analysis.
- Land and Bartolo, "Fishing for Magikarp" (EMNLP 2024), for the adjacent problem of
  under-trained tokens.
