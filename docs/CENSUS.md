# Vocabulary reachability census

`bytepair census` decides, for every token id in a `.bpv` vocabulary, whether any valid
UTF-8 input can make the encoder emit it, and produces evidence either way. It is exact
where exactness is achievable and explicit about its bounds where it is not.

```sh
bytepair census build/qwen3.bpv census.json --deep    # full report, ~0.5 s
bytepair witness build/qwen3.bpv 123828               # one token, with proof of work
```

## Verdicts

| verdict | meaning | evidence |
| --- | --- | --- |
| `added` | matched literally by the added-token pass | the token's own string |
| `self` | the token's bytes alone encode to it | a witness string |
| `context` | a constructed context produces it | a witness string |
| `impossible` | no input can produce it | a stated reason |
| `unresolved` | no witness within the search bound, no proof | the bound searched |

Every witness is re-encoded through the public `bp_encode` before it is reported;
`census` exits nonzero if any witness fails that re-verification. Witnesses are plain
strings, so any independent implementation, including the HuggingFace reference, can
confirm every reachable verdict by encoding them.

## The boundary lemma

Within one pretoken, BPE merge ranks are global: the presence of neighboring bytes never
changes which merges are available inside a byte range or their relative order. A
neighboring merge either crosses the range boundary, permanently absorbing boundary
bytes into a token that spans outside the range (after which the range can never again
compose exactly the token whose bytes it held), or it does not cross, in which case the
range's internal derivation proceeds exactly as it would in isolation.

Consequence: if pure BPE over `bytes(t)` does not yield `[t]`, then no input whatsoever
can produce `t`. Merge-order failure is therefore a proof of unreachability, not merely
a failed test. Every remaining reachability question is a question about the
pretokenizer: which byte strings can sit inside a single pretoken of the profile-1
language, given that inputs are NFC-normalized first.

## Impossibility reasons

- `merge-order`: pure BPE over the token's bytes yields a different partition (the
  boundary lemma makes this final).
- `invalid-utf8`: the bytes cannot occur inside any well-formed UTF-8 string (for
  example the bytes 0xC0, 0xC1, 0xF5 to 0xFF).
- `pretoken-shape`: no pretoken of the profile-1 language can contain the bytes. The
  prover enumerates every UTF-8 alignment of the token (complete characters plus
  partial characters at either end, whose possible completions are enumerated exactly)
  and checks each against a relaxed superset of the pretoken language. Letters never
  share a pretoken with digits, digits never pair, interior spaces cannot follow
  content, and a letter is never followed by a combining mark inside one pretoken.
- `contraction`: the token requires an apostrophe at pretoken start followed by a
  contraction suffix and further letters, a reading that alternative A1 always claims
  first.
- `nfc-excluded`: every alignment that fits the pretoken language requires a codepoint
  whose NFC quick-check property is No. Such codepoints never occur in NFC output, and
  the pipeline normalizes before scanning, so no post-normalization text can contain
  them. CJK compatibility ideographs are the dominant case.

All proofs are sound by construction: where enumeration is loosened, it is loosened
toward "possible", so a token is never called impossible without the property actually
holding.

## Witness search

Reachable-in-principle tokens that fail self-encoding are witnessed constructively:
affix pools of representative characters, constructed glue that completes partial
characters at either end (a token holding continuation bytes is embedded into a real
character), exhaustive single-character completions, and a rank-steal stage that
follows a completion byte with a second character whose pair outranks the absorbing
merge, freeing the fragment to surface. The `--deep` flag enables the exhaustive
stages.

## Result for Qwen3 (measured 2026-08-06, reproducible from the tree)

| verdict | tokens | share |
| --- | --- | --- |
| reachable, witnessed | 149,734 | 98.72% |
| of which: added / self / context | 26 / 148,286 / 1,422 | |
| impossible | 1,932 | 1.27% |
| of which: pretoken-shape | 1,679 | |
| of which: nfc-excluded | 240 | |
| of which: invalid-utf8 | 13 | |
| of which: merge-order | 0 | |
| unresolved | 3 | 0.002% |

Every one of the 149,734 witnesses was additionally re-encoded through HuggingFace
`tokenizers` 0.22.2, with zero mismatches, and every decodable `nfc-excluded` token was
confirmed rewritten by the reference's own normalizer. The census itself runs in about
half a second on a 2016 Xeon.

Notable content of the impossible set:

- The `pretoken-shape` class is dominated by Thai: whole-syllable tokens such as
  `ที่`, `เป็น`, and `ได้` combine letters with combining vowel and tone marks, and the
  split regex always separates a letter run from a following mark. The vocabulary
  contains on the order of 1,500 such Thai entries that the shipped pretokenizer can
  never assemble. Fullwidth digit pairs (`１０`, `２０`) and variation-selector
  sequences account for most of the rest.
- The `nfc-excluded` class is dominated by CJK compatibility ideographs, whose
  codepoints never survive NFC.
- Zero `merge-order` failures: the native merge table is internally consistent, a
  property that published work on tokenizer adaptation shows is easily lost when
  vocabularies are extended.

At 4,096 embedding dimensions in bf16, 1,932 dead rows amount to roughly 15.8 MB of
parameters per embedding matrix that no input can ever activate.

The three unresolved tokens are id 127 (the bare lead byte 0xC3, all of whose 64
completions are absorbed by Latin-1 merges), id 125388 (a hiragana fragment), and id
139793 (a Thai fragment); each survived exhaustive one- and two-character completion
searches without either a witness or a proof.

## Related work

Isolated-token reachability tests exist in prior work: Sander Land's self-encoding
analysis of o200k_base ("Unreachable tokens in GPT-4o", Token Contributions) finds
twelve unreachable tokens by decoding and re-encoding each token alone, and the
tokenizer-adaptation literature audits inserted tokens through the merge graph
(strict merge reachability, arXiv 2608.00582; the merge ordering problem, arXiv
2512.03989). This census differs in producing a three-way exact partition for a stock
vocabulary: witnesses for the reachable (including tokens only reachable in context,
which the isolated test misclassifies), sound impossibility proofs with reasons, and
an explicit bound on the remainder. The boundary lemma is what turns the isolated
merge test from a heuristic into a proof. "Fishing for Magikarp" (Land and Bartolo,
EMNLP 2024) addresses the adjacent problem of tokens that are reachable but
under-trained.

## Scope notes

Reachability is defined for valid UTF-8 input under default encoding flags (added-token
matching on, matching the reference's behavior). The census applies to any `.bpv`
vocabulary; the pretoken-language prover is specific to profile 1, which is the only
profile the format currently loads.
