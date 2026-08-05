#!/usr/bin/env python3
"""The long-s contraction rule, made observable.

For the Qwen3 vocabulary the A1 long-s rule (U+017F matching 's after an
apostrophe) is invisible in token ids: no Qwen3 token bridges the long-s
bytes and an ASCII letter, so no merge can cross the boundary the rule
moves (verified by scanning every vocabulary entry, 2026-08-05). This toy
vocabulary contains exactly such a bridge ("ſx", id 257), so the split
decision changes the ids and the rule becomes testable:

    with A1 long-s:  "'ſx" -> ["'ſ", "x"]  -> [', ſ, x]  (3 ids)
    without:         "'ſx" -> ["'ſx"]      -> [', ſx]    (2 ids)

Run under a python with `tokenizers` (the oracle loads the same toy file):

    test_longs_toy.py <bytepair-binary> <toy.bpv> <toy.json>
"""
import subprocess
import sys

def main():
    if len(sys.argv) != 4:
        print("usage: test_longs_toy.py <bytepair> <toy.bpv> <toy.json>",
              file=sys.stderr)
        return 2
    binary, bpv, toyjson = sys.argv[1:4]
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(toyjson)

    # fixture guard: the bridge token must exist or this whole test is vacuous
    assert hf.get_vocab().get("Å¿x") == 257, "toy vocabulary lost its bridge token"

    # the toy also carries two added tokens with DIFFERING first bytes
    # ("<|end|>", "[PAD]"), exercising the general matcher path that the
    # Qwen3 vocabulary (all added tokens starting "<") never reaches
    cases = ("'ſx", "a'ſx", "'ſ", "ſx",
             "a<|end|>b", "x[PAD]y", "<|end|>[PAD]", "a<|endx[PADy")
    failures = 0
    for text in cases:
        want = hf.encode(text, add_special_tokens=False).ids
        r = subprocess.run([binary, "encode", bpv], input=text.encode(),
                           capture_output=True)
        got = [int(x) for x in r.stdout.split()] if r.returncode == 0 else None
        if got != want:
            print(f"FAIL {text!r}: hf {want} bytepair {got}")
            failures += 1
    print(f"long-s toy: {len(cases)} cases, {failures} failures")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
