#!/usr/bin/env python3
"""Differential test: bytepair vs HuggingFace tokenizers, token-for-token.

Run under a Python environment with `tokenizers` installed:

    python3 tests/differential.py --bytepair build/bytepair \
        --bpv build/qwen3.bpv --tokenizer tests/data/fetched/tokenizer.json \
        [--corpus path] [--quick]

Exit codes: 0 all equal, 1 mismatch found, 2 setup error.
On a bulk-document mismatch the failing document is bisected to a minimal
failing slice before reporting.
"""
import argparse
import random
import subprocess
import sys

def bp_encode(args, text: str, raw=False):
    cmd = [args.bytepair] + (["--raw"] if raw else []) + ["encode", args.bpv]
    r = subprocess.run(cmd, input=text.encode("utf-8", "surrogatepass"),
                       capture_output=True)
    if r.returncode != 0:
        print(f"FAIL bytepair exited {r.returncode}: {r.stderr.decode()!r} "
              f"on {text[:80]!r}")
        return None
    return [int(x) for x in r.stdout.split()]

def bp_decode(args, ids):
    r = subprocess.run([args.bytepair, "decode", args.bpv],
                       input=" ".join(map(str, ids)).encode(),
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None

def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))

class Runner:
    def __init__(self, args, hf):
        self.args = args
        self.hf = hf
        self.cases = 0
        self.failures = 0

    def check(self, text: str, label: str, bisect=False):
        self.cases += 1
        want = self.hf.encode(text, add_special_tokens=False).ids
        got = bp_encode(self.args, text)
        if got == want:
            return True
        self.failures += 1
        if bisect and len(text) > 64:
            text = self.bisect(text)
            want = self.hf.encode(text, add_special_tokens=False).ids
            got = bp_encode(self.args, text)
        i = first_diff(got or [], want)
        print(f"FAIL [{label}] {text[:120]!r}")
        print(f"  hf[{i}:{i+8}] = {want[i:i+8]}   bytepair[{i}:{i+8}] = "
              f"{(got or [])[i:i+8]}  (lens {len(want)}/{len(got or [])})")
        return False

    def bisect(self, text):
        """Shrink a failing text to a small failing slice (split on chars)."""
        def fails(t):
            return bp_encode(self.args, t) != \
                   self.hf.encode(t, add_special_tokens=False).ids
        cur = text
        step = 0
        while len(cur) > 64 and step < 60:
            step += 1
            half = len(cur) // 2
            for cand in (cur[:half], cur[half:],
                         cur[len(cur)//4:len(cur)*3//4]):
                if cand and fails(cand):
                    cur = cand
                    break
            else:
                break
        return cur

def gen_cases(rng, quick):
    """Adversarial + random case generators. Deterministic (seeded)."""
    ws = " \t\n\r   　"
    cases = []

    # worked consequences from docs/DESIGN.md (must stay in sync)
    cases += ["it's", "IT'S", "it’s", "'ſx", "it'ſ", "'lLx", "can'T do",
              "a   b", " 5", "  \n\n  \nx", "x  \n", "x  ", "\tword",
              "a<think>b", "<|endoftext|>", "x<|fim_prefix|>y",
              "a<|im_start|>b<|im_end|>c", "<|imposter|>", "<not_a_token>",
              "< |im_start|>", "<|im_start", "'", "''", "'''s", "’’'s",
              "don't can't won't I'll you're we've he'd she's it'S"]

    # NFC-sensitive
    cases += ["café", "café", "ȩ́", "ȩ́",
              "ḍ̇", "ḍ̇", "각",
              "각", "q̣̇s", "ཷ",
              "à̖̀b", "ﬁle", "ﬁle",
              "Å vs Å vs Å"]

    # whitespace torture
    for n in (1, 2, 3, 5, 17):
        for w in (" ", "\t", " ", "　", " \t"):
            cases += [w * n, w * n + "x", "x" + w * n, w * n + "5",
                      w * n + "\n" + w * n, w * n + "!"]

    # punctuation and newline runs
    cases += ["!!!", " !!!", "!!!\n\n", " ...\r\n", "#$%^&*()\n",
              "a-b", "a - b", "--—–", "…", "«»", "\r\n\r\n", "\r\r", "\n\r"]

    # digits
    cases += ["123", "12345678901234567890", "1a2b3", "١٢٣", "㊸", "Ⅻ",
              "3.14159", "1,000,000", "2²", "½"]

    # CJK / emoji / RTL
    cases += ["中文分词测试", "日本語のトークン化", "한국어 토큰화",
              "👨‍👩‍👧‍👦", "🇺🇳🇺🇸", "é🎉́",
              "مرحبا بالعالم", "שלום עולם", "हिन्दी", "ไทย"]

    # long-s and contraction edge combinations
    cases += ["'ſ", "ſ's", "'ſſ", "'S'ſ'll'LL", "'re're", "o'clock",
              "'veſ", "'llſ"]

    n_random = 200 if quick else 2000
    alphabets = [
        "abcdefghijklmnop  '\n\t.,!?0123456789",
        "abcĉ中日한'ſ  \n123!()",
        ws + "'ſ" + "asdSD",
        "".join(chr(c) for c in range(0x20, 0x2000, 7)),
        "".join(chr(c) for c in list(range(0x1F300, 0x1F600, 13)) +
                list(range(0x300, 0x370, 3)) + [0x27, 0x20, 0x0A]),
    ]
    for i in range(n_random):
        al = alphabets[i % len(alphabets)]
        ln = rng.choice([1, 2, 3, 7, 20, 80, 400])
        cases.append("".join(rng.choice(al) for _ in range(ln)))

    return cases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytepair", required=True)
    ap.add_argument("--bpv", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("setup error: the `tokenizers` package is required", file=sys.stderr)
        return 2
    hf = Tokenizer.from_file(args.tokenizer)
    r = Runner(args, hf)
    rng = random.Random(args.seed)

    for case in gen_cases(rng, args.quick):
        r.check(case, "case")

    # decode round-trip on a sample: bytepair decode must equal HF decode
    sample = ["it's café 中文 <think>x</think> 123  \n ok", "'ſx  \r\n!"]
    for s in sample:
        ids = hf.encode(s, add_special_tokens=False).ids
        got = bp_decode(args, ids)
        want = hf.decode(ids, skip_special_tokens=False).encode()
        r.cases += 1
        if got != want:
            r.failures += 1
            print(f"FAIL [decode] {s!r}: {got!r} != {want!r}")

    # full-codepoint sweep document (fixture must be non-trivial)
    if not args.quick:
        doc = "".join(chr(c) for c in range(0x20, 0x110000, 17)
                      if not 0xD800 <= c <= 0xDFFF)
        assert len(doc) > 50000, "codepoint fixture unexpectedly small"
        r.check(doc, "codepoint-sweep", bisect=True)
        doc2 = " ".join(chr(c) for c in range(0x20, 0x40000, 11)
                        if not 0xD800 <= c <= 0xDFFF)
        r.check(doc2, "codepoint-spaced", bisect=True)

    # bulk corpus
    if args.corpus:
        raw = open(args.corpus, "rb").read().decode("utf-8", "ignore")
        if len(raw) < 1_000_000:
            print("setup error: corpus fixture under 1 MB (vacuous)", file=sys.stderr)
            return 2
        want = hf.encode(raw, add_special_tokens=False).ids
        if len(want) < 100_000:
            print("setup error: corpus tokenizes under 100k tokens (vacuous)",
                  file=sys.stderr)
            return 2
        r.cases += 1
        got = bp_encode(args, raw)
        if got != want:
            r.failures += 1
            # bisect by paragraphs
            paras = raw.split("\n\n")
            for k, p in enumerate(paras):
                if bp_encode(args, p) != hf.encode(p, add_special_tokens=False).ids:
                    r.check(p, f"corpus-para-{k}", bisect=True)
                    break
            else:
                print("FAIL [corpus] whole-file mismatch but every paragraph "
                      "matches (boundary effect)")

    print(f"{r.cases} cases, {r.failures} failures")
    return 1 if r.failures else 0

if __name__ == "__main__":
    sys.exit(main())
