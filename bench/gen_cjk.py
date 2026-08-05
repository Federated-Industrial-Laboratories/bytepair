#!/usr/bin/env python3
"""Deterministic synthetic CJK corpus (seeded; byte-identical on every run).

Random CJK defeats every pretoken cache, so this corpus is the worst-case
probe for cache-dependent tokenizers. It is synthetic and Zipf-free by
design; real Chinese text runs considerably faster on all tools.
"""
import random
import sys

def main():
    if len(sys.argv) != 2:
        print("usage: gen_cjk.py <output>", file=sys.stderr)
        return 2
    rng = random.Random(7)
    common = [chr(c) for c in range(0x4E00, 0x9FA5)]
    out = []
    for _ in range(20000):
        out.append("".join(rng.choice(common)
                           for _ in range(rng.randint(2, 12))))
        out.append(rng.choice(["，", "。", " ", "、", "\n",
                               "："]))
    data = "".join(out).encode("utf-8")
    open(sys.argv[1], "wb").write(data)
    print(f"cjk corpus: {len(data)} bytes")
    return 0

if __name__ == "__main__":
    sys.exit(main())
