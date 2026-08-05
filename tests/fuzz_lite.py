#!/usr/bin/env python3
"""Seeded robustness fuzz: encode/decode must never crash, whatever the
input. Valid UTF-8, invalid UTF-8, truncated multibyte sequences, and
hostile decode ids all go through the CLI; every trial asserts a clean,
documented exit. Run against a sanitizer build for the full effect
(make SANITIZE=1). Python stdlib only; deterministic.

    fuzz_lite.py <bytepair-binary> <vocab.bpv> [trials]
"""
import random
import subprocess
import sys

def main():
    if len(sys.argv) not in (3, 4):
        print("usage: fuzz_lite.py <bytepair> <vocab.bpv> [trials]",
              file=sys.stderr)
        return 2
    binary, bpv = sys.argv[1], sys.argv[2]
    trials = int(sys.argv[3]) if len(sys.argv) == 4 else 400
    rng = random.Random(99)
    fails = 0

    for trial in range(trials):
        kind = trial % 4
        n = rng.choice([0, 1, 2, 7, 100, 4096])
        if kind == 0:
            data = bytes(rng.randrange(256) for _ in range(n))
        elif kind == 1:
            data = "".join(chr(rng.randrange(0x20, 0x110000))
                           for _ in range(n)).encode("utf-8", "ignore")
        elif kind == 2:
            base = "aé中'ſ <think>\n\t\U0001d11e".encode()
            cut = base * max(1, n // len(base))
            data = cut[:rng.randrange(len(cut) + 1)]
        else:
            data = bytes([0xC5, 0xFF, 0x80] * (n or 1))
        r = subprocess.run([binary, "encode", bpv], input=data,
                           capture_output=True)
        if r.returncode != 0:
            print(f"FAIL encode rc {r.returncode} on {data[:40]!r}")
            fails += 1
            continue
        r2 = subprocess.run([binary, "decode", bpv], input=r.stdout,
                            capture_output=True)
        if r2.returncode != 0:
            print(f"FAIL decode rc {r2.returncode}")
            fails += 1

    # hostile decode ids: out of range and beyond 32 bits, exact exit 4
    for bad in (b"151669", b"4294967295", b"4294967296", b"99999999999",
                b"0 1 2 151668 151669"):
        r = subprocess.run([binary, "decode", bpv], input=bad,
                           capture_output=True)
        if r.returncode != 4:
            print(f"FAIL hostile decode {bad!r}: expected exit 4, "
                  f"got {r.returncode}")
            fails += 1

    print(f"fuzz-lite: {trials} trials + 5 hostile, {fails} failures")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
