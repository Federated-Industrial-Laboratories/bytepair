#!/usr/bin/env python3
"""Census verdicts on the toy vocabulary, where the truth is known by
construction. The toy carries one designed probe per proof class:

  bytes C0, C1, F5..FF     -> impossible, invalid-utf8 (13 tokens)
  "12" (two digits)        -> impossible, pretoken-shape
  "abc" (bc outranks ab)   -> impossible, merge-order
  continuation-byte tokens -> context, witnessed by character gluing
  everything else          -> self or added, zero unresolved

Python stdlib only:  test_census.py <bytepair-binary> <toy.bpv>
"""
import json
import subprocess
import sys
import tempfile

DEAD_BYTES = [0xC0, 0xC1] + list(range(0xF5, 0x100))

def main():
    if len(sys.argv) != 3:
        print("usage: test_census.py <bytepair> <toy.bpv>", file=sys.stderr)
        return 2
    binary, bpv = sys.argv[1], sys.argv[2]
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        r = subprocess.run([binary, "census", bpv, tf.name],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL census exited {r.returncode}: {r.stderr}")
            return 1
        rep = json.load(open(tf.name))

    tokens = {t["id"]: t for t in rep["tokens"]}
    failures = 0

    def expect(cond, what):
        nonlocal failures
        if not cond:
            failures += 1
            print(f"FAIL {what}")

    # fixture guard: the probes must exist or everything below is vacuous
    expect(len(tokens) >= 264, "toy vocabulary lost its census probes")

    by_hex = {t["hex"]: t for t in rep["tokens"]}
    for b in DEAD_BYTES:
        t = by_hex.get(f"{b:02x}")
        expect(t is not None and t["verdict"] == "impossible"
               and t["reason"] == "invalid-utf8",
               f"byte {b:#04x} should be impossible/invalid-utf8, "
               f"got {t and t['verdict']}")

    t12 = by_hex.get("3132")
    expect(t12 is not None and t12["verdict"] == "impossible"
           and t12["reason"] == "pretoken-shape",
           f"'12' should be impossible/pretoken-shape, got {t12}")

    tabc = by_hex.get("616263")
    expect(tabc is not None and tabc["verdict"] == "impossible"
           and tabc["reason"] == "merge-order",
           f"'abc' should be impossible/merge-order, got {tabc}")

    ta1 = by_hex.get("a1")
    expect(ta1 is not None and ta1["verdict"] == "context"
           and "witness_hex" in ta1,
           f"continuation byte a1 should be context-witnessed, got {ta1}")

    counts = {}
    for t in rep["tokens"]:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    expect(counts.get("unresolved", 0) == 0,
           f"expected zero unresolved on the toy, got {counts}")
    expect(counts.get("impossible", 0) == len(DEAD_BYTES) + 2,
           f"expected exactly {len(DEAD_BYTES) + 2} impossible, got {counts}")
    expect(counts.get("added", 0) == 2, f"expected 2 added, got {counts}")

    # every reported witness must itself be valid UTF-8
    for t in rep["tokens"]:
        if "witness_hex" in t:
            try:
                bytes.fromhex(t["witness_hex"]).decode("utf-8")
            except UnicodeDecodeError:
                failures += 1
                print(f"FAIL witness for id {t['id']} is not valid UTF-8")

    print(f"census toy: {len(tokens)} tokens checked, {failures} failures")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
