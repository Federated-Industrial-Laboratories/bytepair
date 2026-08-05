#!/usr/bin/env python3
"""Fingerprint tool checks against the reference tokenizer (quick mode).

Asserts the pinned behavioral facts this repository is built on: the NFC
data behaves as Unicode 9.0, the contraction alternative accepts long s,
a fingerprint agrees with itself, and an edited tokenizer is reported
with names rather than a bare mismatch.

Run under a python with `tokenizers`:
    test_fingerprint.py <tokenizer.json>
"""
import json
import os
import subprocess
import sys
import tempfile

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "tools", "fingerprint.py")

def run(*argv, **kw):
    return subprocess.run([sys.executable, TOOL, *argv],
                          capture_output=True, text=True, **kw)

def main():
    if len(sys.argv) != 2:
        print("usage: test_fingerprint.py <tokenizer.json>", file=sys.stderr)
        return 2
    tok = sys.argv[1]
    failures = 0

    def expect(cond, what):
        nonlocal failures
        if not cond:
            failures += 1
            print(f"FAIL {what}")

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "fp.json")
        r = run("take", tok, "-o", fp, "--quick")
        expect(r.returncode == 0, f"take exited {r.returncode}: {r.stderr}")
        data = json.load(open(fp))

        expect(data["nfc"]["effective_version"] == "9.0",
               f"nfc version {data['nfc']['effective_version']} != 9.0")
        expect(data["nfc"]["monotonic"], "nfc ladder not monotonic")
        expect("ſ" in data["fold"]["single_after_apostrophe"],
               "long s missing from contraction fold set")
        expect(all(len(v) > 0 for v in data["golden"].values()),
               "empty golden vector")

        r = run("diff", fp, fp)
        expect(r.returncode == 0 and "agree" in r.stdout,
               f"self-diff not clean: {r.stdout}")

        # an edited tokenizer must be reported by name
        drift = os.path.join(td, "drift.json")
        d = json.load(open(tok))
        d["added_tokens"][0]["content"] = "<|edited|>"
        json.dump(d, open(drift, "w"))
        fp2 = os.path.join(td, "fp2.json")
        r = run("take", drift, "-o", fp2, "--quick")
        expect(r.returncode == 0, "take on drifted copy failed")
        r = run("diff", fp, fp2)
        expect(r.returncode == 1, f"drift diff exited {r.returncode}, not 1")
        expect("added_tokens" in r.stdout,
               "drift report does not name added_tokens")

    print(f"fingerprint: 8 checks, {failures} failures")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
