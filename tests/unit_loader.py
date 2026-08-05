#!/usr/bin/env python3
"""Loader validation tests: every corruption class must be rejected with the
exact CLI exit code 3 (vocabulary open/validation error), and the pristine
file must load with exit 0. Python stdlib only.

    python3 tests/unit_loader.py <bytepair-binary> <valid.bpv>
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile

# header field offsets (docs/DESIGN.md, pinned layout)
MAGIC = 0
VERSION = 4
FLAGS = 8
PROFILE = 12
VOCAB_COUNT = 16
ADDED_COUNT = 20
PAIR_LOG2 = 24
SEC_TOKEN_OFFSETS = 32
SEC_TOKEN_BLOB = 48
SEC_PAIR = 64
SEC_BYTE_TO_ID = 80
SEC_ADDED = 96
SEC_META = 112

def run_info(binary, path):
    r = subprocess.run([binary, "info", path], capture_output=True)
    return r.returncode, r.stdout.decode(), r.stderr.decode()

def patched(data, off, fmt, value):
    b = bytearray(data)
    struct.pack_into(fmt, b, off, value)
    return bytes(b)

def main():
    if len(sys.argv) != 3:
        print("usage: unit_loader.py <bytepair> <valid.bpv>", file=sys.stderr)
        return 2
    binary, valid = sys.argv[1], sys.argv[2]
    data = open(valid, "rb").read()

    # fixture guards: a trivial fixture would make every test vacuous
    assert len(data) > 1_000_000, "fixture too small to be the real vocabulary"
    vocab_count = struct.unpack_from("<I", data, VOCAB_COUNT)[0]
    assert vocab_count > 100_000, "fixture vocabulary suspiciously small"

    tok_off_off = struct.unpack_from("<Q", data, SEC_TOKEN_OFFSETS)[0]

    # find a token id whose blob span is longer than 1 byte
    multi_id = None
    for i in range(vocab_count):
        a, b = struct.unpack_from("<II", data, tok_off_off + 4 * i)
        if b - a > 1:
            multi_id = i
            break
    assert multi_id is not None

    byte_to_id_off = struct.unpack_from("<Q", data, SEC_BYTE_TO_ID)[0]
    added_off = struct.unpack_from("<Q", data, SEC_ADDED)[0]

    cases = [
        ("truncated-100", data[:100]),
        ("truncated-mid", data[: len(data) // 2]),
        ("bad-magic", patched(data, MAGIC, "<I", 0xDEADBEEF)),
        ("bad-version", patched(data, VERSION, "<I", 99)),
        ("bad-profile", patched(data, PROFILE, "<I", 7)),
        ("zero-vocab", patched(data, VOCAB_COUNT, "<I", 0)),
        ("huge-vocab", patched(data, VOCAB_COUNT, "<I", 1 << 24)),
        ("pair-log2-33", patched(data, PAIR_LOG2, "<I", 33)),
        ("unaligned-section", patched(
            data, SEC_TOKEN_BLOB, "<Q",
            struct.unpack_from("<Q", data, SEC_TOKEN_BLOB)[0] + 4)),
        ("oob-section", patched(data, SEC_PAIR + 8, "<Q", 1 << 40)),
        # offset far beyond EOF, 64-aligned, size untouched: only the
        # in-bounds rule stands between the loader and unmapped memory
        ("section-off-huge", patched(data, SEC_BYTE_TO_ID, "<Q", 1 << 40)),
        ("nonmonotonic-offsets", patched(
            data, tok_off_off + 4 * 5, "<I", 0xFFFFFFF0)),
        ("byte-id-oob", patched(data, byte_to_id_off + 4 * 65, "<I",
                                0x00FFFFFF)),
        ("byte-id-multibyte", patched(data, byte_to_id_off + 4 * 65, "<I",
                                      multi_id)),
        ("added-id-oob", patched(data, added_off, "<I", vocab_count + 5)),
        ("added-len-zero", patched(data, added_off + 8, "<I", 0)),
    ]

    # meta section shifted +4 with its content intact: every field is
    # coherent and in bounds, so ONLY the 64-byte alignment rule rejects
    # this file. Guards the alignment check against deletion.
    meta_off = struct.unpack_from("<Q", data, SEC_META)[0]
    shifted = bytearray(data[:meta_off] + b"\x00\x00\x00\x00" + data[meta_off:])
    struct.pack_into("<Q", shifted, SEC_META, meta_off + 4)
    cases.append(("unaligned-coherent-meta", bytes(shifted)))

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        # pristine file loads
        ok = os.path.join(td, "ok.bpv")
        shutil.copyfile(valid, ok)
        code, out, err = run_info(binary, ok)
        if code != 0 or f"vocab: {vocab_count}" not in out:
            print(f"FAIL pristine: exit {code}, out {out!r} err {err!r}")
            failures += 1
        else:
            print("ok pristine (exit 0)")

        for name, blob in cases:
            p = os.path.join(td, name + ".bpv")
            with open(p, "wb") as f:
                f.write(blob)
            code, out, err = run_info(binary, p)
            if code != 3:
                print(f"FAIL {name}: expected exit 3, got {code} "
                      f"(out {out!r} err {err!r})")
                failures += 1
            else:
                print(f"ok {name} (exit 3)")

    print(f"{len(cases) + 1} loader cases, {failures} failures")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
