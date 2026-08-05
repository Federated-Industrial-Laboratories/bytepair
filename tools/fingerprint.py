#!/usr/bin/env python3
"""Behavioral fingerprint of a HuggingFace tokenizer.

A tokenizer's real specification is its behavior, and behavior drifts:
library upgrades change Unicode tables, model releases quietly edit added
tokens, reimplementations diverge on edges nobody documented. This tool
pins behavior to a file, and diffs two such files with names and
codepoints rather than a bare checksum mismatch.

    fingerprint.py take <tokenizer.json> -o fp.json [--quick] [--jobs N]
    fingerprint.py diff a.json b.json

What a fingerprint records, all probed from the running implementation:

  pipeline   normalizer/pretokenizer/model shapes, added tokens, file hash
  classes    the split regex's per-codepoint classification over all of
             Unicode (letter / number / whitespace / CR-LF / other),
             probed behaviorally, stored complete (gzip) plus a digest
  nfc        the normalizer's effective Unicode version, located by a
             ladder of combining marks dated by their assignment version
  fold       which characters the contraction alternative accepts at each
             position, beyond ASCII
  norms      normalizer outputs for a fixed probe set
  golden     token ids for a fixed adversarial corpus

--quick skips the full-Unicode class sweep (the slowest part); such a
fingerprint records "classes": null and diffs accordingly.

Requires the `tokenizers` package. Everything else is stdlib.
"""
import argparse
import base64
import gzip
import hashlib
import json
import multiprocessing as mp
import sys

# combining marks dated by the Unicode version that assigned them, with
# nonzero canonical combining class > 1 so the U+0334 reorder probe is
# informative at every rung (established against UCD data, 2026-08-05)
NFC_LADDER = [
    (0x0653, "3.0"), (0x0357, "4.0"), (0x0358, "4.1"), (0x1B44, "5.0"),
    (0xABED, "5.2"), (0x0859, "6.0"), (0x1AB0, "7.0"), (0x08E3, "8.0"),
    (0x08D4, "9.0"), (0x1DF9, "10.0"), (0x07FD, "11.0"), (0x1E130, "12.0"),
    (0x1193D, "13.0"), (0x089A, "14.0"), (0x10EFD, "15.0"), (0x0897, "16.0"),
]

# candidates the contraction alternative might accept beyond ASCII, per
# position (curated from case-folding curiosities; --thorough is the full
# letter sweep and takes minutes)
FOLD_SINGLE = "sStTmMdD" + "ſKµßıİẛ"
FOLD_FIRST = "rRvVlL" + "ſK"
FOLD_SECOND = "eElL" + "ſKẛ"

NORM_PROBES = [
    "café", "é", "가", "가", "豈", "豈",
    "q̣̇", "̴࢚", "̴߽", "̴᷹",
    "ཱི", "ſ", "ﬁ",
]

GOLDEN = [
    "hello world", "it's", "IT'S", "'ſx", "a   b", " 5", "  \n\n  \nx",
    "x  ", "\tword", "3.14159", "1,000,000", "中文分词测试", "ไทยที่",
    "한국어", "👨‍👩‍👧‍👦", "café", "café", "don't we've I'll",
    "a<think>b", "<|endoftext|>", "x<|im_start|>y", "  leading", "trail  ",
    "CamelCase snake_case kebab-case", "line\r\nline", "α β γ", "مرحبا",
    "!!!", " ...", "#include <stdio.h>", "https://example.com/path?q=1",
]

SHAPE_TO_CLASS = { (1, 2): 1, (3, 3): 2, (2, 3): 3, (2, 2): 3, (3, 2): 4,
                   (2, 1): 0 }
CLASS_NAMES = ["other", "letter", "number", "whitespace", "crlf",
               "unmodelled"]

_worker_pt = None

def _worker_init(path):
    global _worker_pt
    from tokenizers import Tokenizer
    _worker_pt = Tokenizer.from_file(path).pre_tokenizer

def _classify_chunk(args):
    start, end = args
    out = bytearray()
    for cp in range(start, end):
        if 0xD800 <= cp <= 0xDFFF:
            out.append(0)
            continue
        x = chr(cp)
        s1 = _worker_pt.pre_tokenize_str(f"a{x}a")
        s2 = _worker_pt.pre_tokenize_str(f"!{x}!")
        out.append(SHAPE_TO_CLASS.get((len(s1), len(s2)), 5))
    return start, bytes(out)

def probe_classes(path, jobs):
    step = 0x2000
    chunks = [(s, min(s + step, 0x110000)) for s in range(0, 0x110000, step)]
    table = bytearray(0x110000)
    with mp.Pool(jobs, _worker_init, (path,)) as pool:
        for start, data in pool.imap_unordered(_classify_chunk, chunks):
            table[start:start + len(data)] = data
    counts = [0] * 6
    for b in table:
        counts[b] += 1
    return {
        "sha256": hashlib.sha256(bytes(table)).hexdigest(),
        "counts": {CLASS_NAMES[i]: counts[i] for i in range(6)},
        "table_gzip_b64": base64.b64encode(
            gzip.compress(bytes(table), 9)).decode(),
    }

def probe_nfc(norm):
    rungs = []
    for cp, ver in NFC_LADDER:
        s = chr(cp) + "̴"
        rungs.append({"cp": f"U+{cp:04X}", "assigned": ver,
                      "known": norm.normalize_str(s) != s})
    effective = "unknown"
    for r in rungs:
        if r["known"]:
            effective = r["assigned"]
    consistent = all(r["known"] == (i <= max(
        (j for j, rr in enumerate(rungs) if rr["known"]), default=-1))
        for i, r in enumerate(rungs))
    return {"rungs": rungs, "effective_version": effective,
            "monotonic": consistent}

def probe_fold(pt):
    def a1_matches(s, tail="x"):
        pieces = pt.pre_tokenize_str(s + tail)
        return len(pieces) == 2 and pieces[1][0] == tail
    single = sorted({c for c in FOLD_SINGLE if a1_matches("'" + c)})
    firsts = {}
    for f in FOLD_FIRST:
        seconds = sorted({c for c in FOLD_SECOND
                          if a1_matches("'" + f + c)})
        if seconds:
            firsts[f] = seconds
    return {"single_after_apostrophe": single,
            "two_letter": {k: v for k, v in sorted(firsts.items())}}

def take(args):
    from tokenizers import Tokenizer
    import importlib.metadata as md
    raw = open(args.tokenizer, "rb").read()
    tok = Tokenizer.from_file(args.tokenizer)
    spec = json.loads(raw)
    fp = {
        "fingerprint_version": 1,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizers_version": md.version("tokenizers"),
        "vocab_size": tok.get_vocab_size(with_added_tokens=True),
        "pipeline": {
            "normalizer": spec.get("normalizer"),
            "pre_tokenizer": spec.get("pre_tokenizer"),
            "model_flags": {k: v for k, v in spec["model"].items()
                            if k not in ("vocab", "merges")},
            "added_tokens": [
                {"id": a["id"], "content": a["content"],
                 "special": a["special"]}
                for a in spec.get("added_tokens", [])],
        },
        "nfc": probe_nfc(tok.normalizer) if tok.normalizer else None,
        "fold": probe_fold(tok.pre_tokenizer) if tok.pre_tokenizer else None,
        "norms": {repr(s): tok.normalizer.normalize_str(s)
                  for s in NORM_PROBES} if tok.normalizer else None,
        "golden": {repr(s): tok.encode(s, add_special_tokens=False).ids
                   for s in GOLDEN},
        "classes": None if args.quick
                   else probe_classes(args.tokenizer, args.jobs),
    }
    json.dump(fp, open(args.out, "w"), indent=1)
    n = fp["classes"]
    print(f"fingerprint written to {args.out}")
    print(f"  nfc effective version : "
          f"{fp['nfc']['effective_version'] if fp['nfc'] else 'no normalizer'}")
    if fp["fold"]:
        print(f"  contraction singles   : "
              f"{''.join(fp['fold']['single_after_apostrophe'])}")
    if n:
        print(f"  class counts          : " + ", ".join(
            f"{k} {v}" for k, v in n["counts"].items() if v))
    return 0

def diff(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    changes = 0

    def report(what, av, bv):
        nonlocal changes
        changes += 1
        print(f"DIFFERS {what}:\n  a: {av}\n  b: {bv}")

    for key in ("vocab_size", "tokenizers_version"):
        if a.get(key) != b.get(key):
            report(key, a.get(key), b.get(key))
    if a["pipeline"] != b["pipeline"]:
        pa, pb = a["pipeline"], b["pipeline"]
        for k in pa:
            if pa.get(k) != pb.get(k):
                report(f"pipeline.{k}", json.dumps(pa.get(k))[:200],
                       json.dumps(pb.get(k))[:200])
    if (a.get("nfc") or {}).get("effective_version") != \
       (b.get("nfc") or {}).get("effective_version"):
        report("nfc.effective_version",
               (a.get("nfc") or {}).get("effective_version"),
               (b.get("nfc") or {}).get("effective_version"))
    if a.get("fold") != b.get("fold"):
        report("contraction fold sets", a.get("fold"), b.get("fold"))
    for probe in (a.get("norms") or {}):
        av = a["norms"].get(probe)
        bv = (b.get("norms") or {}).get(probe)
        if av != bv:
            report(f"normalizer output for {probe}", av, bv)
    for s in a.get("golden", {}):
        av, bv = a["golden"].get(s), (b.get("golden") or {}).get(s)
        if av != bv:
            report(f"golden ids for {s}", av, bv)

    ca, cb = a.get("classes"), b.get("classes")
    if ca and cb and ca["sha256"] != cb["sha256"]:
        ta = gzip.decompress(base64.b64decode(ca["table_gzip_b64"]))
        tb = gzip.decompress(base64.b64decode(cb["table_gzip_b64"]))
        diffs = [i for i in range(min(len(ta), len(tb)))
                 if ta[i] != tb[i]]
        changes += 1
        print(f"DIFFERS classes: {len(diffs)} codepoints; first 20:")
        for cp in diffs[:20]:
            print(f"  U+{cp:04X}: {CLASS_NAMES[ta[cp]]} -> "
                  f"{CLASS_NAMES[tb[cp]]}")
    elif (ca is None) != (cb is None):
        print("note: class sweep present in only one fingerprint "
              "(one was taken with --quick); not compared")

    if changes == 0:
        print("fingerprints agree")
        return 0
    print(f"{changes} behavioral difference(s)")
    return 1

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("take")
    t.add_argument("tokenizer")
    t.add_argument("-o", "--out", required=True)
    t.add_argument("--quick", action="store_true")
    t.add_argument("--jobs", type=int, default=mp.cpu_count())
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args()
    return take(args) if args.cmd == "take" else diff(args)

if __name__ == "__main__":
    sys.exit(main())
