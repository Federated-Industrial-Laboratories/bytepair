#!/usr/bin/env python3
"""Probe the shipped HuggingFace ``tokenizers`` build for its character classes.

The bytepair pretokenizer is a hand-compiled scanner for the Qwen3 split regex::

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}
    | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

Exactness therefore depends on agreeing with the regex engine inside ``tokenizers``
about which codepoints are ``\\p{L}``, ``\\p{N}`` and ``\\s``.  That engine carries its
own bundled Unicode tables, whose version is not documented at runtime, so the
authoritative answer is behavioural, not documentary: we ask the shipped build.

Method
------
For a codepoint X we call ``pre_tokenizer.pre_tokenize_str`` on two probe strings and
count the pieces.  Under the ordered-alternation semantics of the regex the piece
counts identify the class uniquely:

===============  ============  ============  =====================================
class            ``"aXa"``     ``"!X!"``     why
===============  ============  ============  =====================================
letter           1             2             ``aXa`` is one maximal ``\\p{L}+`` run
number           3             3             ``\\p{N}`` matches exactly one char
whitespace       2             3             ``\\s`` blocks the ``[^\\s...]+`` run;
                                             ``[^\\r\\n\\p{L}\\p{N}]?`` still takes it
whitespace 0x20  2             2             plus the literal ``" ?"`` of alternative 4
CR / LF          3             2             ``[\\r\\n]*`` tail of alternative 4;
                                             ``\\s*[\\r\\n]+`` splits ``aXa`` in three
other            2             1             one punctuation run swallows ``!X!``
===============  ============  ============  =====================================

Any other shape is an unmodelled interaction and aborts the probe rather than being
guessed at.  The shapes for U+0020, U+000D and U+000A are additionally asserted to be
produced by those codepoints alone, so the CR/LF set is confirmed by behaviour and not
merely assumed.

Surrogates U+D800..U+DFFF cannot cross the Python/Rust string boundary and cannot occur
in valid UTF-8; they are recorded as class 0 without being probed.

Artifact format (``tests/data/hf_classes.bin.gz``)
--------------------------------------------------
gzip (mtime forced to 0, so the file is byte-reproducible) of exactly 0x110000 bytes:
one class byte per codepoint, indexed by codepoint, U+0000 first.  Bits:

    1  BP_CC_L   letter      matches \\p{L}
    2  BP_CC_N   number      matches \\p{N}
    4  BP_CC_S   whitespace  matches \\s
    8  BP_CC_NL  CR or LF    matches [\\r\\n]   (always accompanied by BP_CC_S)

Class 0 means none of the above.  ``read_classes()`` below is the reader;
``tools/gen_tables.py`` uses it and treats this artifact as ground truth.

The companion ``tests/data/hf_classes.meta.json`` records which build was probed.

Usage::

    python3 tools/probe_classes.py --tokenizer path/to/tokenizer.json [--jobs N]

Requires the ``tokenizers`` package (probe only; ``read_classes`` is stdlib-only).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone

MAX_CP = 0x110000  # one past the last codepoint
SURROGATE_FIRST = 0xD800
SURROGATE_LAST = 0xDFFF

BP_CC_L = 1
BP_CC_N = 2
BP_CC_S = 4
BP_CC_NL = 8

CLASS_NAMES = (
    (BP_CC_L, "letter"),
    (BP_CC_N, "number"),
    (BP_CC_S, "whitespace"),
    (BP_CC_NL, "CR/LF"),
)

# (pieces("aXa"), pieces("!X!")) -> class byte.  See the module docstring.
SHAPE_TO_CLASS = {
    (1, 2): BP_CC_L,
    (3, 3): BP_CC_N,
    (2, 3): BP_CC_S,
    (2, 2): BP_CC_S,             # expected for U+0020 alone
    (3, 2): BP_CC_S | BP_CC_NL,  # expected for U+000D and U+000A alone
    (2, 1): 0,
}

ARTIFACT_NAME = "hf_classes.bin.gz"
META_NAME = "hf_classes.meta.json"


# --------------------------------------------------------------------------- reader


def read_classes(path):
    """Read a probe artifact and return its 0x110000 class bytes.

    Stdlib only, so table generation does not need the tokenizers package.
    """
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    if len(data) != MAX_CP:
        raise ValueError(
            "%s: expected %d class bytes, found %d" % (path, MAX_CP, len(data))
        )
    return data


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- probe

_PRE = None


def _worker_init(tokenizer_path):
    global _PRE
    from tokenizers import Tokenizer  # imported here: the reader stays stdlib-only

    _PRE = Tokenizer.from_file(tokenizer_path).pre_tokenizer


def _probe_chunk(args):
    """Classify [lo, hi).

    Returns (lo, class bytes, unmodelled shapes, rare shapes).  "Rare shapes" are the
    two probe shapes expected from a single codepoint each, returned so the caller can
    confirm that expectation instead of assuming it.
    """
    lo, hi = args
    pre = _PRE
    out = bytearray(hi - lo)
    odd = []
    rare = []
    for cp in range(lo, hi):
        if SURROGATE_FIRST <= cp <= SURROGATE_LAST:
            continue  # class 0; cannot be probed and cannot occur in valid UTF-8
        ch = chr(cp)
        shape = (
            len(pre.pre_tokenize_str("a" + ch + "a")),
            len(pre.pre_tokenize_str("!" + ch + "!")),
        )
        cls = SHAPE_TO_CLASS.get(shape)
        if cls is None:
            odd.append((cp, shape[0], shape[1]))
            cls = 0
        elif shape in ((2, 2), (3, 2)):
            rare.append((shape, cp))
        out[cp - lo] = cls
    return lo, bytes(out), odd, rare


def probe_all(tokenizer_path, jobs, chunk=4096):
    chunks = [(lo, min(lo + chunk, MAX_CP)) for lo in range(0, MAX_CP, chunk)]
    classes = bytearray(MAX_CP)
    unmodelled = []
    rare = []
    pool = None
    if jobs <= 1:
        _worker_init(tokenizer_path)
        results = map(_probe_chunk, chunks)
    else:
        pool = multiprocessing.Pool(
            processes=jobs, initializer=_worker_init, initargs=(tokenizer_path,)
        )
        results = pool.imap_unordered(_probe_chunk, chunks, chunksize=1)
    for lo, blob, odd, few in results:
        classes[lo : lo + len(blob)] = blob
        unmodelled.extend(odd)
        rare.extend(few)
    if pool is not None:
        pool.close()
        pool.join()
    unmodelled.sort()
    rare.sort()
    return classes, unmodelled, rare


def write_artifact(path, classes):
    """Write the gzipped class bytes reproducibly (no embedded timestamp)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(bytes(classes))
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())


def summarize(classes):
    counts = {name: 0 for _, name in CLASS_NAMES}
    counts["other"] = 0
    for cls in classes:
        if cls == 0:
            counts["other"] += 1
            continue
        for bit, name in CLASS_NAMES:
            if cls & bit:
                counts[name] += 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True, help="path to a tokenizer.json")
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
            "data",
        ),
        help="output directory (default: tests/data)",
    )
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    artifact = os.path.join(args.out, ARTIFACT_NAME)
    meta_path = os.path.join(args.out, META_NAME)

    import tokenizers  # fail early and loudly if the package is missing

    started = time.time()
    classes, unmodelled, rare = probe_all(args.tokenizer, args.jobs)
    elapsed = time.time() - started

    # Behavioural confirmation of the two literal sets the scanner hard-codes:
    # the " ?" of alternative 4 (U+0020) and the "[\r\n]" classes (U+000D, U+000A).
    shape_22 = sorted(cp for shape, cp in rare if shape == (2, 2))
    shape_32 = sorted(cp for shape, cp in rare if shape == (3, 2))
    problems = []
    if unmodelled:
        problems.append(
            "unmodelled probe shapes for %d codepoints, first 10: %s"
            % (len(unmodelled), unmodelled[:10])
        )
    if shape_22 != [0x20]:
        problems.append(
            "the literal-space alternative fires for %s, expected U+0020 alone"
            % ["U+%04X" % cp for cp in shape_22]
        )
    if shape_32 != [0x0A, 0x0D]:
        problems.append(
            "the CR/LF classes fire for %s, expected U+000A and U+000D alone"
            % ["U+%04X" % cp for cp in shape_32]
        )

    counts = summarize(classes)
    write_artifact(artifact, classes)
    digest = sha256_file(artifact)

    meta = {
        "format": "gzip of %d class bytes, one per codepoint from U+0000; "
        "bits L=1 N=2 S=4 NL=8" % MAX_CP,
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": digest,
        "probed_build": "tokenizers %s (Python %s)"
        % (tokenizers.__version__, sys.version.split()[0]),
        "tokenizer_file": os.path.basename(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "probe_strings": ['a<CP>a', '!<CP>!'],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": counts,
        "surrogates": "U+D800..U+DFFF recorded as class 0 without probing",
        "unmodelled_shapes": unmodelled,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")

    total = sum(counts.values()) - counts["CR/LF"]  # NL is a subset of whitespace
    print("probed %d codepoints in %.1f s with %d job(s)" % (MAX_CP, elapsed, args.jobs))
    print("  build          : %s" % meta["probed_build"])
    for key in ("letter", "number", "whitespace", "CR/LF", "other"):
        print("  %-14s : %d" % (key, counts[key]))
    print("  accounted for  : %d of %d" % (total, MAX_CP))
    print("  whitespace set : %s" % " ".join(
        "U+%04X" % cp for cp in range(MAX_CP) if classes[cp] & BP_CC_S))
    print("  artifact       : %s (%d bytes, sha256 %s)"
          % (artifact, os.path.getsize(artifact), digest))
    print("  metadata       : %s" % meta_path)

    if problems:
        for problem in problems:
            print("PROBE PROBLEM: %s" % problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
