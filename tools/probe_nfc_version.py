#!/usr/bin/env python3
"""Pin the Unicode version of the shipped normalizer's NFC data, behaviourally.

The regex engine and the normalizer inside HuggingFace ``tokenizers`` carry separate
bundled Unicode tables, and they are not the same vintage.  ``tools/probe_classes.py``
established that the split regex classifies characters per Unicode 16.0.0.  This script
establishes, by the same method -- asking the shipped build rather than reading
documentation -- that its NFC data is Unicode 9.0.0.

Method
------
A reference NFC implementation is built from the pinned UCD 9.0.0 files and first
checked against that release's own NormalizationTest.txt.  Every codepoint is then put
through four probe forms and the engine's output is compared with the reference:

    X            round trip: decomposition followed by recomposition
    X U+0334     partner with combining class 1: reorders if X's class exceeds 1
    U+0345 X     partner with combining class 240: reorders if X's class is in (0, 240)
    a X          composition against a starter

The two partners bracket the whole combining-class range, so between them any codepoint
the engine treats as a non-starter is detected; a codepoint the engine treats as unknown
(class 0) reorders in neither.  Any codepoint whose four outputs all match the reference
agrees with Unicode 9.0.0 for every behaviour NFC can exhibit on short sequences.

Two derived tallies are reported and stored because they are the direct evidence for the
version pin:

* the 120 codepoints that gained a nonzero combining class between Unicode 9.0.0 and
  16.0.0 must all be treated as starters by the engine;
* the 21 codepoints that gained a canonical decomposition after 9.0.0 must not be
  composed by the engine from their parts.

The result is written to ``tests/data/hf_nfc_probe.json``, which ``tools/gen_tables.py``
reads and refuses to generate without: it is the standing evidence that the emitted NFC
tables match the build bytepair must be exact against.

Usage::

    python3 tools/probe_nfc_version.py --tokenizer path/to/tokenizer.json [--jobs N]

Requires the ``tokenizers`` package.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_tables import (  # noqa: E402  (path set above)
    composition_pairs,
    expand_decompositions,
    parse_composition_exclusions,
    parse_derived_normalization_props,
    parse_unicode_data,
    verify_sums,
)
from probe_classes import MAX_CP, sha256_file  # noqa: E402

ARTIFACT_NAME = "hf_nfc_probe.json"

SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT

PARTNER_LOW = 0x0334   # COMBINING TILDE OVERLAY, canonical combining class 1
PARTNER_HIGH = 0x0345  # COMBINING GREEK YPOGEGRAMMENI, canonical combining class 240

PROBE_FORMS = ("X", "X+U+0334", "U+0345+X", "a+X")


class Reference:
    """NFC and NFD over one pinned set of UCD tables."""

    def __init__(self, ccc, decomp_full, pairs):
        self.ccc = ccc
        self.decomp = decomp_full
        self.pairs = pairs

    def _hangul_decomp(self, cp):
        if not SBASE <= cp < SBASE + SCOUNT:
            return None
        si = cp - SBASE
        out = [LBASE + si // NCOUNT, VBASE + (si % NCOUNT) // TCOUNT]
        if si % TCOUNT:
            out.append(TBASE + si % TCOUNT)
        return out

    def _compose(self, a, b):
        if LBASE <= a < LBASE + LCOUNT and VBASE <= b < VBASE + VCOUNT:
            return SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT
        if (SBASE <= a < SBASE + SCOUNT and (a - SBASE) % TCOUNT == 0
                and TBASE < b < TBASE + TCOUNT):
            return a + (b - TBASE)
        return self.pairs.get((a, b), 0)

    def nfd(self, cps):
        out = []
        for cp in cps:
            hangul = self._hangul_decomp(cp)
            if hangul is not None:
                out.extend(hangul)
            else:
                out.extend(self.decomp.get(cp, (cp,)))
        ccc = self.ccc
        for i in range(1, len(out)):
            ch = out[i]
            cls = ccc.get(ch, 0)
            if cls == 0:
                continue
            j = i
            while j > 0 and ccc.get(out[j - 1], 0) > cls:
                out[j] = out[j - 1]
                j -= 1
            out[j] = ch
        return out

    def nfc(self, cps):
        d = self.nfd(cps)
        if not d:
            return []
        ccc = self.ccc
        out = [d[0]]
        starter_pos = 0
        starter_ch = d[0]
        last_class = 256 if ccc.get(d[0], 0) else 0
        for ch in d[1:]:
            ch_class = ccc.get(ch, 0)
            composite = self._compose(starter_ch, ch)
            if composite and (last_class < ch_class or last_class == 0):
                out[starter_pos] = composite
                starter_ch = composite
            else:
                if ch_class == 0:
                    starter_pos = len(out)
                    starter_ch = ch
                last_class = ch_class
                out.append(ch)
        return out

    def nfc_str(self, text):
        return "".join(chr(cp) for cp in self.nfc([ord(c) for c in text]))


def load_reference(ucd_dir):
    _, ccc, raw_decomp = parse_unicode_data(os.path.join(ucd_dir, "UnicodeData.txt"))
    exclusions = parse_composition_exclusions(
        os.path.join(ucd_dir, "CompositionExclusions.txt"))
    _, fce = parse_derived_normalization_props(
        os.path.join(ucd_dir, "DerivedNormalizationProps.txt"))
    full = expand_decompositions(raw_decomp)
    pairs = composition_pairs(raw_decomp, ccc, exclusions, fce)
    return Reference(ccc, full, pairs), ccc, raw_decomp


def check_reference(reference, path):
    """Run the reference against the NormalizationTest.txt of its own release."""
    rows = checks = failures = 0
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("@"):
                continue
            fields = line.split(";")
            if len(fields) < 5:
                continue
            cols = [[int(x, 16) for x in f.split()] for f in fields[:5]]
            rows += 1
            for source, want in ((cols[0], cols[1]), (cols[1], cols[1]),
                                 (cols[2], cols[1]), (cols[3], cols[3]),
                                 (cols[4], cols[3])):
                checks += 1
                if reference.nfc(source) != want:
                    failures += 1
                    if len(examples) < 5:
                        examples.append(["%04X" % c for c in source])
    return {"rows": rows, "assertions": checks, "failures": failures,
            "examples": examples}


_ENGINE = None
_REF = None


def _init(tokenizer_path, reference):
    global _ENGINE, _REF
    from tokenizers import Tokenizer

    _ENGINE = Tokenizer.from_file(tokenizer_path).normalizer
    _REF = reference


def _probe_chunk(bounds):
    lo, hi = bounds
    engine, ref = _ENGINE, _REF
    bad = []
    for cp in range(lo, hi):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        for form, text in (
            (0, ch),
            (1, ch + chr(PARTNER_LOW)),
            (2, chr(PARTNER_HIGH) + ch),
            (3, "a" + ch),
        ):
            got = engine.normalize_str(text)
            want = ref.nfc_str(text)
            if got != want:
                bad.append({
                    "cp": "U+%04X" % cp,
                    "form": PROBE_FORMS[form],
                    "engine": ["%04X" % ord(c) for c in got],
                    "reference": ["%04X" % ord(c) for c in want],
                })
    return bad


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ucd9", default=os.path.join(root, "tests", "data", "ucd9"))
    ap.add_argument("--ucd", default=os.path.join(root, "tests", "data", "ucd"))
    ap.add_argument("--out", default=os.path.join(root, "tests", "data"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)

    import tokenizers

    digests9 = verify_sums(args.ucd9)
    digests16 = verify_sums(args.ucd)

    started = time.time()
    reference, ccc9, decomp9 = load_reference(args.ucd9)
    self_check = check_reference(
        reference, os.path.join(args.ucd9, "NormalizationTest.txt"))
    print("reference NFC from UCD 9.0.0: %d rows, %d assertions, %d failures"
          % (self_check["rows"], self_check["assertions"], self_check["failures"]))
    if self_check["failures"]:
        print("the reference implementation is wrong; nothing downstream is trustworthy",
              file=sys.stderr)
        return 1

    _, ccc16, decomp16 = parse_unicode_data(os.path.join(args.ucd, "UnicodeData.txt"))
    new_marks = sorted(set(ccc16) - set(ccc9))
    new_decomps = sorted(set(decomp16) - set(decomp9))
    changed = [cp for cp in set(ccc16) & set(ccc9) if ccc16[cp] != ccc9[cp]]
    if changed:
        print("combining classes changed between 9.0.0 and 16.0.0 for %d codepoints; "
              "the additive assumption behind this probe does not hold" % len(changed),
              file=sys.stderr)
        return 1

    chunks = [(lo, min(lo + 4096, MAX_CP)) for lo in range(0, MAX_CP, 4096)]
    if args.jobs <= 1:
        _init(args.tokenizer, reference)
        results = map(_probe_chunk, chunks)
        pool = None
    else:
        pool = multiprocessing.Pool(args.jobs, initializer=_init,
                                    initargs=(args.tokenizer, reference))
        results = pool.imap_unordered(_probe_chunk, chunks, chunksize=1)
    disagreements = [item for part in results for item in part]
    if pool is not None:
        pool.close()
        pool.join()
    disagreements.sort(key=lambda d: (d["cp"], d["form"]))
    elapsed = time.time() - started

    # The two direct consequences of the version pin, checked explicitly.
    _init(args.tokenizer, reference)
    starter_exceptions = []
    for cp in new_marks:
        ch = chr(cp)
        low = _ENGINE.normalize_str(ch + chr(PARTNER_LOW))
        high = _ENGINE.normalize_str(chr(PARTNER_HIGH) + ch)
        if low != ch + chr(PARTNER_LOW) or high != chr(PARTNER_HIGH) + ch:
            starter_exceptions.append("U+%04X" % cp)

    compose_exceptions = []
    composable = []
    for cp in new_decomps:
        parts = decomp16[cp]
        if len(parts) != 2 or ccc16.get(parts[0], 0) != 0:
            continue  # not a primary composite even under Unicode 16 rules
        composable.append(cp)
        text = "".join(chr(p) for p in parts)
        if _ENGINE.normalize_str(text) != text:
            compose_exceptions.append("U+%04X" % cp)

    probed = MAX_CP - 0x800  # surrogates are skipped
    meta = {
        "artifact": ARTIFACT_NAME,
        "purpose": "behavioural pin of the Unicode version of the normalizer's NFC data",
        "probed_build": "tokenizers %s (Python %s)"
                        % (tokenizers.__version__, sys.version.split()[0]),
        "tokenizer_file": os.path.basename(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "nfc_unicode_version": "9.0.0",
        "class_unicode_version": "16.0.0",
        "ucd9_sha256": digests9,
        "ucd16_sha256": digests16,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probe_forms": list(PROBE_FORMS),
        "codepoints_probed": probed,
        "assertions": probed * len(PROBE_FORMS),
        "reference_self_check": self_check,
        "marks_added_after_9_0": {
            "count": len(new_marks),
            "codepoints": ["U+%04X" % cp for cp in new_marks],
            "engine_treats_all_as_starters": not starter_exceptions,
            "exceptions": starter_exceptions,
        },
        "decompositions_added_after_9_0": {
            "count": len(new_decomps),
            "composable_under_unicode_16": ["U+%04X" % cp for cp in composable],
            "engine_composes_none": not compose_exceptions,
            "exceptions": compose_exceptions,
        },
        "disagreements": disagreements,
    }
    out_path = os.path.join(args.out, ARTIFACT_NAME)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")

    print("probed %d codepoints x %d forms (%d assertions) in %.1f s with %d job(s)"
          % (probed, len(PROBE_FORMS), probed * len(PROBE_FORMS), elapsed, args.jobs))
    print("  build                        : %s" % meta["probed_build"])
    print("  disagreements with UCD 9.0.0 : %d" % len(disagreements))
    for item in disagreements[:20]:
        print("      %s %s: engine %s, reference %s"
              % (item["cp"], item["form"], item["engine"], item["reference"]))
    print("  marks added after 9.0.0      : %d, all treated as starters: %s"
          % (len(new_marks), not starter_exceptions))
    print("  decomps added after 9.0.0    : %d (%d composable under 16.0.0), "
          "engine composes none: %s"
          % (len(new_decomps), len(composable), not compose_exceptions))
    print("  artifact                     : %s" % out_path)
    return 1 if (disagreements or starter_exceptions or compose_exceptions) else 0


if __name__ == "__main__":
    sys.exit(main())
