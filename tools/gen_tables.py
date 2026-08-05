#!/usr/bin/env python3
"""Generate the bytepair Unicode tables (src/tables/bp_uctables.{h,c}).

The shipped HuggingFace ``tokenizers`` build carries **two different Unicode versions**,
because its regex engine and its normalizer are separate components with separately
bundled tables.  Both were established behaviourally, by probing the build:

* character classes follow **Unicode 16.0.0** (``tools/probe_classes.py``);
* NFC data follows **Unicode 9.0.0** (``tools/probe_nfc_version.py``).

Since bytepair's guarantee is token-for-token equality with that build, the tables must
reproduce both pins, not the newest data.  Inputs:

* ``tests/data/hf_classes.bin.gz`` -- the class probe.  It is ground truth; the emitted
  character-class table always follows it, and ``tests/data/ucd/`` (Unicode 16.0.0)
  supplies an independent derivation that is compared against it.
* ``tests/data/ucd9/`` -- pinned Unicode 9.0.0 files, which supply all normalization
  data: combining classes, canonical decompositions, composition pairs, NFC quick-check.
* ``tests/data/hf_nfc_probe.json`` -- the standing evidence for the 9.0.0 pin.
  Generation refuses to proceed if it is missing, was produced from different UCD files
  or a different tokenizer, or records any disagreement with the build.

The two class derivations are compared and every disagreement is reported.  A large
disagreement count means the regex engine's bundled Unicode version differs materially
from the pinned UCD, which is a fact the project must decide about rather than paper
over, so generation stops above ``--max-disagreements``.

Stdlib only.  Usage::

    python3 tools/gen_tables.py [--probe ...] [--ucd ...] [--ucd9 ...] [--out ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_classes import (  # noqa: E402  (path set above)
    BP_CC_L,
    BP_CC_N,
    BP_CC_NL,
    BP_CC_S,
    MAX_CP,
    read_classes,
    sha256_file,
)

UCD_CLASS_VERSION = "16.0.0"  # what the build's regex engine implements
UCD_NFC_VERSION = "9.0.0"     # what the build's normalizer implements
GENERATOR = "tools/gen_tables.py"
NFC_PROBE_NAME = "hf_nfc_probe.json"
BLOCK = 256
NBLOCK_INDEX = MAX_CP // BLOCK  # 4352 blocks of 256 codepoints

LETTER_CATEGORIES = ("Lu", "Ll", "Lt", "Lm", "Lo")
NUMBER_CATEGORIES = ("Nd", "Nl", "No")
SPACE_CATEGORIES = ("Zs", "Zl", "Zp")
# White_Space members that are not in a Z* category.  PropList.txt is not among the
# pinned files, so this short list is spelled out; it is exactly the Cc part of
# White_Space in every Unicode version from 4.0 onward, and the probe checks it.
WHITESPACE_CONTROLS = (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85)

HANGUL_SYLLABLE_FIRST = 0xAC00
HANGUL_SYLLABLE_LAST = 0xD7A3
HANGUL_JAMO_L_FIRST = 0x1100
HANGUL_JAMO_V_FIRST = 0x1161
HANGUL_JAMO_T_FIRST = 0x11A7

MAX_DECOMP = 4  # longest full canonical decomposition in Unicode is 4 codepoints


# ------------------------------------------------------------------ UCD parsing


def verify_sums(ucd_dir):
    """Verify every file listed in SHA256SUMS.  Returns {name: digest}."""
    path = os.path.join(ucd_dir, "SHA256SUMS")
    digests = {}
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            name = name.lstrip("*").strip()
            actual = sha256_file(os.path.join(ucd_dir, name))
            if actual != digest:
                raise SystemExit(
                    "UCD file %s does not match SHA256SUMS\n  expected %s\n  actual   %s"
                    % (name, digest, actual)
                )
            digests[name] = digest
    if not digests:
        raise SystemExit("%s lists no files" % path)
    return digests


def check_nfc_probe(path, digests9, digests16, class_meta_path):
    """Refuse to generate unless the NFC version pin is currently evidenced.

    The emitted normalization tables are deliberately older than the pinned UCD 16
    used for character classes.  That is only defensible while the probe artifact
    says the shipped build really behaves that way, was produced from these exact UCD
    files, and was produced against the same tokenizer as the class probe.
    """
    if not os.path.exists(path):
        raise SystemExit(
            "%s is missing; run tools/probe_nfc_version.py before generating" % path)
    with open(path, "r", encoding="utf-8") as fh:
        probe = json.load(fh)
    problems = []
    if probe.get("disagreements"):
        count = len(probe["disagreements"])
        problems.append("the probe records %d disagreement%s between the build and "
                        "UCD %s" % (count, "" if count == 1 else "s", UCD_NFC_VERSION))
    if probe.get("nfc_unicode_version") != UCD_NFC_VERSION:
        problems.append("the probe pins NFC to %s, this generator emits %s"
                        % (probe.get("nfc_unicode_version"), UCD_NFC_VERSION))
    if probe.get("ucd9_sha256") != digests9:
        problems.append("the probe ran against different Unicode 9.0.0 files")
    if probe.get("ucd16_sha256") != digests16:
        problems.append("the probe ran against different Unicode 16.0.0 files")
    if probe.get("reference_self_check", {}).get("failures"):
        problems.append("the probe's reference normalizer failed its own "
                        "NormalizationTest.txt")
    if not probe.get("marks_added_after_9_0", {}).get(
            "engine_treats_all_as_starters"):
        problems.append("the build does not treat every post-9.0.0 combining mark as a "
                        "starter, so the 9.0.0 pin is wrong")
    if not probe.get("decompositions_added_after_9_0", {}).get("engine_composes_none"):
        problems.append("the build composes a decomposition added after 9.0.0, so the "
                        "9.0.0 pin is wrong")
    with open(class_meta_path, "r", encoding="utf-8") as fh:
        class_meta = json.load(fh)
    if probe.get("tokenizer_sha256") != class_meta.get("tokenizer_sha256"):
        problems.append("the class probe and the NFC probe used different tokenizer "
                        "files")
    if probe.get("probed_build") != class_meta.get("probed_build"):
        problems.append("the class probe and the NFC probe used different builds (%s "
                        "vs %s)" % (class_meta.get("probed_build"),
                                    probe.get("probed_build")))
    if problems:
        raise SystemExit(
            "NFC version pin not evidenced:\n  " + "\n  ".join(problems))
    return probe


def parse_unicode_data(path):
    """Return (category, ccc, decomposition) maps keyed by codepoint.

    ``decomposition`` holds only canonical (untagged) mappings, one step, as tuples.
    First/Last range rows are expanded for category and combining class; such ranges
    never carry decompositions.
    """
    category = {}
    ccc = {}
    decomp = {}
    pending_range_start = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split(";")
            if len(fields) < 6:
                continue
            cp = int(fields[0], 16)
            name = fields[1]
            cat = fields[2]
            cls = int(fields[3])
            mapping = fields[5].strip()

            if name.endswith(", First>"):
                pending_range_start = (cp, cat, cls)
                continue
            if name.endswith(", Last>"):
                if pending_range_start is None:
                    raise SystemExit("UnicodeData.txt: Last row without First at %04X" % cp)
                start, rcat, rcls = pending_range_start
                for c in range(start, cp + 1):
                    category[c] = rcat
                    if rcls:
                        ccc[c] = rcls
                pending_range_start = None
                continue

            category[cp] = cat
            if cls:
                ccc[cp] = cls
            if mapping and not mapping.startswith("<"):
                decomp[cp] = tuple(int(x, 16) for x in mapping.split())
    if pending_range_start is not None:
        raise SystemExit("UnicodeData.txt: unterminated First/Last range")
    return category, ccc, decomp


def parse_composition_exclusions(path):
    excl = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            field = line.split(";", 1)[0].strip()
            if ".." in field:
                lo, hi = field.split("..")
                excl.update(range(int(lo, 16), int(hi, 16) + 1))
            else:
                excl.add(int(field, 16))
    return excl


def parse_derived_normalization_props(path):
    """Return (nfc_qc ranges, Full_Composition_Exclusion set).

    ``nfc_qc`` is a list of (lo, hi, value) with value 0 = No, 2 = Maybe, sorted and
    non-overlapping; anything not covered is 1 = Yes.
    """
    qc = []
    fce = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(";")]
            rng = fields[0]
            if ".." in rng:
                lo, hi = (int(x, 16) for x in rng.split(".."))
            else:
                lo = hi = int(rng, 16)
            prop = fields[1]
            if prop == "Full_Composition_Exclusion":
                fce.update(range(lo, hi + 1))
            elif prop == "NFC_QC":
                value = {"N": 0, "M": 2, "Y": 1}[fields[2]]
                if value != 1:
                    qc.append((lo, hi, value))
    qc.sort()
    for i in range(1, len(qc)):
        if qc[i][0] <= qc[i - 1][1]:
            raise SystemExit("DerivedNormalizationProps.txt: overlapping NFC_QC ranges")
    return qc, fce


# ------------------------------------------------------- derived Unicode data


def ucd_char_classes(category):
    """Derive the L / N / S / NL class bytes from the UCD alone."""
    classes = bytearray(MAX_CP)
    for cp, cat in category.items():
        if cat in LETTER_CATEGORIES:
            classes[cp] |= BP_CC_L
        elif cat in NUMBER_CATEGORIES:
            classes[cp] |= BP_CC_N
        elif cat in SPACE_CATEGORIES:
            classes[cp] |= BP_CC_S
    for cp in WHITESPACE_CONTROLS:
        classes[cp] |= BP_CC_S
    classes[0x0D] |= BP_CC_NL
    classes[0x0A] |= BP_CC_NL
    return classes


def expand_decompositions(decomp):
    """Fully expand canonical decompositions.  Hangul is absent from the UCD data."""
    cache = {}

    def expand(cp, seen):
        if cp in cache:
            return cache[cp]
        mapping = decomp.get(cp)
        if mapping is None:
            return (cp,)
        if cp in seen:
            raise SystemExit("cyclic canonical decomposition at U+%04X" % cp)
        seen = seen | {cp}
        out = []
        for part in mapping:
            out.extend(expand(part, seen))
        result = tuple(out)
        cache[cp] = result
        return result

    full = {}
    for cp in decomp:
        if HANGUL_SYLLABLE_FIRST <= cp <= HANGUL_SYLLABLE_LAST:
            raise SystemExit("unexpected Hangul syllable decomposition at U+%04X" % cp)
        expanded = expand(cp, frozenset())
        if len(expanded) > MAX_DECOMP:
            raise SystemExit(
                "U+%04X expands to %d codepoints, table holds %d"
                % (cp, len(expanded), MAX_DECOMP)
            )
        full[cp] = expanded
    return full


def composition_pairs(decomp, ccc, exclusions, fce):
    """Primary composites, per UAX #15.

    A codepoint composes from (a, b) when its canonical decomposition is exactly two
    codepoints, it is not a script composition exclusion, and its decomposition does
    not begin with a non-starter.  Singleton decompositions are excluded by the
    length-two requirement.  The result is cross-checked against the derived
    Full_Composition_Exclusion property; any difference is a hard error, since it
    would mean this reimplementation of the rule disagrees with the UCD's own.
    """
    pairs = {}
    excluded = set()
    for cp, mapping in decomp.items():
        if len(mapping) != 2:
            excluded.add(cp)  # singleton: never a primary composite
            continue
        if cp in exclusions or ccc.get(mapping[0], 0) != 0:
            excluded.add(cp)
            continue
        key = (mapping[0], mapping[1])
        if key in pairs:
            raise SystemExit(
                "two primary composites for (U+%04X, U+%04X)" % key
            )
        pairs[key] = cp
    disagree = (excluded ^ (fce & set(decomp)))
    if disagree:
        raise SystemExit(
            "composition-exclusion rule disagrees with Full_Composition_Exclusion "
            "for %d codepoints, first: %s"
            % (len(disagree), ["U+%04X" % c for c in sorted(disagree)[:10]])
        )
    for (a, b), cp in pairs.items():
        if HANGUL_JAMO_L_FIRST <= a < HANGUL_JAMO_V_FIRST or (
            HANGUL_JAMO_V_FIRST <= a <= HANGUL_JAMO_T_FIRST
        ):
            raise SystemExit("unexpected Hangul jamo composition pair")
    return pairs


# ---------------------------------------------------------------- table layout


def build_two_level(data):
    """Deduplicate 256-entry blocks.  Returns (index list, packed block bytes)."""
    index = []
    blocks = []
    seen = {}
    for base in range(0, len(data), BLOCK):
        chunk = bytes(data[base : base + BLOCK])
        slot = seen.get(chunk)
        if slot is None:
            slot = len(blocks)
            seen[chunk] = slot
            blocks.append(chunk)
        index.append(slot)
    if len(blocks) > 0xFFFF:
        raise SystemExit("block index does not fit in uint16")
    return index, b"".join(blocks)


def c_array(values, per_line, fmt="%u"):
    lines = []
    for i in range(0, len(values), per_line):
        lines.append("    " + " ".join(fmt % v + "," for v in values[i : i + per_line]))
    return "\n".join(lines)


# --------------------------------------------------------------------- emitters


def file_banner(command, probe_digest, nfc_probe_digest, digests16, digests9):
    lines = [
        "/* Generated by %s -- do not edit." % GENERATOR,
        " *",
        " * TWO Unicode versions are pinned here, and that is not a mistake.  The",
        " * HuggingFace tokenizers build this library must match token for token uses",
        " * separate components for regex matching and for normalization, and they",
        " * carry different bundled Unicode data.  Both pins were established by",
        " * probing the build itself, not by reading its documentation.",
        " *",
        " * Character classes: Unicode %s." % UCD_CLASS_VERSION,
        " *   Taken from the behavioural probe, which is the exactness ground truth:",
        " *     tests/data/hf_classes.bin.gz",
        " *     sha256 %s" % probe_digest,
        " *   Cross-checked against the pinned UCD %s files:" % UCD_CLASS_VERSION,
    ]
    for name in sorted(digests16):
        lines.append(" *     %-30s sha256 %s" % (name, digests16[name]))
    lines += [
        " *",
        " * NFC data: Unicode %s." % UCD_NFC_VERSION,
        " *   The build ships 2016-era normalization data: it treats every combining",
        " *   mark assigned after %s as an unknown starter, and does not compose"
        % UCD_NFC_VERSION,
        " *   decompositions added since.  Emitting newer data here would break",
        " *   exactness rather than improve it.  Evidence, regenerated by",
        " *   tools/probe_nfc_version.py and verified at generation time:",
        " *     tests/data/%s" % NFC_PROBE_NAME,
        " *     sha256 %s" % nfc_probe_digest,
        " *   Taken from the pinned UCD %s files:" % UCD_NFC_VERSION,
    ]
    for name in sorted(digests9):
        lines.append(" *     %-30s sha256 %s" % (name, digests9[name]))
    lines += [
        " *",
        " * Regenerate with:",
        " *   %s" % command,
        " */",
        "",
        "",
    ]
    return "\n".join(lines)


def emit_header(path, banner, nblocks):
    text = banner + """#ifndef BP_UCTABLES_H
#define BP_UCTABLES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The two Unicode versions the target build implements.  BP_UCD_VERSION governs the
   character classes; the normalization functions below follow the older
   BP_UCD_NFC_VERSION, because that is what the build does.  See the banner above. */
#define BP_UCD_VERSION "%(version)s"
#define BP_UCD_NFC_VERSION "%(nfc_version)s"

/* Character classes of the Qwen3 split regex.  BP_CC_NL is set only for U+000D and
   U+000A, and always together with BP_CC_S. */
enum { BP_CC_L = 1, BP_CC_N = 2, BP_CC_S = 4, BP_CC_NL = 8 };

/* Two-level lookup: 4352 blocks of 256 codepoints, deduplicated. */
#define BP_CC_BLOCK_COUNT %(nblocks)u
extern const uint16_t bp_cc_index[%(nindex)u];
extern const uint8_t bp_cc_blocks[%(nbytes)u];

/* Class bits of a codepoint; 0 for unclassified and for anything above U+10FFFF. */
static inline uint8_t bp_char_class(uint32_t cp)
{
    if (cp > 0x10FFFFu) {
        return 0;
    }
    return bp_cc_blocks[((uint32_t)bp_cc_index[cp >> 8] << 8) | (cp & 0xFFu)];
}

/* Canonical combining class per Unicode %(nfc_version)s; 0 for starters, for anything above
   U+10FFFF, and for every mark assigned after %(nfc_version)s -- see the banner. */
uint8_t bp_ccc(uint32_t cp);

/* NFC quick check: 0 = No, 1 = Yes, 2 = Maybe. */
int bp_nfc_qc(uint32_t cp);

/* Full canonical decomposition, already recursively expanded.  Returns the number of
   codepoints written to out (1..%(maxdecomp)u), or 0 if cp has none.  Hangul syllables are
   excluded: their decomposition is algorithmic and belongs in the normalizer. */
int bp_decomp(uint32_t cp, uint32_t out[%(maxdecomp)u]);

/* Primary composite of the pair (a, b), or 0 if the pair does not compose.  Hangul
   jamo composition is excluded and likewise belongs in the normalizer. */
uint32_t bp_compose(uint32_t a, uint32_t b);

#ifdef __cplusplus
}
#endif

#endif /* BP_UCTABLES_H */
""" % {
        "version": UCD_CLASS_VERSION,
        "nfc_version": UCD_NFC_VERSION,
        "nblocks": nblocks,
        "nindex": NBLOCK_INDEX,
        "nbytes": nblocks * BLOCK,
        "maxdecomp": MAX_DECOMP,
    }
    with open(path, "w", encoding="ascii") as fh:
        fh.write(text)
    return len(text)


def emit_source(path, banner, cc_index, cc_blocks, ccc_index, ccc_blocks, qc, decomp,
                pairs):
    decomp_cps = sorted(decomp)
    pool = []
    offsets = []
    lengths = []
    for cp in decomp_cps:
        offsets.append(len(pool))
        lengths.append(len(decomp[cp]))
        pool.extend(decomp[cp])

    pair_keys = sorted((a << 32) | b for (a, b) in pairs)
    pair_vals = [pairs[(k >> 32, k & 0xFFFFFFFF)] for k in pair_keys]

    out = [banner]
    out.append('#include "bp_uctables.h"\n')
    out.append("#include <stddef.h>\n")

    out.append("\n/* Character classes (from the behavioural probe). */")
    out.append("const uint16_t bp_cc_index[%u] = {" % NBLOCK_INDEX)
    out.append(c_array(cc_index, 16))
    out.append("};\n")
    out.append("const uint8_t bp_cc_blocks[%u] = {" % len(cc_blocks))
    out.append(c_array(list(cc_blocks), 24))
    out.append("};\n")

    out.append("\n/* Canonical combining classes (UnicodeData.txt field 3). */")
    out.append("static const uint16_t bp_ccc_index[%u] = {" % NBLOCK_INDEX)
    out.append(c_array(ccc_index, 16))
    out.append("};\n")
    out.append("static const uint8_t bp_ccc_blocks[%u] = {" % len(ccc_blocks))
    out.append(c_array(list(ccc_blocks), 24))
    out.append("};\n")
    out.append("""
uint8_t bp_ccc(uint32_t cp)
{
    if (cp > 0x10FFFFu) {
        return 0;
    }
    return bp_ccc_blocks[((uint32_t)bp_ccc_index[cp >> 8] << 8) | (cp & 0xFFu)];
}
""")

    out.append("\n/* NFC_QC ranges (DerivedNormalizationProps.txt); unlisted is Yes. */")
    out.append("#define BP_QC_COUNT %u" % len(qc))
    out.append("static const uint32_t bp_qc_lo[BP_QC_COUNT] = {")
    out.append(c_array([r[0] for r in qc], 8, "0x%06X"))
    out.append("};\n")
    out.append("static const uint32_t bp_qc_hi[BP_QC_COUNT] = {")
    out.append(c_array([r[1] for r in qc], 8, "0x%06X"))
    out.append("};\n")
    out.append("static const uint8_t bp_qc_val[BP_QC_COUNT] = {")
    out.append(c_array([r[2] for r in qc], 24))
    out.append("};\n")
    out.append("""
int bp_nfc_qc(uint32_t cp)
{
    size_t lo = 0, hi = BP_QC_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < bp_qc_lo[mid]) {
            hi = mid;
        } else if (cp > bp_qc_hi[mid]) {
            lo = mid + 1;
        } else {
            return (int)bp_qc_val[mid];
        }
    }
    return 1; /* Yes */
}
""")

    out.append("\n/* Full canonical decompositions, recursively expanded, Hangul aside. */")
    out.append("#define BP_DECOMP_COUNT %u" % len(decomp_cps))
    out.append("static const uint32_t bp_decomp_cp[BP_DECOMP_COUNT] = {")
    out.append(c_array(decomp_cps, 8, "0x%06X"))
    out.append("};\n")
    out.append("static const uint32_t bp_decomp_off[BP_DECOMP_COUNT] = {")
    out.append(c_array(offsets, 12))
    out.append("};\n")
    out.append("static const uint8_t bp_decomp_len[BP_DECOMP_COUNT] = {")
    out.append(c_array(lengths, 32))
    out.append("};\n")
    out.append("static const uint32_t bp_decomp_pool[%u] = {" % len(pool))
    out.append(c_array(pool, 8, "0x%06X"))
    out.append("};\n")
    out.append("""
int bp_decomp(uint32_t cp, uint32_t out[%(maxdecomp)u])
{
    size_t lo = 0, hi = BP_DECOMP_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint32_t key = bp_decomp_cp[mid];
        if (cp < key) {
            hi = mid;
        } else if (cp > key) {
            lo = mid + 1;
        } else {
            unsigned n = bp_decomp_len[mid];
            unsigned off = bp_decomp_off[mid];
            unsigned i;
            for (i = 0; i < n; i++) {
                out[i] = bp_decomp_pool[off + i];
            }
            return (int)n;
        }
    }
    return 0;
}
""" % {"maxdecomp": MAX_DECOMP})

    out.append("\n/* Primary composites, keyed by (a << 32) | b. */")
    out.append("#define BP_COMPOSE_COUNT %u" % len(pair_keys))
    out.append("static const uint64_t bp_compose_key[BP_COMPOSE_COUNT] = {")
    out.append(c_array(pair_keys, 4, "UINT64_C(0x%016X)"))
    out.append("};\n")
    out.append("static const uint32_t bp_compose_val[BP_COMPOSE_COUNT] = {")
    out.append(c_array(pair_vals, 8, "0x%06X"))
    out.append("};\n")
    out.append("""
uint32_t bp_compose(uint32_t a, uint32_t b)
{
    uint64_t key = ((uint64_t)a << 32) | (uint64_t)b;
    size_t lo = 0, hi = BP_COMPOSE_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t k = bp_compose_key[mid];
        if (key < k) {
            hi = mid;
        } else if (key > k) {
            lo = mid + 1;
        } else {
            return bp_compose_val[mid];
        }
    }
    return 0;
}
""")

    text = "\n".join(out)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(text)
    return len(text)


# The codepoints are chosen by hand to cover every class and every NFC property;
# the expected values are computed from the probe artifact and the UCD at generation
# time, so the self-test cannot drift from the data it is meant to guard.
SELFTEST_CLASS_CPS = [
    0x0000, 0x0009, 0x000A, 0x000D, 0x0020, 0x0027, 0x0030, 0x0041, 0x005A, 0x007A,
    0x0085, 0x00A0, 0x00B2, 0x00C9, 0x0301, 0x0660, 0x06F0, 0x0F20, 0x180E, 0x1680,
    0x2000, 0x200A, 0x200B, 0x2028, 0x2029, 0x202F, 0x205F, 0x2160, 0x3000, 0x3042,
    0x4E00, 0xAC00, 0xFFFD, 0x10000, 0x1D7CE, 0x1F600, 0xE0001, 0x10FFFF,
]
SELFTEST_CCC_CPS = [
    0x0041, 0x0300, 0x0301, 0x0316, 0x0334, 0x0345, 0x05B0, 0x093C, 0x0E38, 0x3099,
    0x1E00, 0x10FFFF,
    # Marks assigned after Unicode 9.0.0.  Every one of these has a nonzero combining
    # class in Unicode 16.0.0 and must nevertheless read as 0 here, because the target
    # build treats them as unknown starters.  These cases exist to fail loudly if the
    # NFC pin is ever silently moved forward.
    0x0897, 0x07FD, 0x09FE, 0x0C3C, 0x1ABF, 0x1DF9, 0x1E130, 0x1193D, 0x089A, 0x10EFD,
]
SELFTEST_QC_CPS = [
    0x0041, 0x0300, 0x0344, 0x0958, 0x09DC, 0x0F43, 0x1E69, 0x2126, 0x212B, 0xFB1D,
    0x1D160,
]
SELFTEST_DECOMP_CPS = [
    0x0041, 0x00C0, 0x00C9, 0x01D5, 0x0344, 0x0958, 0x1E69, 0x1F82, 0x2126, 0x212B,
    0x2ADC, 0xAC00, 0xD7A3, 0x1109A,
    # Canonical decompositions added after Unicode 9.0.0: must read as absent.
    0x105C9, 0x11383, 0x113C5, 0x11938, 0x16121, 0x16D68,
]
SELFTEST_COMPOSE_PAIRS = [
    (0x0041, 0x0301), (0x0045, 0x0301), (0x0041, 0x030A), (0x0043, 0x0327),
    (0x0055, 0x0308), (0x0928, 0x093C), (0x0915, 0x093C), (0x1100, 0x1161),
    (0xAC00, 0x11A8), (0x0041, 0x0041), (0x0000, 0x0000), (0x03A9, 0x0301),
    (0x0073, 0x0323), (0x1E63, 0x0307),
    # Pairs that compose only from Unicode 13.0.0 / 16.0.0 onward: must not compose.
    (0x11935, 0x11930), (0x113C2, 0x113C2), (0x1611E, 0x1611E), (0x16D67, 0x16D67),
    (0x105D2, 0x0307),
]


def emit_selftest(path, banner, classes, ccc, qc_lookup, decomp, pairs):
    def qc_of(cp):
        for lo, hi, value in qc_lookup:
            if lo <= cp <= hi:
                return value
        return 1

    rows = [banner]
    rows.append('#include "bp_uctables.h"\n')
    rows.append("#include <stdio.h>\n")
    rows.append("""
static int failures;

static void check(const char *what, unsigned long cp, long got, long want)
{
    if (got != want) {
        printf("FAIL %s U+%04lX: got %ld, want %ld\\n", what, cp, got, want);
        failures++;
    }
}
""")

    rows.append("struct u32_case { uint32_t cp; uint32_t want; };\n")
    rows.append("static const struct u32_case class_cases[] = {")
    for cp in SELFTEST_CLASS_CPS:
        rows.append("    { 0x%06Xu, %uu }," % (cp, classes[cp] if cp < MAX_CP else 0))
    rows.append("};\n")

    rows.append("static const struct u32_case ccc_cases[] = {")
    for cp in SELFTEST_CCC_CPS:
        rows.append("    { 0x%06Xu, %uu }," % (cp, ccc.get(cp, 0)))
    rows.append("};\n")

    rows.append("static const struct u32_case qc_cases[] = {")
    for cp in SELFTEST_QC_CPS:
        rows.append("    { 0x%06Xu, %uu }," % (cp, qc_of(cp)))
    rows.append("};\n")

    rows.append("struct decomp_case { uint32_t cp; int n; uint32_t out[%d]; };\n"
                % MAX_DECOMP)
    rows.append("static const struct decomp_case decomp_cases[] = {")
    for cp in SELFTEST_DECOMP_CPS:
        seq = decomp.get(cp, ())
        padded = list(seq) + [0] * (MAX_DECOMP - len(seq))
        rows.append(
            "    { 0x%06Xu, %d, { %s } },"
            % (cp, len(seq), ", ".join("0x%06Xu" % v for v in padded))
        )
    rows.append("};\n")

    rows.append("struct compose_case { uint32_t a, b, want; };\n")
    rows.append("static const struct compose_case compose_cases[] = {")
    for a, b in SELFTEST_COMPOSE_PAIRS:
        rows.append("    { 0x%06Xu, 0x%06Xu, 0x%06Xu }," % (a, b, pairs.get((a, b), 0)))
    rows.append("};\n")

    rows.append("""
int main(void)
{
    size_t i, j;
    uint32_t out[%(maxdecomp)d];
    int n;

    for (i = 0; i < sizeof class_cases / sizeof class_cases[0]; i++) {
        check("class", class_cases[i].cp,
              bp_char_class(class_cases[i].cp), (long)class_cases[i].want);
    }
    check("class", 0x110000ul, bp_char_class(0x110000u), 0);
    check("class", 0xFFFFFFFFul, bp_char_class(0xFFFFFFFFu), 0);

    for (i = 0; i < sizeof ccc_cases / sizeof ccc_cases[0]; i++) {
        check("ccc", ccc_cases[i].cp, bp_ccc(ccc_cases[i].cp), (long)ccc_cases[i].want);
    }
    check("ccc", 0x110000ul, bp_ccc(0x110000u), 0);

    for (i = 0; i < sizeof qc_cases / sizeof qc_cases[0]; i++) {
        check("nfc_qc", qc_cases[i].cp,
              bp_nfc_qc(qc_cases[i].cp), (long)qc_cases[i].want);
    }

    for (i = 0; i < sizeof decomp_cases / sizeof decomp_cases[0]; i++) {
        n = bp_decomp(decomp_cases[i].cp, out);
        check("decomp count", decomp_cases[i].cp, n, decomp_cases[i].n);
        if (n == decomp_cases[i].n) {
            for (j = 0; j < (size_t)n; j++) {
                check("decomp element", decomp_cases[i].cp,
                      (long)out[j], (long)decomp_cases[i].out[j]);
            }
        }
    }

    for (i = 0; i < sizeof compose_cases / sizeof compose_cases[0]; i++) {
        check("compose", ((unsigned long)compose_cases[i].a << 16)
              | compose_cases[i].b,
              (long)bp_compose(compose_cases[i].a, compose_cases[i].b),
              (long)compose_cases[i].want);
    }

    if (failures != 0) {
        printf("%(fmt)s failure(s)\\n", failures);
        return 1;
    }
    printf("bp_uctables self-test passed (classes UCD " BP_UCD_VERSION
           ", NFC UCD " BP_UCD_NFC_VERSION ")\\n");
    return 0;
}
""" % {"maxdecomp": MAX_DECOMP, "fmt": "%d"})

    text = "\n".join(rows)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(text)
    return len(text)


# ------------------------------------------------------------------------- main


def describe(cls):
    if cls == 0:
        return "other"
    names = []
    for bit, name in ((BP_CC_L, "L"), (BP_CC_N, "N"), (BP_CC_S, "S"), (BP_CC_NL, "NL")):
        if cls & bit:
            names.append(name)
    return "|".join(names)


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", default=os.path.join(root, "tests", "data",
                                                    "hf_classes.bin.gz"))
    ap.add_argument("--ucd", default=os.path.join(root, "tests", "data", "ucd"),
                    help="UCD %s files: character classes" % UCD_CLASS_VERSION)
    ap.add_argument("--ucd9", default=os.path.join(root, "tests", "data", "ucd9"),
                    help="UCD %s files: all NFC data" % UCD_NFC_VERSION)
    ap.add_argument("--nfc-probe", default=os.path.join(root, "tests", "data",
                                                        NFC_PROBE_NAME))
    ap.add_argument("--out", default=os.path.join(root, "src", "tables"))
    ap.add_argument("--max-disagreements", type=int, default=200,
                    help="stop instead of emitting above this many class differences")
    args = ap.parse_args(argv)

    command = " ".join(
        ["python3", GENERATOR] + (argv if argv is not None else sys.argv[1:]))

    ucd_digests = verify_sums(args.ucd)
    ucd9_digests = verify_sums(args.ucd9)
    class_meta = os.path.join(os.path.dirname(args.probe), "hf_classes.meta.json")
    nfc_probe = check_nfc_probe(args.nfc_probe, ucd9_digests, ucd_digests, class_meta)
    nfc_probe_digest = sha256_file(args.nfc_probe)
    probe_digest = sha256_file(args.probe)
    probe_classes = read_classes(args.probe)

    # Character classes: Unicode 16.0.0, cross-checked against the behavioural probe.
    category, _, _ = parse_unicode_data(os.path.join(args.ucd, "UnicodeData.txt"))
    # Normalization: Unicode 9.0.0, which is what the target build implements.
    _, ccc, raw_decomp = parse_unicode_data(
        os.path.join(args.ucd9, "UnicodeData.txt"))
    exclusions = parse_composition_exclusions(
        os.path.join(args.ucd9, "CompositionExclusions.txt"))
    qc, fce = parse_derived_normalization_props(
        os.path.join(args.ucd9, "DerivedNormalizationProps.txt"))

    derived = ucd_char_classes(category)
    disagreements = [cp for cp in range(MAX_CP) if derived[cp] != probe_classes[cp]]

    print("bytepair table generator")
    print("  classes       : UCD %s (%d files verified), probe sha256 %s"
          % (UCD_CLASS_VERSION, len(ucd_digests), probe_digest))
    print("  NFC           : UCD %s (%d files verified), pin evidenced by %d "
          "assertions over %d codepoints"
          % (UCD_NFC_VERSION, len(ucd9_digests), nfc_probe["assertions"],
             nfc_probe["codepoints_probed"]))
    print("  class check   : %d codepoints differ between UCD and probe"
          % len(disagreements))
    for cp in disagreements[:200]:
        print("      U+%06X  ucd=%-6s probe=%-6s  category=%s"
              % (cp, describe(derived[cp]), describe(probe_classes[cp]),
                 category.get(cp, "Cn")))
    if len(disagreements) > args.max_disagreements:
        print("STOPPING: %d disagreements exceeds --max-disagreements %d; the regex "
              "engine's Unicode version is materially different from the pinned UCD "
              "and the project must decide what to do about it."
              % (len(disagreements), args.max_disagreements), file=sys.stderr)
        return 1

    full_decomp = expand_decompositions(raw_decomp)
    pairs = composition_pairs(raw_decomp, ccc, exclusions, fce)

    ccc_bytes = bytearray(MAX_CP)
    for cp, value in ccc.items():
        ccc_bytes[cp] = value

    cc_index, cc_blocks = build_two_level(probe_classes)
    ccc_index, ccc_blocks = build_two_level(ccc_bytes)

    os.makedirs(args.out, exist_ok=True)
    banner = file_banner(command, probe_digest, nfc_probe_digest, ucd_digests,
                         ucd9_digests)
    header = os.path.join(args.out, "bp_uctables.h")
    source = os.path.join(args.out, "bp_uctables.c")
    selftest = os.path.join(args.out, "bp_uctables_selftest.c")

    emit_header(header, banner, len(cc_blocks) // BLOCK)
    emit_source(source, banner, cc_index, cc_blocks, ccc_index, ccc_blocks, qc,
                full_decomp, pairs)
    emit_selftest(selftest, banner, probe_classes, ccc, qc, full_decomp, pairs)

    print("  generated     : %s (%.1f KB source)"
          % (header, os.path.getsize(header) / 1024.0))
    print("                  %s (%.1f KB source)"
          % (source, os.path.getsize(source) / 1024.0))
    print("                  %s (%.1f KB source)"
          % (selftest, os.path.getsize(selftest) / 1024.0))
    print("  table data    : classes %d blocks (%d bytes) + index %d bytes"
          % (len(cc_blocks) // BLOCK, len(cc_blocks), NBLOCK_INDEX * 2))
    print("                  ccc     %d blocks (%d bytes) + index %d bytes"
          % (len(ccc_blocks) // BLOCK, len(ccc_blocks), NBLOCK_INDEX * 2))
    print("                  nfc_qc  %d ranges (%d bytes)" % (len(qc), len(qc) * 9))
    pool_len = sum(len(v) for v in full_decomp.values())
    print("                  decomp  %d entries, %d pool codepoints (%d bytes)"
          % (len(full_decomp), pool_len,
             len(full_decomp) * 9 + pool_len * 4))
    print("                  compose %d pairs (%d bytes)" % (len(pairs), len(pairs) * 12))
    total = (len(cc_blocks) + NBLOCK_INDEX * 2 + len(ccc_blocks) + NBLOCK_INDEX * 2
             + len(qc) * 9 + len(full_decomp) * 9 + pool_len * 4 + len(pairs) * 12)
    print("                  total   %.1f KB" % (total / 1024.0))
    print("  generated at  : %s"
          % datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
