#!/usr/bin/env python3
"""Convert a HuggingFace tokenizer.json into a .bpv vocabulary image.

Python 3 standard library only. Offline tool: no network, no third-party packages.

The .bpv format is specified in docs/DESIGN.md, section "The .bpv format". This file is
the reference producer; tools/bpv_dump.py is an independent reader used to check it, and
the C loader must agree with both byte for byte.

Determinism
-----------
The output is a pure function of (input file bytes, --source-name, --profile-regex-check
settings). There are no timestamps, no random values, no dependence on dict iteration
order or locale: every table is emitted in a fixed order (token blob in id order, pair
table by insertion in merge-rank order into a table whose size is derived from the merge
count). Converting the same input twice, on any machine, produces byte-identical output.

Refusals
--------
The converter validates every assumption bytepair's runtime makes about the tokenizer and
refuses, with a single-line error and a nonzero exit status, rather than emitting an image
that would silently mistokenize. Exit statuses:

    0  success
    1  I/O or usage error
    2  the source tokenizer violates an assumption (refusal)
    3  the written image failed self-verification (a bug in this tool)

Usage
-----
    bpv_convert.py TOKENIZER_JSON OUTPUT_BPV [--source-name NAME] [--quiet]
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from array import array

CONVERTER_VERSION = "bpv_convert 0.1.0"

# --- format constants (docs/DESIGN.md, "The .bpv format") ---------------------

MAGIC = 0x31565042  # b"BPV1" read as a little-endian u32
FORMAT_VERSION = 1
FLAG_NFC = 1 << 0
PROFILE_ID_GPT4_STYLE = 1

HEADER_FIXED_END = 136  # bytes 0..135 are the fixed fields, 136..191 are zero
HEADER_SIZE = 192  # first section starts here
SECTION_ALIGN = 64
SECTION_COUNT = 6
(
    SEC_TOKEN_OFFSETS,
    SEC_TOKEN_BLOB,
    SEC_PAIR_TABLE,
    SEC_BYTE_TO_ID,
    SEC_ADDED,
    SEC_META,
) = range(SECTION_COUNT)
SECTION_NAMES = (
    "token_offsets",
    "token_blob",
    "pair_table",
    "byte_to_id",
    "added",
    "meta",
)

EMPTY_KEY = 0xFFFFFFFFFFFFFFFF
PAIR_SLOT_BYTES = 16
MAX_LOAD_FACTOR = 0.65
MIN_PAIR_SLOTS_LOG2 = 4

MASK64 = (1 << 64) - 1
FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3

# The profile 1 pretokenizer regex, exactly as it appears in the source file.
PROFILE1_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class Refusal(Exception):
    """The source tokenizer violates an assumption bytepair depends on."""


class VerifyError(Exception):
    """The written image failed its own self-verification."""


# --- primitives ---------------------------------------------------------------


def fnv1a64(data):
    """FNV-1a, 64-bit, over a byte string."""
    h = FNV64_OFFSET
    for b in data:
        h = ((h ^ b) * FNV64_PRIME) & MASK64
    return h


def fmix64(h):
    """murmur3 fmix64, pinned in docs/DESIGN.md as the .bpv pair-table slot function."""
    h &= MASK64
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & MASK64
    h ^= h >> 33
    return h


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def byte_to_unicode():
    """The GPT-2 byte-to-unicode alphabet used by ByteLevel pretokenizers.

    Printable bytes map to themselves as codepoints; the remaining 68 bytes map, in
    ascending byte order, to U+0100 and upward.
    """
    printable = (
        list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    )
    codepoints = list(printable)
    extra = 0
    seen = set(printable)
    for b in range(256):
        if b not in seen:
            printable.append(b)
            codepoints.append(0x100 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, codepoints)}


def u32_array(values):
    """Serialize a sequence of u32 as little-endian bytes."""
    a = array("I", values)
    if a.itemsize != 4:
        raise VerifyError("platform array('I') is not 32-bit")
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def u64_array(values):
    a = array("Q", values)
    if a.itemsize != 8:
        raise VerifyError("platform array('Q') is not 64-bit")
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


# --- source validation --------------------------------------------------------


def _require(cond, message):
    if not cond:
        raise Refusal(message)


def validate_normalizer(doc):
    """Return the nfc flag bit for this tokenizer, refusing anything unsupported."""
    norm = doc.get("normalizer")
    if norm is None:
        return 0
    _require(
        isinstance(norm, dict) and set(norm.keys()) == {"type"} and norm["type"] == "NFC",
        "unsupported normalizer: bytepair implements exactly {\"type\":\"NFC\"} or null, "
        "got %s" % json.dumps(norm, sort_keys=True)[:200],
    )
    return FLAG_NFC


def validate_pre_tokenizer(doc):
    pre = doc.get("pre_tokenizer")
    _require(
        isinstance(pre, dict) and pre.get("type") == "Sequence",
        "unsupported pre_tokenizer: expected a Sequence of [Split, ByteLevel]",
    )
    stages = pre.get("pretokenizers")
    _require(
        isinstance(stages, list) and len(stages) == 2,
        "unsupported pre_tokenizer: expected exactly 2 stages, got %d"
        % (len(stages) if isinstance(stages, list) else -1),
    )
    split, bl = stages

    _require(
        isinstance(split, dict) and split.get("type") == "Split",
        "unsupported pre_tokenizer: stage 1 must be Split, got %r"
        % (split.get("type") if isinstance(split, dict) else type(split).__name__),
    )
    pattern = split.get("pattern")
    _require(
        isinstance(pattern, dict) and set(pattern.keys()) == {"Regex"},
        "unsupported Split pattern: bytepair implements only pattern.Regex",
    )
    _require(
        pattern["Regex"] == PROFILE1_REGEX,
        "unsupported Split regex: this build implements profile 1 only; source regex "
        "does not match the pinned profile 1 pattern",
    )
    _require(
        split.get("behavior") == "Isolated",
        "unsupported Split behavior: expected Isolated, got %r" % (split.get("behavior"),),
    )
    _require(
        split.get("invert") is False,
        "unsupported Split: invert must be false, got %r" % (split.get("invert"),),
    )

    _require(
        isinstance(bl, dict) and bl.get("type") == "ByteLevel",
        "unsupported pre_tokenizer: stage 2 must be ByteLevel, got %r"
        % (bl.get("type") if isinstance(bl, dict) else type(bl).__name__),
    )
    _require(
        bl.get("add_prefix_space") is False,
        "unsupported ByteLevel: add_prefix_space must be false, got %r"
        % (bl.get("add_prefix_space"),),
    )
    _require(
        bl.get("use_regex") is False,
        "unsupported ByteLevel: use_regex must be false, got %r" % (bl.get("use_regex"),),
    )
    # trim_offsets affects reported character offsets only. bytepair 0.1 has no offsets
    # API (docs/DESIGN.md, "Non-goals for 0.1"), so the field cannot change any id.


def validate_model(doc):
    model = doc.get("model")
    _require(isinstance(model, dict), "tokenizer.json has no model object")
    _require(
        model.get("type") == "BPE",
        "unsupported model type: expected BPE, got %r" % (model.get("type"),),
    )
    # Absent optional fields take their documented defaults.
    checks = (
        ("dropout", None, model.get("dropout", None)),
        ("unk_token", None, model.get("unk_token", None)),
        ("continuing_subword_prefix", "", model.get("continuing_subword_prefix", "") or ""),
        ("end_of_word_suffix", "", model.get("end_of_word_suffix", "") or ""),
        ("fuse_unk", False, model.get("fuse_unk", False)),
        ("byte_fallback", False, model.get("byte_fallback", False)),
        ("ignore_merges", False, model.get("ignore_merges", False)),
    )
    for name, expected, actual in checks:
        _require(
            actual == expected and type(actual) is type(expected),
            "unsupported BPE model: %s must be %r, got %r" % (name, expected, actual),
        )
    _require(isinstance(model.get("vocab"), dict), "model.vocab is missing or not an object")
    _require(isinstance(model.get("merges"), list), "model.merges is missing or not a list")
    return model


def validate_added_tokens(doc):
    added = doc.get("added_tokens", [])
    _require(isinstance(added, list), "added_tokens must be a list")
    out = []
    for entry in added:
        _require(isinstance(entry, dict), "added_tokens entry is not an object")
        content = entry.get("content")
        tid = entry.get("id")
        _require(isinstance(content, str) and content != "", "added token has no content")
        _require(
            isinstance(tid, int) and not isinstance(tid, bool) and tid >= 0,
            "added token %r has a non-integer id" % (content,),
        )
        for field in ("single_word", "lstrip", "rstrip", "normalized"):
            _require(
                entry.get(field, False) is False,
                "unsupported added token %r: %s must be false; bytepair implements only "
                "the all-false configuration and will not approximate it"
                % (content, field),
            )
        special = bool(entry.get("special", False))
        out.append((tid, content, special))
    ids = [t[0] for t in out]
    _require(len(set(ids)) == len(ids), "added_tokens contain duplicate ids")
    contents = [t[1] for t in out]
    _require(len(set(contents)) == len(contents), "added_tokens contain duplicate contents")
    # Emit in id order so the added section is independent of file ordering.
    out.sort(key=lambda t: t[0])
    return out


def warn_unsupported_stage(name, stage):
    if stage is None:
        return
    if isinstance(stage, dict) and stage.get("type") == "ByteLevel":
        return
    kind = stage.get("type") if isinstance(stage, dict) else type(stage).__name__
    sys.stderr.write(
        "bpv_convert: warning: %s is %r, not ByteLevel; bytepair implements the "
        "ByteLevel behavior only\n" % (name, kind)
    )


# --- build --------------------------------------------------------------------


def decode_vocab(vocab, b2u):
    """Map each vocabulary entry to its raw bytes, refusing anything off-alphabet."""
    u2b = {c: b for b, c in b2u.items()}
    _require(len(u2b) == 256, "internal: byte alphabet is not a bijection")
    id_to_bytes = {}
    for text, tid in vocab.items():
        _require(
            isinstance(tid, int) and not isinstance(tid, bool) and tid >= 0,
            "vocabulary entry %r has a non-integer id" % (text,),
        )
        try:
            raw = bytes(u2b[ch] for ch in text)
        except KeyError as exc:
            raise Refusal(
                "vocabulary entry %r contains %r, which is not in the GPT-2 byte "
                "alphabet" % (text, exc.args[0])
            )
        if tid in id_to_bytes:
            raise Refusal("vocabulary id %d is claimed by two entries" % tid)
        id_to_bytes[tid] = raw
    return id_to_bytes


def build_byte_to_id(vocab, b2u):
    """u32[256]: the single-byte token id for every byte value."""
    table = []
    for b in range(256):
        ch = b2u[b]
        tid = vocab.get(ch)
        _require(
            tid is not None,
            "vocabulary has no single-byte token for byte 0x%02X (%r); byte-level BPE "
            "requires all 256" % (b, ch),
        )
        table.append(tid)
    _require(len(set(table)) == 256, "single-byte tokens are not distinct")
    return table


def build_pair_table(merges, vocab, id_to_bytes, nslots):
    """Insert every merge into the open-addressed table; return (keys, values, stats)."""
    keys = [EMPTY_KEY] * nslots
    values = [EMPTY_KEY] * nslots
    mask = nslots - 1
    probes = 0
    max_probe = 0
    byte_mismatches = 0
    for rank, entry in enumerate(merges):
        if isinstance(entry, str):
            parts = entry.split(" ")
            _require(
                len(parts) == 2,
                "merge %d (%r) is not a two-token string; byte-level vocabularies never "
                "contain a literal space" % (rank, entry),
            )
            left_s, right_s = parts
        else:
            _require(
                isinstance(entry, (list, tuple)) and len(entry) == 2,
                "merge %d is neither a string nor a 2-element list" % rank,
            )
            left_s, right_s = entry
        _require(
            isinstance(left_s, str) and isinstance(right_s, str),
            "merge %d has non-string members" % rank,
        )
        merged_s = left_s + right_s
        left = vocab.get(left_s)
        right = vocab.get(right_s)
        merged = vocab.get(merged_s)
        _require(left is not None, "merge %d: left token %r is not in the vocabulary" % (rank, left_s))
        _require(right is not None, "merge %d: right token %r is not in the vocabulary" % (rank, right_s))
        _require(
            merged is not None,
            "merge %d: merged token %r is not in the vocabulary" % (rank, merged_s),
        )
        if id_to_bytes[left] + id_to_bytes[right] != id_to_bytes[merged]:
            byte_mismatches += 1
            continue
        _require(
            left < (1 << 32) and right < (1 << 32) and merged < (1 << 32),
            "merge %d references an id that does not fit in u32" % rank,
        )
        key = (left << 32) | right
        _require(key != EMPTY_KEY, "merge %d produces the reserved empty key" % rank)
        value = (rank << 32) | merged
        slot = fmix64(key) & mask
        step = 0
        while keys[slot] != EMPTY_KEY:
            if keys[slot] == key:
                raise Refusal(
                    "merge %d duplicates an earlier pair (%r, %r)" % (rank, left_s, right_s)
                )
            slot = (slot + 1) & mask
            step += 1
        keys[slot] = key
        values[slot] = value
        probes += step + 1
        if step + 1 > max_probe:
            max_probe = step + 1
    _require(
        byte_mismatches == 0,
        "%d merge(s) do not satisfy bytes(left)+bytes(right) == bytes(merged); the "
        "vocabulary and merge list disagree" % byte_mismatches,
    )
    return keys, values, {"probes": probes, "max_probe": max_probe}


def encode_meta(source_name, converter_version, profile_regex):
    out = bytearray()
    for text in (source_name, converter_version, profile_regex):
        blob = text.encode("utf-8")
        out += struct.pack("<I", len(blob))
        out += blob
    return bytes(out)


def convert(raw, source_name):
    """Build the complete .bpv image from the source file bytes. Returns (image, stats)."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refusal("input is not valid UTF-8 JSON: %s" % exc)
    _require(isinstance(doc, dict), "input JSON is not an object")

    flags = validate_normalizer(doc)
    validate_pre_tokenizer(doc)
    model = validate_model(doc)
    added = validate_added_tokens(doc)
    warn_unsupported_stage("decoder", doc.get("decoder"))
    warn_unsupported_stage("post_processor", doc.get("post_processor"))

    vocab = model["vocab"]
    merges = model["merges"]
    b2u = byte_to_unicode()
    id_to_bytes = decode_vocab(vocab, b2u)

    added_bytes = {}
    for tid, content, _special in added:
        _require(
            tid not in id_to_bytes,
            "added token id %d (%r) collides with a base vocabulary id" % (tid, content),
        )
        added_bytes[tid] = content.encode("utf-8")

    max_id = max(list(id_to_bytes) + [t[0] for t in added]) if (id_to_bytes or added) else -1
    _require(max_id >= 0, "the tokenizer has no tokens")
    vocab_count = max_id + 1
    _require(vocab_count < (1 << 32), "vocab_count does not fit in u32")

    # token_offsets + token_blob, in id order. Ids present in neither table (holes) get a
    # zero-length span; the runtime treats them as undefined ids.
    blob = bytearray()
    offsets = [0] * (vocab_count + 1)
    holes = 0
    for tid in range(vocab_count):
        offsets[tid] = len(blob)
        piece = id_to_bytes.get(tid)
        if piece is None:
            piece = added_bytes.get(tid)
        if piece is None:
            holes += 1
            piece = b""
        blob += piece
    offsets[vocab_count] = len(blob)
    _require(len(blob) < (1 << 32), "token blob does not fit in u32 offsets")

    byte_ids = build_byte_to_id(vocab, b2u)
    for b, tid in enumerate(byte_ids):
        _require(
            id_to_bytes[tid] == bytes([b]),
            "single-byte token for 0x%02X decodes to %d bytes, not 1"
            % (b, len(id_to_bytes[tid])),
        )

    nmerges = len(merges)
    if nmerges == 0:
        slots_log2 = MIN_PAIR_SLOTS_LOG2
    else:
        slots_log2 = max(
            MIN_PAIR_SLOTS_LOG2, int(math.ceil(math.log2(nmerges / MAX_LOAD_FACTOR)))
        )
    nslots = 1 << slots_log2
    _require(
        nmerges / nslots <= MAX_LOAD_FACTOR,
        "internal: computed pair table load factor exceeds %.2f" % MAX_LOAD_FACTOR,
    )
    keys, values, probe_stats = build_pair_table(merges, vocab, id_to_bytes, nslots)

    added_records = []
    for tid, content, special in added:
        piece = added_bytes[tid]
        added_records += [tid, offsets[tid], len(piece), 1 if special else 0]

    payloads = [None] * SECTION_COUNT
    payloads[SEC_TOKEN_OFFSETS] = u32_array(offsets)
    payloads[SEC_TOKEN_BLOB] = bytes(blob)
    pair_flat = [0] * (2 * nslots)
    pair_flat[0::2] = keys
    pair_flat[1::2] = values
    payloads[SEC_PAIR_TABLE] = u64_array(pair_flat)
    payloads[SEC_BYTE_TO_ID] = u32_array(byte_ids)
    payloads[SEC_ADDED] = u32_array(added_records)
    payloads[SEC_META] = encode_meta(source_name, CONVERTER_VERSION, PROFILE1_REGEX)

    table = []
    cursor = HEADER_SIZE
    for payload in payloads:
        table.append((cursor, len(payload)))
        cursor = align_up(cursor + len(payload), SECTION_ALIGN)
    total = cursor

    image = bytearray(total)
    struct.pack_into(
        "<8I",
        image,
        0,
        MAGIC,
        FORMAT_VERSION,
        flags,
        PROFILE_ID_GPT4_STYLE,
        vocab_count,
        len(added),
        slots_log2,
        0,
    )
    for i, (off, size) in enumerate(table):
        struct.pack_into("<QQ", image, 32 + 16 * i, off, size)
    struct.pack_into("<Q", image, 128, fnv1a64(raw))
    assert HEADER_FIXED_END == 136
    for (off, _size), payload in zip(table, payloads):
        image[off : off + len(payload)] = payload

    stats = {
        "vocab_count": vocab_count,
        "added_count": len(added),
        "holes": holes,
        "merges": nmerges,
        "pair_slots": nslots,
        "pair_slots_log2": slots_log2,
        "load_factor": nmerges / nslots,
        "mean_probe": probe_stats["probes"] / nmerges if nmerges else 0.0,
        "max_probe": probe_stats["max_probe"],
        "blob_bytes": len(blob),
        "file_bytes": total,
        "flags": flags,
        "source_hash": fnv1a64(raw),
        "sections": table,
    }
    return bytes(image), stats


# --- self-verification (cold read of the written file) -------------------------


def _vrequire(cond, message):
    if not cond:
        raise VerifyError(message)


def verify_image(path, raw_source, source_name):
    """Re-read the written file from disk and check every structural invariant.

    Reads nothing from the build state: the image is parsed as the C loader will parse
    it, and cross-checked against the source tokenizer.json independently.
    """
    with open(path, "rb") as fh:
        image = fh.read()
    size = len(image)
    _vrequire(size >= HEADER_SIZE, "file shorter than the header")
    _vrequire(size % SECTION_ALIGN == 0, "file size is not a multiple of %d" % SECTION_ALIGN)

    (magic, version, flags, profile_id, vocab_count, added_count, slots_log2, reserved) = (
        struct.unpack_from("<8I", image, 0)
    )
    _vrequire(magic == MAGIC, "bad magic 0x%08X" % magic)
    _vrequire(version == FORMAT_VERSION, "bad format_version %d" % version)
    _vrequire(flags & ~FLAG_NFC == 0, "unknown flag bits set: 0x%08X" % flags)
    _vrequire(profile_id == PROFILE_ID_GPT4_STYLE, "bad profile_id %d" % profile_id)
    _vrequire(reserved == 0, "reserved header field is not zero")
    _vrequire(vocab_count > 0, "vocab_count is zero")
    _vrequire(MIN_PAIR_SLOTS_LOG2 <= slots_log2 <= 40, "implausible pair_slots_log2 %d" % slots_log2)
    _vrequire(
        image[HEADER_FIXED_END:HEADER_SIZE] == b"\0" * (HEADER_SIZE - HEADER_FIXED_END),
        "header padding bytes 136..191 are not zero",
    )

    table = [struct.unpack_from("<QQ", image, 32 + 16 * i) for i in range(SECTION_COUNT)]
    source_hash = struct.unpack_from("<Q", image, 128)[0]
    _vrequire(source_hash == fnv1a64(raw_source), "source_hash does not match the input file")

    prev_end = HEADER_SIZE
    for i, (off, sz) in enumerate(table):
        name = SECTION_NAMES[i]
        _vrequire(off % SECTION_ALIGN == 0, "section %s is not 64-byte aligned" % name)
        _vrequire(off >= prev_end, "section %s starts before the previous section ends" % name)
        _vrequire(off + sz <= size, "section %s runs past the end of the file" % name)
        prev_end = off + sz
    _vrequire(table[0][0] == HEADER_SIZE, "first section does not start at offset 192")

    def section(i):
        off, sz = table[i]
        return image[off : off + sz]

    # token_offsets / token_blob
    _vrequire(
        table[SEC_TOKEN_OFFSETS][1] == 4 * (vocab_count + 1),
        "token_offsets size does not match vocab_count",
    )
    offsets = array("I")
    offsets.frombytes(section(SEC_TOKEN_OFFSETS))
    if sys.byteorder != "little":
        offsets.byteswap()
    blob = section(SEC_TOKEN_BLOB)
    _vrequire(offsets[0] == 0, "token_offsets[0] is not zero")
    _vrequire(offsets[vocab_count] == len(blob), "token_offsets tail does not equal blob size")
    for i in range(vocab_count):
        _vrequire(offsets[i] <= offsets[i + 1], "token_offsets not monotonic at id %d" % i)
    _vrequire(offsets[vocab_count] <= len(blob), "token_offsets run past the blob")

    def token_bytes(tid):
        return blob[offsets[tid] : offsets[tid + 1]]

    # byte_to_id
    _vrequire(table[SEC_BYTE_TO_ID][1] == 1024, "byte_to_id is not 1024 bytes")
    byte_ids = array("I")
    byte_ids.frombytes(section(SEC_BYTE_TO_ID))
    if sys.byteorder != "little":
        byte_ids.byteswap()
    seen = set()
    for b in range(256):
        tid = byte_ids[b]
        _vrequire(tid < vocab_count, "byte_to_id[0x%02X] = %d is out of range" % (b, tid))
        _vrequire(
            token_bytes(tid) == bytes([b]),
            "byte_to_id[0x%02X] points at a token that is not that single byte" % b,
        )
        seen.add(tid)
    _vrequire(len(seen) == 256, "byte_to_id entries are not distinct")

    # pair table: every merge must be retrievable through the C probe sequence
    nslots = 1 << slots_log2
    _vrequire(
        table[SEC_PAIR_TABLE][1] == nslots * PAIR_SLOT_BYTES,
        "pair_table size does not match pair_slots_log2",
    )
    pair_off = table[SEC_PAIR_TABLE][0]
    pairs = array("Q")
    pairs.frombytes(image[pair_off : pair_off + nslots * PAIR_SLOT_BYTES])
    if sys.byteorder != "little":
        pairs.byteswap()
    mask = nslots - 1

    def lookup(key):
        slot = fmix64(key) & mask
        while True:
            k = pairs[2 * slot]
            if k == key:
                return pairs[2 * slot + 1]
            if k == EMPTY_KEY:
                return None
            slot = (slot + 1) & mask

    doc = json.loads(raw_source.decode("utf-8"))
    model = doc["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    occupied = sum(1 for s in range(nslots) if pairs[2 * s] != EMPTY_KEY)
    _vrequire(
        occupied == len(merges),
        "pair table holds %d entries but the source has %d merges" % (occupied, len(merges)),
    )
    for rank, entry in enumerate(merges):
        left_s, right_s = entry.split(" ") if isinstance(entry, str) else entry
        key = (vocab[left_s] << 32) | vocab[right_s]
        value = lookup(key)
        _vrequire(value is not None, "merge %d is not retrievable from the pair table" % rank)
        _vrequire(
            value == ((rank << 32) | vocab[left_s + right_s]),
            "merge %d has the wrong rank or merged id in the pair table" % rank,
        )

    # every base vocabulary token's bytes are where the offsets say they are
    b2u = byte_to_unicode()
    u2b = {c: b for b, c in b2u.items()}
    for text, tid in vocab.items():
        _vrequire(tid < vocab_count, "vocabulary id %d exceeds vocab_count" % tid)
        _vrequire(
            token_bytes(tid) == bytes(u2b[ch] for ch in text),
            "token blob content for id %d does not match the source vocabulary" % tid,
        )

    # added records
    _vrequire(
        table[SEC_ADDED][1] == 16 * added_count, "added section size does not match added_count"
    )
    added_raw = array("I")
    added_raw.frombytes(section(SEC_ADDED))
    if sys.byteorder != "little":
        added_raw.byteswap()
    source_added = {t["id"]: t for t in doc.get("added_tokens", [])}
    _vrequire(len(source_added) == added_count, "added_count does not match the source")
    for i in range(added_count):
        tid, off, ln, aflags = added_raw[4 * i : 4 * i + 4]
        _vrequire(tid < vocab_count, "added record %d has an out-of-range id" % i)
        _vrequire(off + ln <= len(blob), "added record %d points outside the blob" % i)
        _vrequire(aflags & ~1 == 0, "added record %d has unknown flag bits" % i)
        src = source_added.get(tid)
        _vrequire(src is not None, "added record %d has an id not present in the source" % i)
        _vrequire(
            blob[off : off + ln] == src["content"].encode("utf-8"),
            "added record %d blob bytes differ from the literal token string" % i,
        )
        _vrequire(
            (off, ln) == (offsets[tid], offsets[tid + 1] - offsets[tid]),
            "added record %d span disagrees with token_offsets" % i,
        )
        _vrequire(
            aflags & 1 == (1 if src.get("special", False) else 0),
            "added record %d special flag differs from the source" % i,
        )

    # meta
    meta = section(SEC_META)
    pos = 0
    strings = []
    for _ in range(3):
        _vrequire(pos + 4 <= len(meta), "meta section truncated")
        (ln,) = struct.unpack_from("<I", meta, pos)
        pos += 4
        _vrequire(pos + ln <= len(meta), "meta string runs past the section")
        strings.append(meta[pos : pos + ln].decode("utf-8"))
        pos += ln
    _vrequire(pos == len(meta), "meta section has trailing bytes")
    _vrequire(strings[0] == source_name, "meta source name mismatch")
    _vrequire(strings[1] == CONVERTER_VERSION, "meta converter version mismatch")
    _vrequire(strings[2] == PROFILE1_REGEX, "meta profile regex mismatch")

    expected_nfc = FLAG_NFC if doc.get("normalizer") is not None else 0
    _vrequire(flags & FLAG_NFC == expected_nfc, "nfc flag does not match the source normalizer")

    return {
        "occupied": occupied,
        "sections": table,
        "file_bytes": size,
    }


# --- entry point ---------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace tokenizer.json into a .bpv vocabulary image."
    )
    parser.add_argument("input", help="path to tokenizer.json")
    parser.add_argument("output", help="path to write the .bpv image to")
    parser.add_argument(
        "--source-name",
        default=None,
        help="name recorded in the meta section (default: the input file's basename)",
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    args = parser.parse_args(argv)

    source_name = args.source_name if args.source_name is not None else os.path.basename(args.input)

    try:
        with open(args.input, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        sys.stderr.write("bpv_convert: error: cannot read %s: %s\n" % (args.input, exc))
        return 1

    t0 = time.perf_counter()
    try:
        image, stats = convert(raw, source_name)
    except Refusal as exc:
        sys.stderr.write("bpv_convert: refusing to convert: %s\n" % exc)
        return 2
    t_build = time.perf_counter() - t0

    try:
        with open(args.output, "wb") as fh:
            fh.write(image)
    except OSError as exc:
        sys.stderr.write("bpv_convert: error: cannot write %s: %s\n" % (args.output, exc))
        return 1
    t_write = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        verify_image(args.output, raw, source_name)
    except VerifyError as exc:
        sys.stderr.write("bpv_convert: self-verification failed: %s\n" % exc)
        return 3
    t_verify = time.perf_counter() - t1

    if not args.quiet:
        out = sys.stdout
        out.write("bpv_convert: wrote %s\n" % args.output)
        out.write("  source            %s (%d bytes, FNV-1a-64 0x%016x)\n"
                  % (source_name, len(raw), stats["source_hash"]))
        out.write("  vocab_count       %d (%d added, %d unused ids)\n"
                  % (stats["vocab_count"], stats["added_count"], stats["holes"]))
        out.write("  merges            %d\n" % stats["merges"])
        out.write("  pair table        %d slots (2^%d), load %.4f, mean probe %.3f, max %d\n"
                  % (stats["pair_slots"], stats["pair_slots_log2"], stats["load_factor"],
                     stats["mean_probe"], stats["max_probe"]))
        out.write("  token blob        %d bytes\n" % stats["blob_bytes"])
        out.write("  flags             0x%08x (nfc=%d), profile_id %d\n"
                  % (stats["flags"], 1 if stats["flags"] & FLAG_NFC else 0,
                     PROFILE_ID_GPT4_STYLE))
        for i, (off, sz) in enumerate(stats["sections"]):
            out.write("  section %-13s offset %9d size %9d\n" % (SECTION_NAMES[i], off, sz))
        out.write("  file size         %d bytes\n" % stats["file_bytes"])
        out.write("  build %.3f s, write %.3f s, verify %.3f s (total %.3f s)\n"
                  % (t_build, t_write - t_build, t_verify, t_write + t_verify))
    return 0


if __name__ == "__main__":
    sys.exit(main())
