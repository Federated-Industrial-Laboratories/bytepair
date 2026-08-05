#!/usr/bin/env python3
"""Inspect and validate a .bpv vocabulary image.

Python 3 standard library only. Prints the header, the section table, every added token,
a sample of vocabulary entries, and then runs the structural integrity checks the C loader
must also enforce. Exits nonzero if any check fails.

This reader is written independently of tools/bpv_convert.py on purpose: it re-derives the
byte layout, the murmur3 fmix64 slot function and the probe sequence from docs/FORMAT.md
rather than importing them, so that agreement between the two programs is evidence rather
than tautology. It needs no source tokenizer.json, so it can be pointed at an image
produced by any tool, including a future C writer.

Usage:
    bpv_dump.py IMAGE.bpv [--samples N] [--quiet]

Exit status: 0 all checks passed, 1 I/O or usage error, 3 an integrity check failed.
"""

import argparse
import struct
import sys
from array import array

MAGIC = 0x31565042  # b"BPV1"
FORMAT_VERSION = 1
FLAG_NFC = 1 << 0
HEADER_FIXED_END = 136
HEADER_SIZE = 192
SECTION_ALIGN = 64
SECTION_COUNT = 6
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
MASK64 = (1 << 64) - 1

PROFILE_NAMES = {0: "none (BPE over whole input)", 1: "GPT-4-style scanner"}


class IntegrityError(Exception):
    """A .bpv image failed a structural check."""


def check(cond, message):
    if not cond:
        raise IntegrityError(message)


def fmix64(h):
    """murmur3 fmix64, pinned in docs/FORMAT.md as the .bpv pair-table slot function."""
    h &= MASK64
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & MASK64
    h ^= h >> 33
    return h


def read_u32_array(buf):
    a = array("I")
    check(a.itemsize == 4, "platform array('I') is not 32-bit")
    check(len(buf) % 4 == 0, "u32 array section length is not a multiple of 4")
    a.frombytes(buf)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def read_u64_array(buf):
    a = array("Q")
    check(a.itemsize == 8, "platform array('Q') is not 64-bit")
    check(len(buf) % 8 == 0, "u64 array section length is not a multiple of 8")
    a.frombytes(buf)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def escape(raw):
    """Render token bytes for human reading: escaped bytes plus a UTF-8 preview."""
    shown = repr(bytes(raw))
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return shown
    if text.isprintable():
        return "%s  %s" % (shown, text)
    return shown


class Image:
    """A parsed .bpv image. Parsing alone enforces the header and section-table rules."""

    def __init__(self, data):
        self.data = data
        size = len(data)
        check(size >= HEADER_SIZE, "file is shorter than the 192-byte header region")
        fields = struct.unpack_from("<8I", data, 0)
        (
            self.magic,
            self.format_version,
            self.flags,
            self.profile_id,
            self.vocab_count,
            self.added_count,
            self.pair_slots_log2,
            self.reserved,
        ) = fields
        check(self.magic == MAGIC, "bad magic 0x%08X (expected 0x%08X)" % (self.magic, MAGIC))
        check(
            self.format_version == FORMAT_VERSION,
            "unsupported format_version %d" % self.format_version,
        )
        check(self.flags & ~FLAG_NFC == 0, "unknown flag bits set: 0x%08X" % self.flags)
        check(self.profile_id in PROFILE_NAMES, "unknown profile_id %d" % self.profile_id)
        check(self.reserved == 0, "reserved header field is not zero")
        check(self.vocab_count > 0, "vocab_count is zero")
        check(self.pair_slots_log2 <= 40, "implausible pair_slots_log2 %d" % self.pair_slots_log2)
        check(
            data[HEADER_FIXED_END:HEADER_SIZE] == b"\0" * (HEADER_SIZE - HEADER_FIXED_END),
            "header bytes 136..191 are not zero",
        )

        self.sections = [struct.unpack_from("<QQ", data, 32 + 16 * i) for i in range(SECTION_COUNT)]
        self.source_hash = struct.unpack_from("<Q", data, 128)[0]

        prev_end = HEADER_SIZE
        for i, (off, sz) in enumerate(self.sections):
            name = SECTION_NAMES[i]
            check(off % SECTION_ALIGN == 0, "section %s is not 64-byte aligned" % name)
            check(off >= prev_end, "section %s overlaps or precedes the previous section" % name)
            check(off + sz <= size, "section %s runs past the end of the file" % name)
            prev_end = off + sz
        check(
            self.sections[0][0] == HEADER_SIZE,
            "first section starts at %d, not %d" % (self.sections[0][0], HEADER_SIZE),
        )
        self.file_bytes = size

    def section(self, index):
        off, sz = self.sections[index]
        return self.data[off : off + sz]


def integrity_checks(img):
    """Run every structural check and return a dict of derived facts for reporting."""
    facts = {}

    offsets_raw = img.section(0)
    check(
        len(offsets_raw) == 4 * (img.vocab_count + 1),
        "token_offsets is %d bytes, expected %d for vocab_count %d"
        % (len(offsets_raw), 4 * (img.vocab_count + 1), img.vocab_count),
    )
    offsets = read_u32_array(offsets_raw)
    blob = img.section(1)
    check(offsets[0] == 0, "token_offsets[0] is not zero")
    for i in range(img.vocab_count):
        check(offsets[i] <= offsets[i + 1], "token_offsets is not monotonic at id %d" % i)
    check(
        offsets[img.vocab_count] == len(blob),
        "token_offsets tail is %d, token_blob is %d bytes"
        % (offsets[img.vocab_count], len(blob)),
    )
    facts["blob_bytes"] = len(blob)
    facts["empty_tokens"] = sum(
        1 for i in range(img.vocab_count) if offsets[i] == offsets[i + 1]
    )

    def token_bytes(tid):
        return blob[offsets[tid] : offsets[tid + 1]]

    byte_raw = img.section(3)
    check(len(byte_raw) == 1024, "byte_to_id is %d bytes, expected 1024" % len(byte_raw))
    byte_ids = read_u32_array(byte_raw)
    distinct = set()
    for b in range(256):
        tid = byte_ids[b]
        check(tid < img.vocab_count, "byte_to_id[0x%02X] = %d exceeds vocab_count" % (b, tid))
        got = token_bytes(tid)
        check(
            got == bytes([b]),
            "byte_to_id[0x%02X] -> id %d whose token is %r, not the single byte"
            % (b, tid, bytes(got)),
        )
        distinct.add(tid)
    check(len(distinct) == 256, "byte_to_id maps 256 bytes onto %d ids" % len(distinct))

    nslots = 1 << img.pair_slots_log2
    pair_raw = img.section(2)
    check(
        len(pair_raw) == nslots * PAIR_SLOT_BYTES,
        "pair_table is %d bytes, expected %d for 2^%d slots"
        % (len(pair_raw), nslots * PAIR_SLOT_BYTES, img.pair_slots_log2),
    )
    pairs = read_u64_array(pair_raw)
    mask = nslots - 1

    occupied = 0
    ranks = set()
    total_probes = 0
    max_probe = 0
    for slot in range(nslots):
        key = pairs[2 * slot]
        if key == EMPTY_KEY:
            continue
        occupied += 1
        value = pairs[2 * slot + 1]
        left = key >> 32
        right = key & 0xFFFFFFFF
        rank = value >> 32
        merged = value & 0xFFFFFFFF
        check(
            left < img.vocab_count and right < img.vocab_count,
            "pair table slot %d references ids (%d, %d) beyond vocab_count" % (slot, left, right),
        )
        check(
            merged < img.vocab_count,
            "pair table slot %d merges to id %d, beyond vocab_count" % (slot, merged),
        )
        check(
            token_bytes(left) + token_bytes(right) == token_bytes(merged),
            "pair table slot %d: bytes(%d)+bytes(%d) != bytes(%d)" % (slot, left, right, merged),
        )
        check(rank not in ranks, "rank %d appears in more than one pair table slot" % rank)
        ranks.add(rank)
        # The slot this key must be found in, walking the probe sequence the C loader uses.
        probe = fmix64(key) & mask
        steps = 1
        while pairs[2 * probe] != key:
            check(
                pairs[2 * probe] != EMPTY_KEY,
                "pair at slot %d is unreachable: probing from its home slot hits an empty "
                "slot first" % slot,
            )
            probe = (probe + 1) & mask
            steps += 1
            check(steps <= nslots, "probe sequence for slot %d does not terminate" % slot)
        check(probe == slot, "probe for slot %d's key lands on slot %d" % (slot, probe))
        total_probes += steps
        if steps > max_probe:
            max_probe = steps
    check(occupied > 0, "pair table is empty")
    check(
        occupied / nslots <= MAX_LOAD_FACTOR,
        "pair table load factor %.4f exceeds %.2f" % (occupied / nslots, MAX_LOAD_FACTOR),
    )
    check(
        ranks == set(range(occupied)),
        "merge ranks are not exactly 0..%d; the merge order is not recoverable" % (occupied - 1),
    )
    facts["pair_slots"] = nslots
    facts["pair_count"] = occupied
    facts["load_factor"] = occupied / nslots
    facts["mean_probe"] = total_probes / occupied
    facts["max_probe"] = max_probe

    added_raw = img.section(4)
    check(
        len(added_raw) == 16 * img.added_count,
        "added section is %d bytes, expected %d for %d records"
        % (len(added_raw), 16 * img.added_count, img.added_count),
    )
    added = read_u32_array(added_raw)
    records = []
    seen_ids = set()
    for i in range(img.added_count):
        tid, off, ln, flags = added[4 * i : 4 * i + 4]
        check(tid < img.vocab_count, "added record %d has id %d beyond vocab_count" % (i, tid))
        check(tid not in seen_ids, "added record %d repeats id %d" % (i, tid))
        seen_ids.add(tid)
        check(off + ln <= len(blob), "added record %d span runs past the token blob" % i)
        check(ln > 0, "added record %d has zero length" % i)
        check(flags & ~1 == 0, "added record %d has unknown flag bits 0x%08X" % (i, flags))
        check(
            (off, ln) == (offsets[tid], offsets[tid + 1] - offsets[tid]),
            "added record %d span (%d,%d) disagrees with token_offsets" % (i, off, ln),
        )
        content = bytes(blob[off : off + ln])
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise IntegrityError("added record %d content is not valid UTF-8" % i)
        records.append((tid, off, ln, flags, text))
    facts["added"] = records

    meta = img.section(5)
    pos = 0
    strings = []
    while pos < len(meta):
        check(pos + 4 <= len(meta), "meta section truncated in a length prefix")
        (ln,) = struct.unpack_from("<I", meta, pos)
        pos += 4
        check(pos + ln <= len(meta), "meta string runs past the section")
        try:
            strings.append(meta[pos : pos + ln].decode("utf-8"))
        except UnicodeDecodeError:
            raise IntegrityError("meta string %d is not valid UTF-8" % len(strings))
        pos += ln
    check(len(strings) == 3, "meta holds %d strings, expected 3" % len(strings))
    facts["meta"] = strings

    facts["offsets"] = offsets
    facts["blob"] = blob
    return facts


def report(img, facts, samples, out):
    out.write("header\n")
    out.write("  magic             %r (0x%08X)\n" % (img.data[0:4].decode("ascii"), img.magic))
    out.write("  format_version    %d\n" % img.format_version)
    out.write("  flags             0x%08X (nfc=%d)\n"
              % (img.flags, 1 if img.flags & FLAG_NFC else 0))
    out.write("  profile_id        %d (%s)\n" % (img.profile_id, PROFILE_NAMES[img.profile_id]))
    out.write("  vocab_count       %d\n" % img.vocab_count)
    out.write("  added_count       %d\n" % img.added_count)
    out.write("  pair_slots_log2   %d (%d slots)\n" % (img.pair_slots_log2, facts["pair_slots"]))
    out.write("  reserved          %d\n" % img.reserved)
    out.write("  source_hash       0x%016x (FNV-1a-64 of the source tokenizer.json)\n"
              % img.source_hash)
    out.write("  file size         %d bytes\n" % img.file_bytes)

    out.write("\nsections\n")
    out.write("  %-14s %12s %12s %12s\n" % ("name", "offset", "size", "end"))
    for i, (off, sz) in enumerate(img.sections):
        out.write("  %-14s %12d %12d %12d\n" % (SECTION_NAMES[i], off, sz, off + sz))

    out.write("\nmeta\n")
    for name, value in zip(("source name", "converter", "profile regex"), facts["meta"]):
        out.write("  %-14s %s\n" % (name, value))

    out.write("\nadded tokens (%d)\n" % img.added_count)
    out.write("  %8s %10s %6s %6s  %s\n" % ("id", "offset", "len", "flags", "content"))
    for tid, off, ln, flags, text in facts["added"]:
        out.write("  %8d %10d %6d %6s  %s\n"
                  % (tid, off, ln, "special" if flags & 1 else "-", text))

    offsets = facts["offsets"]
    blob = facts["blob"]
    out.write("\nvocabulary samples (%d of %d, evenly spaced)\n" % (samples, img.vocab_count))
    for k in range(samples):
        tid = k * (img.vocab_count - 1) // max(samples - 1, 1)
        raw = blob[offsets[tid] : offsets[tid + 1]]
        out.write("  %8d  %s\n" % (tid, escape(raw)))

    out.write("\npair table\n")
    out.write("  entries           %d in %d slots\n" % (facts["pair_count"], facts["pair_slots"]))
    out.write("  load factor       %.4f (limit %.2f)\n" % (facts["load_factor"], MAX_LOAD_FACTOR))
    out.write("  probes            mean %.3f, max %d\n" % (facts["mean_probe"], facts["max_probe"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect and validate a .bpv vocabulary image.")
    parser.add_argument("image", help="path to the .bpv file")
    parser.add_argument(
        "--samples", type=int, default=10, help="vocabulary entries to print (default 10)"
    )
    parser.add_argument("--quiet", action="store_true", help="run the checks, print no report")
    args = parser.parse_args(argv)
    if args.samples < 1:
        sys.stderr.write("bpv_dump: error: --samples must be at least 1\n")
        return 1

    try:
        with open(args.image, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        sys.stderr.write("bpv_dump: error: cannot read %s: %s\n" % (args.image, exc))
        return 1

    try:
        img = Image(data)
        facts = integrity_checks(img)
    except IntegrityError as exc:
        sys.stderr.write("bpv_dump: integrity check failed: %s\n" % exc)
        return 3

    if not args.quiet:
        report(img, facts, min(args.samples, img.vocab_count), sys.stdout)
        sys.stdout.write("\nall integrity checks passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
