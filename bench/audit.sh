#!/bin/sh
# Auditable benchmark and verification run.
#
# One command reproduces, on the machine it runs on, every public claim this
# repository makes: builds the library from source, verifies token-for-token
# equality against the HuggingFace reference (pinned version), and benchmarks
# bytepair against the reference and two third-party implementations on
# distributable corpora (a hash-pinned enwik8 slice and a deterministically
# generated CJK corpus). Results are written to bench/results/ as JSON with
# the hardware, versions, corpus hashes, and git commit stamped in.
#
#   sh bench/audit.sh          full run (verification + benchmarks)
#   sh bench/audit.sh --bench  benchmarks only (skips the differential suite)
#
# Requirements: gcc or clang, GNU make, python3 with venv, curl, unzip.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/build"
VENV="$BUILD/audit-venv"
ENWIK8_ZIP_SHA256="547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"
ENWIK8_SHA256="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"

echo "== build =="
make
make vocab

echo "== audit environment =="
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --require-virtualenv \
        -r bench/requirements.txt
fi
"$VENV/bin/pip" list 2>/dev/null | grep -Ei "tokenizers|gigatoken|bpe" || true

echo "== corpora =="
mkdir -p "$BUILD/corpora"
if [ ! -f "$BUILD/corpora/enwik8" ]; then
    curl -sL -o "$BUILD/corpora/enwik8.zip" "https://mattmahoney.net/dc/enwik8.zip"
    echo "$ENWIK8_ZIP_SHA256  $BUILD/corpora/enwik8.zip" | sha256sum -c - >/dev/null
    unzip -q -o "$BUILD/corpora/enwik8.zip" -d "$BUILD/corpora"
fi
echo "$ENWIK8_SHA256  $BUILD/corpora/enwik8" | sha256sum -c - >/dev/null
python3 bench/gen_cjk.py "$BUILD/corpora/cjk.txt"
python3 - "$BUILD/corpora/enwik8" "$BUILD/corpora/enwik8-8mb.txt" <<'EOF'
import sys
raw = open(sys.argv[1], "rb").read()[:8_000_000]
text = raw.decode("utf-8", errors="ignore")   # drop a split trailing sequence
open(sys.argv[2], "wb").write(text.encode("utf-8"))
EOF

if [ "$1" != "--bench" ]; then
    echo "== verification: differential vs reference =="
    "$VENV/bin/python" tests/differential.py \
        --bytepair "$BUILD/bytepair" --bpv "$BUILD/qwen3.bpv" \
        --tokenizer tests/data/fetched/qwen3-tokenizer.json \
        --corpus "$BUILD/corpora/enwik8-8mb.txt"
    echo "== verification: loader and long-s =="
    python3 tests/unit_loader.py "$BUILD/bytepair" "$BUILD/qwen3.bpv"
    python3 tools/bpv_convert.py tests/data/toy-longs.json \
        "$BUILD/toy-longs.bpv" --quiet
    "$VENV/bin/python" tests/test_longs_toy.py "$BUILD/bytepair" \
        "$BUILD/toy-longs.bpv" tests/data/toy-longs.json
fi

echo "== benchmarks =="
"$VENV/bin/python" bench/audit.py \
    --bytepair "$BUILD/bytepair" --bpv "$BUILD/qwen3.bpv" \
    --tokenizer tests/data/fetched/qwen3-tokenizer.json \
    --corpus "enwik8-8mb:$BUILD/corpora/enwik8-8mb.txt" \
    --corpus "cjk-synthetic:$BUILD/corpora/cjk.txt" \
    --out bench/results

echo "== audit complete =="
