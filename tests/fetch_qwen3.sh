#!/bin/sh
# Fetch the Qwen3-8B tokenizer.json used by the differential tests, and verify it against
# a pinned SHA-256. The file is large (11 MB) and belongs to its upstream publisher, so it
# is downloaded rather than committed; downloads land in tests/data/fetched/, which is not
# part of the source tree's tracked content.
#
# Re-running is cheap: if the file is already present and matches the pin, nothing happens.
#
# Exit status: 0 file present and verified, 1 usage/environment error, 2 download failed,
# 3 checksum mismatch.

set -eu

URL="https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json"

# SHA-256 of Qwen/Qwen3-8B tokenizer.json, revision as published 2026-08-05.
# Cross-checked on 2026-08-05 against the x-linked-etag header returned by
# huggingface.co for this URL, which carries the object's SHA-256, and against the
# 11422654-byte local copy.
SHA256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST_DIR="$SCRIPT_DIR/data/fetched"
DEST="$DEST_DIR/qwen3-tokenizer.json"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | sed 's/.*= *//'
    else
        echo "fetch_qwen3: no sha256sum, shasum or openssl available" >&2
        exit 1
    fi
}

if [ -f "$DEST" ]; then
    have=$(sha256_of "$DEST")
    if [ "$have" = "$SHA256" ]; then
        echo "fetch_qwen3: $DEST already present and matches the pin"
        exit 0
    fi
    echo "fetch_qwen3: $DEST exists but does not match the pin; re-downloading" >&2
fi

mkdir -p "$DEST_DIR"
TMP="$DEST.part"
trap 'rm -f "$TMP"' EXIT INT TERM

echo "fetch_qwen3: downloading $URL"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$TMP" "$URL" || {
        echo "fetch_qwen3: download failed" >&2
        exit 2
    }
elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$TMP" "$URL" || {
        echo "fetch_qwen3: download failed" >&2
        exit 2
    }
else
    echo "fetch_qwen3: neither curl nor wget is available" >&2
    exit 1
fi

got=$(sha256_of "$TMP")
if [ "$got" != "$SHA256" ]; then
    echo "fetch_qwen3: checksum mismatch for $URL" >&2
    echo "  expected $SHA256" >&2
    echo "  got      $got" >&2
    echo "The pinned hash is part of the test contract: the differential tests compare" >&2
    echo "bytepair against this exact tokenizer. A mismatch means upstream published a" >&2
    echo "different file. Do not edit the pin to silence this. Inspect the new file," >&2
    echo "re-run the differential suite against it, and update SHA256 in this script as a" >&2
    echo "deliberate, reviewed change recording which upstream revision it refers to." >&2
    exit 3
fi

mv "$TMP" "$DEST"
trap - EXIT INT TERM
echo "fetch_qwen3: wrote $DEST ($SHA256)"
