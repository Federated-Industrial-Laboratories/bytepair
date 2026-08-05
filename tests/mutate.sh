#!/bin/sh
# Mutation discipline: each named mutation breaks one load-bearing behavior;
# the test suite must FAIL under every one of them. A mutation the suite
# survives means the guarding test does not exist yet - this script then
# exits nonzero and names it.
#
# Each sed is verified to have actually changed the source, so drift in the
# code cannot silently turn a mutation into a no-op.
#
# Environment: BP_BPV, BP_TOKENIZER, BP_PY, BP_CORPUS as in run_tests.sh.
set -e

BP_BPV="${BP_BPV:-build/qwen3.bpv}"
BP_TOKENIZER="${BP_TOKENIZER:-tests/data/fetched/qwen3-tokenizer.json}"
BP_PY="${BP_PY:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/build/mutate"

run_one() {
    name="$1" file="$2" pattern="$3" replacement="$4" suite="$5"
    dir="$WORK/$name"
    rm -rf "$dir"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/include" "$ROOT/cli" "$ROOT/tests" "$dir/"
    before=$(cksum "$dir/$file")
    sed -i "s#$pattern#$replacement#" "$dir/$file"
    after=$(cksum "$dir/$file")
    if [ "$before" = "$after" ]; then
        echo "MUTATION $name: sed did not change $file (source drift?)" >&2
        exit 2
    fi
    # every defined mutation is valid C; a build failure means the harness
    # itself is broken, and counting it as "caught" would be vacuous
    ( cd "$dir" && \
      gcc -std=c11 -O2 -Wall -fno-strict-aliasing -mavx2 -mbmi2 \
        -Iinclude -Isrc \
        src/bp_util.c src/bp_vocab.c src/bp_nfc.c src/bp_scan.c \
        src/bp_scan_avx2.c src/bp_bpe.c src/bp_encode.c \
        src/tables/bp_uctables.c cli/main.c \
        -o bytepair-mut -lpthread ) || {
        echo "MUTATION $name: build failed (harness defect)" >&2
        exit 2
    }
    if [ "$suite" = "loader" ]; then
        if python3 "$dir/tests/unit_loader.py" "$dir/bytepair-mut" "$BP_BPV" \
            >/dev/null 2>&1; then
            echo "SURVIVED $name: loader suite stayed green" >&2
            exit 1
        fi
    elif [ "$suite" = "longs" ]; then
        python3 "$ROOT/tools/bpv_convert.py" "$ROOT/tests/data/toy-longs.json" \
            "$dir/toy-longs.bpv" --quiet
        if "$BP_PY" "$dir/tests/test_longs_toy.py" "$dir/bytepair-mut" \
            "$dir/toy-longs.bpv" "$ROOT/tests/data/toy-longs.json" \
            >/dev/null 2>&1; then
            echo "SURVIVED $name: long-s toy suite stayed green" >&2
            exit 1
        fi
    else
        if "$BP_PY" "$dir/tests/differential.py" --bytepair "$dir/bytepair-mut" \
            --bpv "$BP_BPV" --tokenizer "$BP_TOKENIZER" --quick \
            ${BP_CORPUS:+--corpus "$BP_CORPUS"} >/dev/null 2>&1; then
            echo "SURVIVED $name: differential suite stayed green" >&2
            exit 1
        fi
    fi
    echo "ok $name (suite screamed)"
}

run_one nfc-disabled src/bp_nfc.c \
    'if (!(c->v->hdr.flags \& BPV_FLAG_NFC) || len == 0 ||' \
    'if (1 || len == 0 ||' diff

run_one giveback-dropped src/bp_scan.c \
    'else if (ncp >= 2)        end = last_cp_start;' \
    'else if (0)               end = last_cp_start;' diff

run_one rank-tie-broken src/bp_bpe.c \
    'if (rank < best_rank)' \
    'if (rank <= best_rank)' diff

run_one loader-align-dropped src/bp_vocab.c \
    'if (s->off < BPV_FIRST_SECTION || (s->off \& 63)) return 0;' \
    'if (0) return 0;' loader

run_one loader-bounds-dropped src/bp_vocab.c \
    'if (s->off > file_size || s->size > file_size - s->off) return 0;' \
    'if (0) return 0;' loader

# dropping the full verification (hash, length, bytes) must be caught; note
# that hash-only comparison is NOT catchable by random tests - 64-bit
# collisions among ~10^6 pretokens are vanishingly rare, which is exactly
# why the memcmp in the real code is defense in depth, not the primary key
run_one cache-verify-dropped src/bp_bpe.c \
    'if (ent->len == n \&\& ent->hash == h \&\& !memcmp(ent->bytes, p, n))' \
    'if (ent->nids != 0)' diff

# id-invisible on the Qwen3 vocabulary (no token bridges long-s and a
# letter, so no merge crosses the moved boundary); the toy vocabulary in
# tests/data/toy-longs.json contains exactly such a bridge
run_one longs-contraction-dropped src/bp_scan.c \
    'if (t\[i + 1\] == 0xC5 \&\& n >= 3 \&\& t\[i + 2\] == 0xBF) return 3;' \
    '' longs

run_one added-tokens-dropped src/bp_encode.c \
    'if ((flags \& BP_RAW) || v->hdr.added_count == 0)' \
    'if (1)' diff

echo "== all mutations caught =="
