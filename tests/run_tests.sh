#!/bin/sh
# bytepair test runner.
#
# Environment (defaults in parentheses):
#   BP_BIN       bytepair binary            (build/bytepair)
#   BP_BPV       .bpv vocabulary            (build/qwen3.bpv)
#   BP_TOKENIZER reference tokenizer.json   (tests/data/fetched/qwen3-tokenizer.json)
#   BP_PY        python with `tokenizers`   (python3)
#   BP_CORPUS    bulk corpus, optional      (unset)
#   BP_QUICK     nonempty = quick differential only
set -e

BP_BIN="${BP_BIN:-build/bytepair}"
BP_BPV="${BP_BPV:-build/qwen3.bpv}"
BP_TOKENIZER="${BP_TOKENIZER:-tests/data/fetched/qwen3-tokenizer.json}"
BP_PY="${BP_PY:-python3}"

[ -x "$BP_BIN" ] || { echo "run_tests: $BP_BIN not built" >&2; exit 2; }
[ -f "$BP_BPV" ] || { echo "run_tests: $BP_BPV missing (run tools/bpv_convert.py)" >&2; exit 2; }
[ -f "$BP_TOKENIZER" ] || { echo "run_tests: $BP_TOKENIZER missing (run tests/fetch_qwen3.sh)" >&2; exit 2; }

echo "== loader unit tests =="
python3 tests/unit_loader.py "$BP_BIN" "$BP_BPV"

echo "== nfc conformance (C normalizer) =="
if [ -x build/nfc_conformance ]; then
    build/nfc_conformance tests/data/ucd9/NormalizationTest.txt
else
    echo "run_tests: build/nfc_conformance not built (make check builds it)" >&2
    exit 2
fi

echo "== robustness fuzz =="
python3 tests/fuzz_lite.py "$BP_BIN" "$BP_BPV"

echo "== long-s toy vocabulary =="
python3 tools/bpv_convert.py tests/data/toy-longs.json build/toy-longs.bpv --quiet
"$BP_PY" tests/test_longs_toy.py "$BP_BIN" build/toy-longs.bpv \
    tests/data/toy-longs.json

echo "== census verdicts on the toy =="
python3 tests/test_census.py "$BP_BIN" build/toy-longs.bpv

echo "== behavioral fingerprint =="
"$BP_PY" tests/test_fingerprint.py "$BP_TOKENIZER"

echo "== differential vs reference =="
if [ -n "$BP_QUICK" ]; then
    "$BP_PY" tests/differential.py --bytepair "$BP_BIN" --bpv "$BP_BPV" \
        --tokenizer "$BP_TOKENIZER" --quick
else
    "$BP_PY" tests/differential.py --bytepair "$BP_BIN" --bpv "$BP_BPV" \
        --tokenizer "$BP_TOKENIZER" ${BP_CORPUS:+--corpus "$BP_CORPUS"}
fi

echo "== differential vs reference, forced scalar =="
# the scalar scanner is the documented non-AVX2 fallback; it gets the same
# suite, always, so a break in it can never hide behind the SIMD path
if [ -n "$BP_QUICK" ]; then
    BYTEPAIR_FORCE_SCALAR=1 "$BP_PY" tests/differential.py \
        --bytepair "$BP_BIN" --bpv "$BP_BPV" --tokenizer "$BP_TOKENIZER" \
        --quick
else
    BYTEPAIR_FORCE_SCALAR=1 "$BP_PY" tests/differential.py \
        --bytepair "$BP_BIN" --bpv "$BP_BPV" --tokenizer "$BP_TOKENIZER" \
        ${BP_CORPUS:+--corpus "$BP_CORPUS"}
fi

echo "== all suites passed =="
