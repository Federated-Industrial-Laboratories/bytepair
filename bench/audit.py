#!/usr/bin/env python3
"""Benchmark driver for the audit run. See bench/audit.sh.

Methodology, identical for every tool measured:
  - documents: the corpus is split after any newline once at least 8192
    bytes have accumulated (the same rule the bytepair CLI bench applies;
    the document counts are asserted equal)
  - single-thread: encode every document in order; best wall-clock of 3
    rounds, and EVERY timed round uses a freshly constructed tokenizer
    instance, so any internal cache starts empty and may only warm within
    a round (the bytepair CLI bench constructs a fresh context per round
    for the same reason). Construction happens outside the timer.
  - multi-thread: the tool's own parallel path over the same documents
    (bytepair: pthreads CLI bench; HuggingFace and GigaToken: their batch
    APIs), all hardware threads; same fresh-instance-per-round rule
  - correctness is checked AFTER timing, on another fresh instance: token
    ids compared against the pinned HuggingFace `tokenizers` output, which
    is the definition of correct here
  - every failure is recorded as a result, never papered over

The fresh-instance rule exists because the first shakedown run measured a
40x inflated number for a cache-bearing tokenizer: the correctness pass had
pre-warmed its process-level cache with the exact benchmark documents.

Output: one JSON file per run under --out, plus a markdown table on stdout.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

def split_docs(data: bytes):
    """The CLI bench rule: split after a newline once >= 8192 bytes passed."""
    offs = [0]
    last = 0
    start = 0
    while True:
        i = data.find(b"\n", start)
        if i == -1:
            break
        if i - last >= 8192:
            offs.append(i + 1)
            last = i + 1
        start = i + 1
    if last < len(data):
        offs.append(len(data))
    return [data[offs[k]:offs[k + 1]].decode("utf-8")
            for k in range(len(offs) - 1)]

def best_of(n, fn):
    best = None
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None or dt < best else best
    return best

def best_of_fresh(n, build, run):
    """Best of n rounds; a fresh tokenizer per round, built outside the
    timer. Returns (best_seconds, mean_build_seconds)."""
    best, built = None, 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        obj = build()
        built += time.perf_counter() - t0
        t0 = time.perf_counter()
        run(obj)
        dt = time.perf_counter() - t0
        best = dt if best is None or dt < best else best
        del obj
    return best, built / n

def cpu_info():
    model, flags = "unknown", ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and model == "unknown":
                model = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not flags:
                fl = set(line.split(":", 1)[1].split())
                flags = " ".join(sorted(fl & {"avx", "avx2", "avx512f",
                                              "bmi2", "sse4_2"}))
    except OSError:
        pass
    return model, flags

def tool_versions():
    import importlib.metadata as md
    out = {}
    for pkg in ("tokenizers", "gigatoken", "bpe-qwen", "transformers"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out

def bench_bytepair(args, path, docs, hf_ids):
    r = subprocess.run([args.bytepair, "info", args.bpv],
                       capture_output=True, text=True, check=True)
    load_ms = float(re.search(r"open: ([\d.]+) ms", r.stdout).group(1))

    def cli_bench(threads):
        r = subprocess.run([args.bytepair, "bench", args.bpv, path,
                            str(threads)], capture_output=True, text=True,
                           check=True)
        m = re.search(r"(\d+) docs, \d+ thread\(s\): ([\d.]+) MB/s", r.stdout)
        return int(m.group(1)), float(m.group(2))

    ndocs, st = cli_bench(1)
    if ndocs != len(docs):
        raise RuntimeError(f"doc split mismatch: cli {ndocs} vs {len(docs)}")
    _, mt = cli_bench(os.cpu_count())

    enc = subprocess.run([args.bytepair, "encode", args.bpv, path],
                         capture_output=True, check=True)
    ids = [int(x) for x in enc.stdout.split()]
    return {"load_ms": load_ms, "st_mbs": round(st, 2),
            "mt_mbs": round(mt, 2), "mt_threads": os.cpu_count(),
            "whole_file_exact": ids == hf_ids}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytepair", required=True)
    ap.add_argument("--bpv", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", action="append", required=True,
                    help="name:path, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    model, flags = cpu_info()
    run = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
                     .isoformat(timespec="seconds"),
        "cpu": model, "cpu_flags": flags, "threads": os.cpu_count(),
        "kernel": platform.release(), "python": platform.python_version(),
        "versions": tool_versions(),
        "bytepair_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True).stdout.strip() or None,
        "methodology": "docs split after newline at >=8192 bytes; best of 3; "
                       "ids compared against pinned HuggingFace tokenizers",
        "corpora": {},
    }

    for spec in args.corpus:
        name, path = spec.split(":", 1)
        raw = open(path, "rb").read()
        docs = split_docs(raw)
        size_mb = len(raw) / 1e6
        cres = {"bytes": len(raw), "docs": len(docs),
                "sha256": hashlib.sha256(raw).hexdigest(), "tools": {}}
        run["corpora"][name] = cres

        # reference: correctness oracle computed once (untimed), then timed
        # on fresh instances like every other tool
        hf = Tokenizer.from_file(args.tokenizer)
        hf_doc_ids = [hf.encode(d, add_special_tokens=False).ids
                      for d in docs]
        hf_ids = hf.encode(raw.decode("utf-8"),
                           add_special_tokens=False).ids
        del hf
        hf_build = lambda: Tokenizer.from_file(args.tokenizer)  # noqa: E731
        st, hf_load = best_of_fresh(
            3, hf_build,
            lambda t: [t.encode(d, add_special_tokens=False) for d in docs])
        mt, _ = best_of_fresh(
            3, hf_build,
            lambda t: t.encode_batch(docs, add_special_tokens=False))
        cres["tools"]["hf-tokenizers"] = {
            "load_ms": round(hf_load * 1e3, 1),
            "st_mbs": round(size_mb / st, 2),
            "mt_mbs": round(size_mb / mt, 2), "mt_threads": os.cpu_count(),
            "tokens": sum(len(x) for x in hf_doc_ids)}

        try:
            r = bench_bytepair(args, path, docs, hf_ids)
            cres["tools"]["bytepair"] = r
        except Exception as e:  # noqa: BLE001 - audit records, never hides
            cres["tools"]["bytepair"] = {"error": repr(e)}

        try:
            import gigatoken
            gt_build = lambda: gigatoken.Tokenizer(args.tokenizer)  # noqa: E731
            st, gt_load = best_of_fresh(
                3, gt_build, lambda t: [t.encode(d) for d in docs])
            mt, _ = best_of_fresh(3, gt_build,
                                  lambda t: t.encode_batch(docs))
            gt = gt_build()
            match = sum(1 for d, want in zip(docs, hf_doc_ids)
                        if list(gt.encode(d)) == want)
            del gt
            cres["tools"]["gigatoken"] = {
                "load_ms": round(gt_load * 1e3, 1),
                "st_mbs": round(size_mb / st, 2),
                "mt_mbs": round(size_mb / mt, 2),
                "mt_threads": os.cpu_count(),
                "exact_docs": f"{match}/{len(docs)}"}
        except Exception as e:  # noqa: BLE001
            cres["tools"]["gigatoken"] = {"error": repr(e)}

        try:
            from bpe_qwen.bpe_qwen import QwenTokenizer
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                import json as _json
                tok = _json.load(open(args.tokenizer))
                # bpe-qwen needs vocab.json + merges.txt
                _json.dump(tok["model"]["vocab"],
                           open(f"{td}/vocab.json", "w"), ensure_ascii=False)
                with open(f"{td}/merges.txt", "w") as f:
                    f.write("#version: 0.2\n")
                    for a, b in tok["model"]["merges"]:
                        f.write(f"{a} {b}\n")
                bq_build = lambda: QwenTokenizer(td)  # noqa: E731
                st, bq_load = best_of_fresh(
                    3, bq_build, lambda t: [t.encode(d) for d in docs])
                bq = bq_build()
                match = sum(1 for d, want in zip(docs, hf_doc_ids)
                            if list(bq.encode(d)) == want)
                del bq
            cres["tools"]["bpe-qwen"] = {
                "load_ms": round(bq_load * 1e3, 1),
                "st_mbs": round(size_mb / st, 2), "mt_mbs": None,
                "exact_docs": f"{match}/{len(docs)}"}
        except Exception as e:  # noqa: BLE001
            cres["tools"]["bpe-qwen"] = {"error": repr(e)}

    os.makedirs(args.out, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-",
                  (model.split("@")[0] + "-" +
                   run["timestamp"][:10]).lower()).strip("-")
    out_path = os.path.join(args.out, slug + ".json")
    json.dump(run, open(out_path, "w"), indent=1)
    print(f"\nresults: {out_path}\n")

    for name, cres in run["corpora"].items():
        print(f"### {name} ({cres['bytes']} bytes, {cres['docs']} docs)")
        print("| tool | load | 1 thread | all threads | exact vs reference |")
        print("|---|---|---|---|---|")
        for tool, r in cres["tools"].items():
            if "error" in r:
                print(f"| {tool} | error | | | {r['error'][:60]} |")
                continue
            exact = r.get("exact_docs",
                          "yes" if r.get("whole_file_exact")
                          else "reference" if tool == "hf-tokenizers"
                          else "NO")
            mt = f"{r['mt_mbs']} MB/s" if r.get("mt_mbs") else "n/a"
            print(f"| {tool} | {r['load_ms']} ms | {r['st_mbs']} MB/s "
                  f"| {mt} | {exact} |")
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
