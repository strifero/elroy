"""Chapter 2: train Elroy's tokenizer.

    python scripts/train_tokenizer.py --sample-mb 400 --out data/tokenizer.json

Samples the raw sources in the training mix proportions, counts pre-tokens with
all CPU cores, learns the BPE merges, saves the tokenizer, and reports
bytes-per-token on a held-out slice of Python and English so the chapter has
numbers to talk about.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.data import SOURCES, sample_docs  # noqa: E402
from elroy.tokenizer import Tokenizer, count_words, train_bpe  # noqa: E402

MIX = {"python_edu": 0.45, "python_stack": 0.20, "fineweb_edu": 0.25, "cosmopedia": 0.10}


def _count(chunk: list[str]) -> Counter:
    return count_words(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mb", type=float, default=400)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--out", default="data/tokenizer.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    t0 = time.time()
    train_docs: list[str] = []
    held: dict[str, list[str]] = {}
    for name, share in MIX.items():
        src = SOURCES[name]
        docs = sample_docs(src, int(args.sample_mb * 1e6 * share), seed=args.seed)
        if not docs:
            sys.exit(f"no data for {name}; run the chapter 1 scripts first")
        held[name] = docs[-200:]           # last 200 docs of the sample are held out
        train_docs.extend(docs[:-200])
        print(f"{name:13s} {len(docs):7,d} docs, {sum(len(d) for d in docs) / 1e6:6.1f} MB", flush=True)
    print(f"sampled in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    chunks = [train_docs[i::args.workers] for i in range(args.workers)]
    with Pool(args.workers) as pool:
        counts = pool.map(_count, chunks)
    word_counts: Counter = Counter()
    for c in counts:
        word_counts.update(c)
    total_words = sum(word_counts.values())
    print(f"{len(word_counts):,} unique pre-tokens from {total_words:,} occurrences "
          f"in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    merges = train_bpe(word_counts, args.vocab_size, log_every=2000)
    print(f"trained {len(merges)} merges in {time.time() - t0:.0f}s", flush=True)

    tok = Tokenizer(merges, args.vocab_size)
    tok.save(args.out)
    print(f"saved {args.out}")

    report = {}
    for name, docs in held.items():
        nbytes = sum(len(d.encode("utf-8")) for d in docs)
        ntok = sum(len(tok.encode_ordinary(d)) for d in docs)
        nlines = sum(d.count("\n") + 1 for d in docs)
        report[name] = {"bytes_per_token": nbytes / ntok, "tokens_per_line": ntok / nlines}
        print(f"{name:13s} held-out: {nbytes / ntok:.2f} bytes/token, {ntok / nlines:.1f} tokens/line")
    with open(os.path.splitext(args.out)[0] + ".report.json", "w") as f:
        json.dump({"sample_mb": args.sample_mb, "vocab_size": args.vocab_size,
                   "unique_pretokens": len(word_counts), "held_out": report}, f, indent=2)


if __name__ == "__main__":
    main()
