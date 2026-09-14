"""Chapter 1, step 3: look at the data before training on it.

    python scripts/inspect_data.py                # summary table for every source
    python scripts/inspect_data.py --show 3       # also print 3 random documents per source
    python scripts/inspect_data.py --max-files 2  # sample only the first 2 parquet files per source
    python scripts/inspect_data.py --only python_edu python_stack

For each source this prints document count, total bytes, the length distribution,
and for the Python sources the fraction of files that Python's own parser accepts.
Numbers are extrapolated from the sampled files when --max-files is set.
"""

import argparse
import ast
import glob
import os
import random
import warnings

import numpy as np
import pyarrow.parquet as pq

# name -> (glob of parquet files, column holding the text, is_python)
SOURCES = {
    "python_edu":   ("data/raw/stack-edu-text/shard-*.parquet", "text", True),
    "python_stack": ("data/raw/starcoderdata/python/*.parquet", "content", True),
    "fineweb_edu":  ("data/raw/fineweb-edu/sample/10BT/*.parquet", "text", False),
    "cosmopedia":   ("data/raw/smollm-corpus/cosmopedia-v2/*.parquet", "text", False),
}


def parses(src: str) -> bool:
    try:
        warnings.simplefilter("ignore", SyntaxWarning)
        ast.parse(src)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--parse-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="subset of source names")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"{'source':14s} {'files':>6s} {'docs':>12s} {'GB':>7s} {'mean':>7s} {'p50':>7s} {'p90':>8s} {'p99':>8s} {'parses':>7s}")
    for name, (pattern, col, is_py) in SOURCES.items():
        if args.only and name not in args.only:
            continue
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"{name:14s} (no files under {os.path.dirname(pattern)})")
            continue
        sampled = files[: args.max_files] if args.max_files else files
        lengths = []
        parse_hits = parse_n = 0
        shown = 0
        for f in sampled:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096, columns=[col]):
                texts = batch.column(0).to_pylist()
                lengths.extend(len(t.encode("utf-8")) for t in texts)
                if is_py and parse_n < args.parse_sample:
                    for t in rng.sample(texts, min(len(texts), 64)):
                        parse_n += 1
                        parse_hits += parses(t)
                if shown < args.show:
                    t = rng.choice(texts)
                    print(f"\n----- {name} sample ({len(t)} chars) -----\n{t[:1200]}\n")
                    shown += 1
        scale = len(files) / len(sampled)
        L = np.array(lengths)
        print(f"{name:14s} {len(files):6d} {int(len(L) * scale):12,d} {L.sum() * scale / 1e9:7.2f} "
              f"{L.mean():7.0f} {np.percentile(L, 50):7.0f} {np.percentile(L, 90):8.0f} {np.percentile(L, 99):8.0f} "
              f"{(parse_hits / parse_n if parse_n else float('nan')):7.1%}")


if __name__ == "__main__":
    main()
