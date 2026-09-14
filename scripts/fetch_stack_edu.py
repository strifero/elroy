"""Chapter 1, step 2: turn the Stack-Edu index into text.

Stack-Edu ships metadata only. Each row names a file by its Software Heritage
blob id and gives it an educational-quality score from 0 to 5. The bytes live in
the public `softwareheritage` S3 bucket, one gzip-compressed object per blob,
readable over plain anonymous HTTPS:

    https://softwareheritage.s3.amazonaws.com/content/<blob_id>

This script reads the index, ranks files by score (5 first, then 4, then 3),
fetches them with a thread pool, and writes parquet shards that contain the
actual source text. It stops when it has collected --target-bytes of text.

    python scripts/fetch_stack_edu.py --target-bytes 32e9 --workers 64

Resumable: each output shard is a deterministic slice of the ranked index, and a
shard is only renamed into place once it is complete. Rerun after a crash and it
picks up at the first missing shard.

Licensing: 82% of Stack-Edu Python is "no license detected" and 18% carries a
permissive license. The Stack v2 and StarCoder2 trained on exactly this basis and
we follow it. Pass --permissive-only if you want the stricter set (about a fifth
of the data).
"""

import argparse
import glob
import gzip
import io
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

S3 = "https://softwareheritage.s3.amazonaws.com/content/"
INDEX_DIR = os.path.join("data", "raw", "stack-edu", "Python")
OUT_DIR = os.path.join("data", "raw", "stack-edu-text")


def load_index(permissive_only: bool, min_score: int, seed: int) -> pa.Table:
    files = sorted(glob.glob(os.path.join(INDEX_DIR, "*.parquet")))
    if not files:
        sys.exit(f"no index parquet under {INDEX_DIR}; run scripts/download_data.py first")
    cols = ["blob_id", "repo_name", "path", "src_encoding", "length_bytes", "int_score", "license_type"]
    t = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    score = t.column("int_score").to_numpy()
    keep = score >= min_score
    if permissive_only:
        keep &= t.column("license_type").to_numpy() == "permissive"
    # Skip the extremes: tiny files are boilerplate, huge ones are data dumps or vendored code.
    length = t.column("length_bytes").to_numpy()
    keep &= (length >= 200) & (length <= 100_000)
    t = t.filter(pa.array(keep))
    # Rank: higher score first, random within a score so every shard is a fair sample.
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(t.num_rows), -t.column("int_score").to_numpy()))
    return t.take(pa.array(order))


def fetch(blob_id: str, encoding: str, retries: int = 4) -> str | None:
    url = S3 + blob_id
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as g:
                data = g.read()
            try:
                return data.decode(encoding or "utf-8")
            except (UnicodeDecodeError, LookupError):
                return data.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError, EOFError):
            time.sleep(1.5 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-bytes", type=float, default=32e9)
    ap.add_argument("--shard-docs", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--permissive-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit-docs", type=int, default=0, help="stop after this many docs (testing)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    index = load_index(args.permissive_only, args.min_score, args.seed)
    if args.limit_docs:
        index = index.slice(0, args.limit_docs)
    print(f"ranked index: {index.num_rows:,} files, "
          f"{index.column('length_bytes').to_numpy().sum() / 1e9:.1f} GB on disk", flush=True)

    total_bytes = 0
    done_shards = sorted(glob.glob(os.path.join(OUT_DIR, "shard-*.parquet")))
    for f in done_shards:
        total_bytes += int(pq.read_table(f, columns=["nbytes"]).column("nbytes").to_numpy().sum())
    print(f"resuming: {len(done_shards)} shards, {total_bytes / 1e9:.2f} GB already fetched", flush=True)

    n_shards = (index.num_rows + args.shard_docs - 1) // args.shard_docs
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for shard_i in range(len(done_shards), n_shards):
            if total_bytes >= args.target_bytes:
                break
            rows = index.slice(shard_i * args.shard_docs, args.shard_docs).to_pylist()
            t0 = time.time()
            texts = list(pool.map(lambda r: fetch(r["blob_id"], r["src_encoding"]), rows))
            keep = [(r, t) for r, t in zip(rows, texts) if t]
            table = pa.table({
                "blob_id": [r["blob_id"] for r, _ in keep],
                "repo_name": [r["repo_name"] for r, _ in keep],
                "path": [r["path"] for r, _ in keep],
                "score": [r["int_score"] for r, _ in keep],
                "license_type": [r["license_type"] for r, _ in keep],
                "text": [t for _, t in keep],
                "nbytes": [len(t.encode("utf-8")) for _, t in keep],
            })
            tmp = os.path.join(OUT_DIR, f"shard-{shard_i:04d}.parquet.tmp")
            final = tmp[:-4]
            pq.write_table(table, tmp, compression="zstd")
            os.replace(tmp, final)
            shard_bytes = int(table.column("nbytes").to_numpy().sum()) if table.num_rows else 0
            total_bytes += shard_bytes
            dt = time.time() - t0
            print(f"shard {shard_i:04d}: {table.num_rows:,}/{len(rows):,} ok, "
                  f"{shard_bytes / 1e6:.0f} MB in {dt:.0f}s ({len(rows) / dt:.0f} files/s), "
                  f"total {total_bytes / 1e9:.2f} GB", flush=True)
    print("done")


if __name__ == "__main__":
    main()
