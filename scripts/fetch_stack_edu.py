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

One process is limited by Python's GIL to about 750 files/s (gunzip, decode,
parquet). To use more cores, run several copies that split the shard numbers
between them:

    for i in 0 1 2 3; do
      python scripts/fetch_stack_edu.py --target-bytes 32e9 --process-index $i --process-count 4 &
    done

Resumable: each output shard is a deterministic slice of the ranked index, and a
shard is only renamed into place once it is complete. Rerun after a crash and it
skips every shard that already exists.

Licensing: 82% of Stack-Edu Python is "no license detected" and 18% carries a
permissive license. The Stack v2 and StarCoder2 trained on exactly this basis and
we follow it. Pass --permissive-only if you want the stricter set (about a fifth
of the data).
"""

import argparse
import glob
import gzip
import http.client
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

S3_HOST = "softwareheritage.s3.amazonaws.com"
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


_local = threading.local()


def _conn() -> http.client.HTTPSConnection:
    """One persistent HTTPS connection per thread.

    A fresh TLS handshake per file costs more than the file itself (a 2KB blob
    takes a few milliseconds to transfer and about 100 ms to negotiate), so each
    thread keeps its connection open and reuses it for every request.
    """
    c = getattr(_local, "conn", None)
    if c is None:
        c = http.client.HTTPSConnection(S3_HOST, timeout=30)
        _local.conn = c
    return c


def fetch(blob_id: str, encoding: str, retries: int = 4) -> str | None:
    for attempt in range(retries):
        try:
            c = _conn()
            c.request("GET", "/content/" + blob_id)
            r = c.getresponse()
            raw = r.read()
            if r.status == 404:
                return None
            if r.status != 200:
                raise OSError(f"HTTP {r.status}")
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as g:
                data = g.read()
            try:
                return data.decode(encoding or "utf-8")
            except (UnicodeDecodeError, LookupError):
                return data.decode("utf-8", errors="replace")
        except (http.client.HTTPException, OSError, EOFError):
            # drop the connection and back off; the next attempt reconnects
            try:
                _local.conn.close()
            except Exception:  # noqa: BLE001
                pass
            _local.conn = None
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
    ap.add_argument("--process-index", type=int, default=0, help="this process handles shards i where i %% count == index")
    ap.add_argument("--process-count", type=int, default=1, help="run N copies with index 0..N-1 to use N cores")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    # The ranked index is built once and saved with one row group per shard, so
    # each fetch process reads only the rows it needs instead of holding all 24M
    # rows (about 9GB in memory) at once.
    ranked_path = os.path.join(OUT_DIR, f"ranked-index-{'permissive' if args.permissive_only else 'all'}"
                                        f"-min{args.min_score}-seed{args.seed}.parquet")
    if not os.path.exists(ranked_path):
        index = load_index(args.permissive_only, args.min_score, args.seed)
        if args.limit_docs:
            index = index.slice(0, args.limit_docs)
        print(f"ranked index: {index.num_rows:,} files, "
              f"{index.column('length_bytes').to_numpy().sum() / 1e9:.1f} GB on disk", flush=True)
        pq.write_table(index.drop_columns(["length_bytes"]), ranked_path + ".tmp",
                       row_group_size=args.shard_docs, compression="zstd")
        os.replace(ranked_path + ".tmp", ranked_path)
        del index
    ranked = pq.ParquetFile(ranked_path)

    def fetched_bytes() -> int:
        files = glob.glob(os.path.join(OUT_DIR, "shard-*.parquet"))
        return sum(int(pq.read_table(f, columns=["nbytes"]).column("nbytes").to_numpy().sum()) for f in files)

    total_bytes = fetched_bytes()
    print(f"resuming: {total_bytes / 1e9:.2f} GB already fetched", flush=True)

    n_shards = ranked.num_row_groups
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for shard_i in range(args.process_index, n_shards, args.process_count):
            final = os.path.join(OUT_DIR, f"shard-{shard_i:04d}.parquet")
            if os.path.exists(final):
                continue
            if total_bytes >= args.target_bytes:
                break
            rows = ranked.read_row_group(shard_i).to_pylist()
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
            tmp = final + ".tmp"
            pq.write_table(table, tmp, compression="zstd")
            os.replace(tmp, final)
            shard_bytes = int(table.column("nbytes").to_numpy().sum()) if table.num_rows else 0
            # with several processes running, re-count what is on disk so every process
            # stops at the same target
            total_bytes = fetched_bytes() if args.process_count > 1 else total_bytes + shard_bytes
            dt = time.time() - t0
            print(f"shard {shard_i:04d}: {table.num_rows:,}/{len(rows):,} ok, "
                  f"{shard_bytes / 1e6:.0f} MB in {dt:.0f}s ({len(rows) / dt:.0f} files/s), "
                  f"total {total_bytes / 1e9:.2f} GB", flush=True)
    print("done")


if __name__ == "__main__":
    main()
