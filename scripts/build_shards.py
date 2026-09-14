"""Chapter 4, step 1: turn cleaned text into token shards.

    python scripts/build_shards.py --tokenizer data/tokenizer.json --out data/shards
    python scripts/build_shards.py --max-docs 20000 --out data/shards-dev   # small, for development

For every source in elroy.data.SOURCES this reads the parquet files, applies
the chapter 1 cleaning rules, tokenizes each document, appends <|endoftext|>,
and writes the resulting stream as uint16 .npy shards of about 100M tokens:

    data/shards/<source>/val.npy           the first --val-tokens tokens, held out
    data/shards/<source>/train-0000.npy    the rest, in order
    data/shards/manifest.json              token counts per source

Code documents are transformed for fill-in-the-middle with probability
--fim-rate before tokenization: two random cut points split the document into
prefix, middle and suffix, and the pieces are rearranged with the FIM special
tokens so the model learns to produce a middle given both sides. Half of the
FIM documents use the PSM order and half SPM; see fim_transform.

Tokenization runs on all cores in chunks of a few thousand documents. The
per-worker tokenizer cache makes this mostly dictionary lookups, a few MB/s per
core in pure Python.
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.data import SOURCES, clean_code, clean_text, source_files  # noqa: E402
from elroy.tokenizer import Tokenizer  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

_tok: Tokenizer | None = None
_cfg: dict = {}
CHUNK_DOCS = 4000


def fim_transform(text: str, rng: random.Random, spm_rate: float) -> str:
    """Rearrange a document into FIM form using the special-token *strings*.

    PSM:  <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
    SPM:  <|fim_prefix|><|fim_suffix|>{suffix}<|fim_middle|>{prefix}{middle}

    The tokenizer turns the special strings into their ids in encode(). Cut points
    are uniform over characters, which is what the original FIM paper did.
    """
    n = len(text)
    a, b = sorted(rng.sample(range(n + 1), 2))
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    if rng.random() < spm_rate:
        return f"<|fim_prefix|><|fim_suffix|>{suffix}<|fim_middle|>{prefix}{middle}"
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"


def _init(tok_path: str, cfg: dict) -> None:
    global _tok, _cfg
    _tok = Tokenizer.load(tok_path)
    _tok.cache_limit = 200_000   # 24 workers x a 2M-entry memo was what ran the box out of RAM
    _cfg = cfg


def _encode_group(job: tuple[str, int, int, int, str, bool, int, int]) -> tuple[str, int, np.ndarray, int]:
    """Tokenize a slice of one parquet row group.

    The unit of work is CHUNK_DOCS documents, not a whole file or even a whole
    row group: Stack-Edu's row groups are 100,000 files, and a worker holding
    100,000 decoded strings plus their token lists is a couple of GB. The first
    version of this script used whole row groups and the OOM killer took it out
    at 30GB. Re-reading a row group to take a slice of it costs a little
    decompression; it is nothing next to tokenization.
    """
    path, group, offset, count, column, is_code, max_docs, seed = job
    rng = random.Random(seed)
    clean = clean_code if is_code else clean_text
    out: list[np.ndarray] = []
    n_docs = 0
    texts = pq.ParquetFile(path).read_row_group(group, columns=[column]).slice(offset, count).column(0).to_pylist()
    for t in texts:
        t = clean(t)
        if t is None:
            continue
        if is_code and rng.random() < _cfg["fim_rate"]:
            t = fim_transform(t, rng, _cfg["fim_spm_rate"])
            ids = _tok.encode(t)                    # specials become ids
        else:
            ids = _tok.encode_ordinary(t)           # any literal "<|endoftext|>" stays text
        ids.append(_tok.eot)
        out.append(np.asarray(ids, dtype=np.uint16))
        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
    return path, group, (np.concatenate(out) if out else np.zeros(0, dtype=np.uint16)), n_docs


class ShardWriter:
    def __init__(self, out_dir: str, shard_tokens: int, val_tokens: int):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.val_tokens = val_tokens
        self.buf: list[np.ndarray] = []
        self.buffered = 0
        self.n_shards = 0
        self.total = 0
        self.val_done = False

    def add(self, arr: np.ndarray) -> None:
        self.buf.append(arr)
        self.buffered += len(arr)
        self.total += len(arr)
        limit = self.val_tokens if not self.val_done else self.shard_tokens
        while self.buffered >= limit:
            self._flush(limit)
            limit = self.val_tokens if not self.val_done else self.shard_tokens

    def _flush(self, n: int) -> None:
        cat = np.concatenate(self.buf)
        chunk, rest = cat[:n], cat[n:]
        if not self.val_done:
            np.save(os.path.join(self.out_dir, "val.npy"), chunk)
            self.val_done = True
        else:
            np.save(os.path.join(self.out_dir, f"train-{self.n_shards:04d}.npy"), chunk)
            self.n_shards += 1
        self.buf = [rest] if len(rest) else []
        self.buffered = len(rest)

    def close(self) -> None:
        if self.buffered:
            self._flush(self.buffered)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--fim-spm-rate", type=float, default=0.5)
    ap.add_argument("--max-docs", type=int, default=0, help="per row group (first 2 groups per file), for dev shards")
    ap.add_argument("--max-files", type=int, default=0, help="per source, for dev shards")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = {"fim_rate": args.fim_rate, "fim_spm_rate": args.fim_spm_rate}
    manifest = {"tokenizer": args.tokenizer, "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate, "sources": {}}
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer, cfg)) as pool:
        for name, src in SOURCES.items():
            if args.only and name not in args.only:
                continue
            files = source_files(src)
            if args.max_files:
                files = files[: args.max_files]
            if not files:
                print(f"{name}: no files, skipping")
                continue
            t0 = time.time()
            writer = ShardWriter(os.path.join(args.out, name), args.shard_tokens, args.val_tokens)
            jobs = []
            for f in files:
                pf = pf_meta = pq.ParquetFile(f)
                groups = range(pf.num_row_groups)
                if args.max_docs:
                    groups = range(min(2, len(groups)))
                for g in groups:
                    n_rows = pf_meta.metadata.row_group(g).num_rows
                    for off in range(0, n_rows, CHUNK_DOCS):
                        jobs.append((f, g, off, min(CHUNK_DOCS, n_rows - off), src.column, src.is_code,
                                     args.max_docs, args.seed + len(jobs)))
            # Shuffle the row groups (seeded) so the token stream, and therefore the val
            # split at its head, is a mix of files rather than the first file in order.
            # Stack-Edu shards are fetched best-score-first, so without this the val
            # set would be all score-5 files and training would see the scores in order.
            random.Random(args.seed).shuffle(jobs)
            docs = 0
            # imap keeps job order, so shards are deterministic for a given seed
            for i, (path, group, arr, n) in enumerate(pool.imap(_encode_group, jobs)):
                writer.add(arr)
                docs += n
                if i % 100 == 0 or i == len(jobs) - 1:
                    print(f"  {name}: {i + 1}/{len(jobs)} chunks, {docs:,} docs, "
                          f"{writer.total / 1e9:.3f}B tokens", flush=True)
            writer.close()
            dt = time.time() - t0
            manifest["sources"][name] = {"docs": docs, "tokens": writer.total,
                                         "train_shards": writer.n_shards, "val_tokens": args.val_tokens}
            print(f"{name}: {docs:,} docs, {writer.total / 1e9:.3f}B tokens, {writer.n_shards} shards, "
                  f"{dt:.0f}s ({writer.total / dt / 1e6:.2f}M tok/s)", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
