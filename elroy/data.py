"""Reading and cleaning the raw sources. Shared by tokenizer training (chapter 2)
and shard building (chapter 4).

The cleaning rules come from chapter 1:
  * strip the StarCoderData `<reponame>/<filename>/<gh_stars>` header lines
  * keep only Python that parses under Python 3
  * drop code files under 200 bytes or over 100KB
"""

from __future__ import annotations

import ast
import glob
import os
import random
import warnings
from dataclasses import dataclass
from typing import Iterator

import pyarrow.parquet as pq
import regex

STARCODER_HEADER = regex.compile(r"^(?:<(?:reponame|filename|gh_stars)>[^\n]*\n)+")


@dataclass(frozen=True)
class Source:
    name: str
    pattern: str        # glob for parquet files under the repo root
    column: str         # column that holds the text
    is_code: bool


SOURCES: dict[str, Source] = {
    "python_edu":   Source("python_edu",   "data/raw/stack-edu-text/shard-*.parquet",      "text",    True),
    "python_stack": Source("python_stack", "data/raw/starcoderdata/python/*.parquet",      "content", True),
    "fineweb_edu":  Source("fineweb_edu",  "data/raw/fineweb-edu/sample/10BT/*.parquet",   "text",    False),
    "cosmopedia":   Source("cosmopedia",   "data/raw/smollm-corpus/cosmopedia-v2/*.parquet", "text",  False),
}


def parses_py3(src: str) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ast.parse(src)
            return True
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return False


def clean_code(text: str) -> str | None:
    """Apply the chapter 1 rules to one code file. Returns None to drop it."""
    text = STARCODER_HEADER.sub("", text, count=1)
    n = len(text)
    if n < 200 or n > 100_000:
        return None
    if not parses_py3(text):
        return None
    return text


def clean_text(text: str) -> str | None:
    text = text.strip()
    return text if len(text) >= 100 else None


def source_files(src: Source, root: str = ".") -> list[str]:
    return sorted(glob.glob(os.path.join(root, src.pattern)))


def iter_docs(src: Source, files: list[str] | None = None, clean: bool = True,
              batch_size: int = 1024) -> Iterator[str]:
    """Yield cleaned documents from a source, file by file, in order."""
    files = files if files is not None else source_files(src)
    fn = clean_code if src.is_code else clean_text
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[src.column]):
            for t in batch.column(0).to_pylist():
                if clean:
                    t = fn(t)
                    if t is None:
                        continue
                yield t


def sample_docs(src: Source, target_bytes: int, seed: int = 0, clean: bool = True) -> list[str]:
    """A roughly uniform random sample of documents totalling about target_bytes.

    Picks files in a seeded random order and takes a random 1/k slice of the rows
    of each, so the sample is not just the first file.
    """
    rng = random.Random(seed)
    files = source_files(src)
    if not files:
        return []
    rng.shuffle(files)
    fn = clean_code if src.is_code else clean_text
    out: list[str] = []
    total = 0
    per_file = max(target_bytes // max(len(files), 1), 2_000_000)
    for f in files:
        got = 0
        pf = pq.ParquetFile(f)
        n_groups = pf.num_row_groups
        for g in rng.sample(range(n_groups), n_groups):
            for t in pf.read_row_group(g, columns=[src.column]).column(0).to_pylist():
                if clean:
                    t = fn(t)
                    if t is None:
                        continue
                out.append(t)
                got += len(t)
                total += len(t)
                if got >= per_file or total >= target_bytes:
                    break
            if got >= per_file or total >= target_bytes:
                break
        if total >= target_bytes:
            break
    rng.shuffle(out)
    return out
