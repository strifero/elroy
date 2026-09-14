"""A byte-level BPE tokenizer, written from scratch. Chapter 2.

Three ideas, in the order they appear below:

1. Pre-tokenization. A regex chops text into "words" before BPE ever sees it, so
   merges never cross a word boundary. Our pattern is a code-aware variant of the
   GPT-4 pattern: digits are split one per token, and runs of spaces or tabs
   before a token stay together, so an 8-space Python indent is one token.

2. Byte level. Each word becomes its UTF-8 bytes, so the base alphabet is exactly
   256 symbols and any string is representable. No unknown token, ever.

3. Byte-pair encoding. Training repeatedly finds the most frequent adjacent pair
   of symbols in the corpus and merges it into a new symbol, vocab_size minus 256
   minus specials times. Encoding a word replays those merges in rank order.

The trainer works on the unique-word frequency table rather than the raw text,
keeps pair counts up to date incrementally, and pulls the best pair from a heap,
which is what makes a 32k vocabulary trainable on a few hundred MB in minutes
rather than days. The encoder caches per-word results because words repeat.

Nothing here depends on torch. `regex` (not `re`) is needed for \\p{L} and \\p{N}.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from typing import Iterable, Iterator

import regex

# Code-aware pre-tokenization. Alternatives are tried left to right.
#   contractions            's, 't, 're ...
#   words                   optional single leading non-letter (usually a space), then letters
#   single digit            so numbers never become opaque tokens like "2023"
#   punctuation run         optional leading spaces, then symbols, then trailing newlines
#   newlines                a run of \r\n (with any leading spaces)
#   indentation             spaces/tabs that are followed by a non-space (the run minus its last
#                           char, which attaches to the next word as its leading space)
#   whitespace              anything left
PRETOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"|[ \t]*[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|[ \t]+(?![ \t])"
    r"|\s+"
)

SPECIAL_TOKENS = [
    "<|endoftext|>",     # document boundary
    "<|fim_prefix|>",    # fill-in-the-middle, chapter 4
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|pad|>",
    "<|system|>",        # chat template, chapter 7
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]
NUM_RESERVED_SPECIALS = 16  # ids vocab_size-16 .. vocab_size-1 are specials (some unused)


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], vocab_size: int,
                 specials: list[str] | None = None, pattern: str = PRETOKEN_PATTERN):
        self.vocab_size = vocab_size
        self.pattern = pattern
        self._re = regex.compile(pattern)
        self.merges = merges
        # rank: (left_id, right_id) -> new_id. Lower new_id means earlier merge, higher priority.
        self.ranks: dict[tuple[int, int], int] = {pair: 256 + i for i, pair in enumerate(merges)}
        # vocab: id -> bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        specials = specials or SPECIAL_TOKENS
        base = vocab_size - NUM_RESERVED_SPECIALS
        self.special_ids: dict[str, int] = {s: base + i for i, s in enumerate(specials)}
        self.id_to_special: dict[int, str] = {v: k for k, v in self.special_ids.items()}
        self._special_re = regex.compile("(" + "|".join(regex.escape(s) for s in specials) + ")")
        self._cache: dict[bytes, list[int]] = {}
        self.cache_limit = 2_000_000   # per-word memo; lower it in memory-tight worker processes
        self.eot = self.special_ids["<|endoftext|>"]

    # ---------------------------------------------------------------- encoding
    def _bpe(self, word: bytes) -> list[int]:
        ids = list(word)
        while len(ids) > 1:
            # find the adjacent pair with the best (lowest) rank
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            ids[best_i:best_i + 2] = [best_rank]
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with no special-token handling (specials are treated as plain text)."""
        out: list[int] = []
        cache = self._cache
        for m in self._re.finditer(text):
            w = m.group().encode("utf-8")
            ids = cache.get(w)
            if ids is None:
                ids = self._bpe(w)
                if len(cache) < self.cache_limit:
                    cache[w] = ids
            out.extend(ids)
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text; occurrences of special-token strings become their ids."""
        if not allowed_special:
            return self.encode_ordinary(text)
        out: list[int] = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            sid = self.special_ids.get(chunk)
            if sid is not None:
                out.append(sid)
            else:
                out.extend(self.encode_ordinary(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.id_to_special:
                parts.append(self.id_to_special[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Human-readable form of one token, for printing."""
        if i in self.id_to_special:
            return self.id_to_special[i]
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "pattern": self.pattern,
                       "specials": list(self.special_ids), "merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path) as f:
            d = json.load(f)
        return cls([tuple(m) for m in d["merges"]], d["vocab_size"], d["specials"], d["pattern"])


# -------------------------------------------------------------------- training
def count_words(texts: Iterable[str], pattern: str = PRETOKEN_PATTERN) -> Counter:
    """Pre-tokenize every text and count how often each word (as bytes) occurs."""
    rx = regex.compile(pattern)
    counts: Counter = Counter()
    for t in texts:
        counts.update(m.group().encode("utf-8") for m in rx.finditer(t))
    return counts


def train_bpe(word_counts: Counter, vocab_size: int, log_every: int = 1000,
              log: "callable | None" = print) -> list[tuple[int, int]]:
    """Learn BPE merges from a word frequency table.

    Returns the merge list in order. Merge i creates token id 256 + i.
    """
    n_merges = vocab_size - 256 - NUM_RESERVED_SPECIALS
    words: list[list[int]] = [list(w) for w in word_counts]
    freqs: list[int] = list(word_counts.values())

    # pair -> total count, and pair -> set of word indices that contain it
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for a, b in zip(w, w[1:]):
            pair_counts[(a, b)] += f
            where[(a, b)].add(wi)

    # max-heap of (-count, pair). Entries go stale when counts change; we check on pop.
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg:
            continue  # stale entry
        if -neg < 2:
            break     # nothing left that repeats
        new_id = 256 + len(merges)
        merges.append(pair)
        a, b = pair
        touched: set[tuple[int, int]] = set()
        for wi in list(where[pair]):
            w = words[wi]
            f = freqs[wi]
            new_w = _merge_word(w, a, b, new_id)
            if len(new_w) == len(w):
                continue  # stale membership: the pair is no longer in this word
            old_pairs = Counter(zip(w, w[1:]))
            new_pairs = Counter(zip(new_w, new_w[1:]))
            for pc, n in old_pairs.items():
                pair_counts[pc] -= f * n
                touched.add(pc)
                if pc not in new_pairs:
                    where[pc].discard(wi)
            for pc, n in new_pairs.items():
                pair_counts[pc] += f * n
                where[pc].add(wi)
                touched.add(pc)
            words[wi] = new_w
        pair_counts[pair] = 0
        del where[pair]
        for pc in touched:
            c = pair_counts[pc]
            if c > 0:
                heapq.heappush(heap, (-c, pc))
        if log and len(merges) % log_every == 0:
            log(f"merge {len(merges)}/{n_merges}: {pair} -> {new_id}  count={-neg}  heap={len(heap)}")
    return merges


def _merge_word(w: list[int], a: int, b: int, new_id: int) -> list[int]:
    """Replace every adjacent (a, b) in w with new_id, left to right."""
    out: list[int] = []
    i = 0
    n = len(w)
    while i < n:
        if i < n - 1 and w[i] == a and w[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return out


def iter_pretokens(text: str, pattern: str = PRETOKEN_PATTERN) -> Iterator[str]:
    """Just the regex split, for inspection in the chapter."""
    for m in regex.finditer(pattern, text):
        yield m.group()
