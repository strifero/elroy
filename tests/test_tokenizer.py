"""Tokenizer checks. Run with:  python tests/test_tokenizer.py

1. The incremental trainer must produce exactly the merges a naive
   recount-everything trainer produces (same tie-break: highest count, then
   lowest pair).
2. encode/decode must round-trip arbitrary text, including special tokens
   and text that is not valid UTF-8 after slicing.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.tokenizer import NUM_RESERVED_SPECIALS, Tokenizer, count_words, train_bpe  # noqa: E402


def naive_train(word_counts: Counter, vocab_size: int) -> list[tuple[int, int]]:
    words = {tuple(w): c for w, c in word_counts.items()}
    merges = []
    for i in range(vocab_size - 256 - NUM_RESERVED_SPECIALS):
        pc: Counter = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pc[p] += c
        if not pc:
            break
        pair, count = min(pc.items(), key=lambda kv: (-kv[1], kv[0]))
        if count < 2:
            break
        merges.append(pair)
        new_id = 256 + i
        merged: dict = {}
        for w, c in words.items():
            out, j = [], 0
            while j < len(w):
                if j < len(w) - 1 and (w[j], w[j + 1]) == pair:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(w[j])
                    j += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + c
        words = merged
    return merges


def main() -> None:
    here = os.path.dirname(__file__)
    text = open(os.path.join(here, "..", "elroy", "tokenizer.py")).read()
    text += open(os.path.join(here, "..", "README.md")).read()
    wc = count_words([text])
    vocab_size = 256 + NUM_RESERVED_SPECIALS + 300

    fast = train_bpe(wc, vocab_size, log=None)
    slow = naive_train(wc, vocab_size)
    assert fast == slow, "incremental trainer diverged from the naive reference"

    tok = Tokenizer(fast, vocab_size)
    samples = [
        "def add(a, b):\n    return a + b\n",
        "        if x == 1:\n\t\treturn 'yes'  # 2024\n",
        "unicode: café ünïcode 日本語 emoji 🚀\n",
        "<|endoftext|>after<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
        "",
    ]
    for s in samples:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, repr(s)
        assert all(0 <= i < vocab_size for i in ids)
    assert tok.encode("<|endoftext|>") == [tok.eot]
    assert tok.encode("<|endoftext|>", allowed_special=False) != [tok.eot]
    tok.save("/tmp/elroy-test-tok.json")
    tok2 = Tokenizer.load("/tmp/elroy-test-tok.json")
    assert tok2.encode(samples[0]) == tok.encode(samples[0])
    print(f"ok: {len(fast)} merges match reference, round-trips pass")


if __name__ == "__main__":
    main()
