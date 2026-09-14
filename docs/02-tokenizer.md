# Chapter 2: The tokenizer

By the end of this chapter you have `elroy/tokenizer.py`, a byte-level BPE tokenizer whose trainer and encoder you wrote yourself, a 32,768-entry vocabulary learned from our corpus, and a number that justifies the whole exercise: how many tokens a line of Python costs.

## Why not characters, why not words

A model predicts one token at a time, so the choice of token decides two things: how long a sequence is, and how much each prediction has to carry. Characters make every sequence long (a 2048-token window would hold about 40 lines of Python) and force the model to spend capacity learning to spell. Words make sequences short but the vocabulary unbounded: every identifier ever written is a new word, and anything unseen at training time has no representation at all.

Byte-pair encoding sits between them. Start with the 256 possible bytes as the alphabet, count which adjacent pair occurs most often across the corpus, merge that pair into a new symbol, and repeat until the vocabulary is the size you want. Common words become single tokens, rare words become a few tokens, and anything at all can still be expressed as raw bytes. No unknown token, ever.

## Three ideas, three pieces of code

### Pre-tokenization

Before BPE sees anything, a regular expression chops the text into "words", and merges never cross a word boundary. Without this, BPE would happily learn tokens like `. The` or `);\n    def`, which are frequent but useless. The pattern is a code-aware variant of GPT-4's:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)     contractions
|[^\r\n\p{L}\p{N}]?\p{L}+        a word, with one leading space or symbol
|\p{N}                           ONE digit
|[ \t]*[^\s\p{L}\p{N}]+[\r\n]*   punctuation runs, with trailing newlines
|\s*[\r\n]+                      newlines
|[ \t]+(?![ \t])                 indentation
|\s+                             whatever is left
```

Two rules are there for code. Digits are split one per token, so `2048` is four tokens and the model never has to learn that `2048` and `2047` are neighbours. And runs of spaces or tabs are kept together, so an 8-space Python indent is one token instead of eight. GPT-2's tokenizer spends one token per space; a deeply nested Python file under GPT-2 is a third whitespace by token count.

```python
>>> list(iter_pretokens("        return x + 1  # 2024\n    def"))
['        ', 'return', ' x', ' +', ' ', '1', '  #', ' ', '2', '0', '2', '4', '\n', '    ', 'def']
```

### Byte level

Each pre-token becomes its UTF-8 bytes. The base alphabet is exactly 256 symbols, `é` is two of them, an emoji is four, and the tokenizer can represent any string because it can always fall back to bytes. Decoding concatenates the bytes of each token and decodes UTF-8 at the end, which is why a token can end in the middle of a multibyte character and nothing breaks.

### Merges

Training works on the frequency table of unique pre-tokens, not on the raw text. Our 800MB sample has 173 million pre-token occurrences but only 2.5 million unique ones, and BPE only cares about the counts. The loop is:

```python
pair = most frequent adjacent pair across all words (weighted by word frequency)
new_id = 256 + number of merges so far
for each word containing pair: replace every (a, b) with new_id, update pair counts
```

Done naively (recount every pair in every word after each merge) this is hours for a 32k vocabulary. Two things make it minutes. Pair counts are updated incrementally: only the words that contained the merged pair change, and each of those adjusts the counts of the handful of pairs it lost and gained. And the best pair comes off a heap rather than a scan; entries go stale as counts change, so the pop loop discards any entry whose recorded count no longer matches.

The incremental trainer is exactly the kind of code that is easy to get subtly wrong, so `tests/test_tokenizer.py` also contains the naive version and asserts that both produce the same merge list on a small corpus, tie-breaks included.

Encoding a word replays the merges: repeatedly find the adjacent pair with the lowest merge rank and merge it, until no pair is mergeable. Words repeat constantly, so the encoder caches the result per word and most of tokenizing a corpus is dictionary lookups.

## Special tokens

The last 16 ids are reserved for special tokens: `<|endoftext|>` marks document boundaries, three FIM tokens are used in Chapter 4, and five chat tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`) are used in Chapter 7. They are never produced by the BPE merges; `encode()` recognises their literal strings and substitutes the ids, and `encode_ordinary()` does not, which is what you want when tokenizing arbitrary text that might happen to contain the string `<|endoftext|>`.

## Training it

```bash
python scripts/train_tokenizer.py --sample-mb 800 --out data/tokenizer.json
```

The script samples the four sources in the training-mix proportions, counts pre-tokens on all cores, learns the merges, saves the tokenizer as JSON (the merge list is the entire model, about 500KB), and measures bytes-per-token on held-out documents.

### First attempt: 40MB

The first run used a 40MB sample to check the code. It trained 32,496 merges in 32 seconds and looked fine until you read the log: the last merges had counts of 28. A merge that fires 28 times in the sample is noise, not vocabulary. The tail of the vocabulary needs a sample big enough that the 32,000th most common pair is still common.

### The real one

```bash
python scripts/train_tokenizer.py --sample-mb 800 --out data/tokenizer.json
```

An 800MB sample in the mix proportions (360MB Stack-Edu, 160MB StarCoderData, 200MB FineWeb-Edu, 80MB Cosmopedia) has 173M pre-token occurrences and 2.48M unique pre-tokens. Sampling took 107 seconds (the code sources go through `ast.parse`), counting took 11 seconds on 24 cores, and the 32,496 merges took 371 seconds in pure Python. The tail of the vocabulary is now real: the 32,000th merge fired 562 times, the 20,000th 1,168 times, the 2,000th 25,414 times. The whole tokenizer is a 413KB JSON file.

## The number that matters

Bytes per token on 400 held-out documents each, our tokenizer against GPT-2's (50k vocabulary) and GPT-4's `cl100k_base` (100k vocabulary), both through `tiktoken`:

| | Elroy (32k) | GPT-2 (50k) | cl100k (100k) |
|---|---|---|---|
| Python, bytes per token | 3.69 | 2.16 | 3.80 |
| Python, tokens per line | 8.2 | 14.0 | 8.0 |
| English, bytes per token | 4.25 | 4.58 | 4.70 |
| English, tokens per line | 56.4 | 52.3 | 50.9 |

On Python, our 32k-entry tokenizer matches a 100k-entry one and uses 41% fewer tokens than GPT-2's. Most of that gap is indentation: GPT-2 spends one token per space. The price is English, where we are 8% worse than GPT-2, because two thirds of the sample was code and the vocabulary went where the data was. That is the trade the plan asked for. A 2048-token window holds about 250 lines of Python for Elroy and about 145 for a GPT-2-tokenized model of the same context length.

The 20B-token budget also changes shape. At 3.7 bytes per token the 41GB of Stack-Edu is 11B tokens and the 20GB of StarCoderData is 5B; at 4.3 the 46GB of FineWeb-Edu is 10B and the 14GB of Cosmopedia is 3B. Chapter 4 replaces those with the exact counts from the shards.

## Next

Chapter 3 builds the model that will read these tokens.
