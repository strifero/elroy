# Chapter 1: Data

By the end of this chapter you have about 100GB of raw text on disk, split into four sources, and you have looked at it closely enough to know what is in it. Two of the four sources had surprises that would have quietly damaged the model if we had skipped this chapter.

## What we are looking for

Elroy's job is to write and explain Python. The model can only learn what is in its training data, so the data has to contain three things: a lot of Python, enough English to read a question and explain an answer, and, ideally, text that already looks like the output we want (explanations with code in them). Everything else is noise that costs compute.

We also want the data to be legally clean enough to release the weights under Apache-2.0, and small enough to download on a home connection. The final mix is about 20B tokens.

## The four sources

| Source | Hub id | Share | What it is |
|---|---|---|---|
| `python_edu` | `HuggingFaceTB/stack-edu`, Python | 45% | Python files from The Stack v2, scored 0 to 5 for educational quality by a classifier. We keep 3 and up, highest first. |
| `python_stack` | `bigcode/starcoderdata`, python | 20% | Permissively licensed, deduplicated Python from The Stack v1, the StarCoder training data. Variety and volume. |
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu`, 10BT sample | 25% | Web text filtered for educational value. This is how the model learns English. |
| `cosmopedia` | `HuggingFaceTB/smollm-corpus`, cosmopedia-v2 | 10% | Synthetic textbook-style articles. Small models pick up the explaining register from it. |

Licenses: FineWeb-Edu and Cosmopedia are ODC-By. StarCoderData is permissive code only and is gated behind a click-through on the Hub. Stack-Edu is discussed below. Stack Overflow, which would be ideal question-and-answer data, is CC-BY-SA; the share-alike clause would flow into the weights, so it is out.

## Step 1: download

```bash
python scripts/download_data.py          # about 65GB from the Hub
python scripts/download_data.py --mini   # one shard per source, for the mini config
```

StarCoderData will refuse with "requires approval" until you have accepted its terms on the dataset page while logged in, and run `hf auth login` once. The script tells you when that is the problem.

What lands under `data/raw/`:

```
stack-edu/Python/          5 parquet files, 2.4GB. An INDEX, see step 2.
starcoderdata/python/      20 of 59 shards, 20GB
fineweb-edu/sample/10BT/   14 parquet files, 27GB
smollm-corpus/cosmopedia-v2/  10 of 104 shards, 11GB
```

## Step 2: Stack-Edu is an index, not a corpus

This was the first surprise. Open a Stack-Edu parquet file and there is no text in it:

```
blob_id: f76b3b840b3db0b3d8c980dc620327745383a006
repo_name: 17BTEC005/virat-kohli
path: /rohit sharma.py
length_bytes: 135
int_score: 3
license_type: no_license
```

Each row is a pointer into Software Heritage, the archive The Stack v2 was built from. The bytes live in a public S3 bucket, one gzip-compressed object per blob, and anonymous HTTPS works:

```
https://softwareheritage.s3.amazonaws.com/content/<blob_id>
```

So step 2 is a fetcher. `scripts/fetch_stack_edu.py` loads all five index files (23.7M Python files after dropping anything under 200 bytes or over 100KB), ranks them by score with a seeded shuffle inside each score, and pulls them down with a thread pool, writing parquet shards of 100,000 files each. It stops at a byte target. Shards are written to a temporary name and renamed when complete, so you can kill it and rerun; it resumes at the first missing shard.

```bash
TARGET=40e9 bash scripts/restart_fetch.sh 4      # four processes, 96 threads each
```

Our numbers, and a lesson. The first version opened a fresh HTTPS connection per file with `urllib` and managed about 780 files per second from one process, about 1.3MB/s of text because score-5 files are short (median 1.1KB); 32GB would have taken seven hours. Running four copies made it *slower*: four processes times 96 threads was 384 TLS handshakes in flight at once, and S3 latency climbed to over a second. The fix was one persistent `http.client.HTTPSConnection` per thread, reused for every request. A 2KB blob takes a few milliseconds to transfer and about 100ms to negotiate TLS for, so keep-alive is worth about 10x: each process now does 1,700 files per second and four of them together fetch 40GB in about an hour. The processes split the shard numbers between them (`--process-index`, `--process-count`) and read the ranked index from a parquet file with one row group per shard, so none of them holds the 24M-row index in memory.

Two things to know about this data. The score distribution is lopsided: of 25M Python files, 1% score 5, 21% score 4, 78% score 3. Fetching highest-first means our 40GB is all of the 5s and 4s plus a random slice of the 3s. And 82% of the files have no detected license. That is not the same as a restrictive license: The Stack v2 excludes non-permissive licenses up front, so "no_license" means the repository had no license file. StarCoder2 trained on exactly this set. We follow that precedent; `--permissive-only` keeps the 18% that carry an explicit permissive license if you would rather be strict.

## Step 3: look at it

```bash
python scripts/inspect_data.py --max-files 3 --show 1
```

The script reports document counts, bytes, the length distribution, and for the Python sources the fraction of files that Python 3's own parser accepts. That last column found the second surprise.

```
source          files         docs      GB    mean     p50      p90      p99  parses
python_edu          1      100,000    0.16    1609    1066     2996     9749   91.4%
python_stack       20    4,361,580   20.70    4747    1893    10542    42549   47.7%
fineweb_edu        14   10,182,666   48.64    4776    2958     9010    35295    nan%
cosmopedia         10    3,762,890   14.11    3750    3454     5624     8096    nan%
```

Less than half of StarCoderData parses. Sampling the failures, the top reason was not Python 2, it was this:

```
<reponame>MTES-MCT/sparte
from rest_framework_gis import serializers
```

StarCoderData ships with the StarCoder training-format header baked into the text: `<reponame>`, `<filename>` and `<gh_stars>` tags on the first line or two of 48% of files. They are not Python. Left in, the model would learn to start every file with `<reponame>` and the tokenizer would waste merges on them. Stripping the header with one regex takes the parse rate from 47.7% to 94.4%.

The remaining failures are mostly Python 2 (`print "x"`, `except E, e:`) plus a handful of tab-and-space mixes and byte-order marks. We drop anything that does not parse under Python 3. It is a blunt filter and it costs about 6% of the code, but a model that is only ever shown valid Python 3 is a model whose output is more likely to be valid Python 3.

Stack-Edu's 91% parse rate on the score-5 shard has the same cause in a different proportion: the highest-scored files skew toward beginner scripts, and a lot of beginner scripts on GitHub are Python 2. Same filter.

The English sources look as expected. FineWeb-Edu documents are web pages, median 3KB, with a long tail; Cosmopedia is uniform (p50 to p99 is 3.4KB to 8KB) because it is generated to a template. The one-document samples the script prints are worth reading. The FineWeb one we got was a 1994 report on pesticide use on Colombian flower farms, which is exactly the kind of thing a language model learns English from and exactly the kind of thing that has nothing to do with Python.

## Decisions this chapter produced

The cleaning rules that Chapter 2's tokenizer training and Chapter 4's shard builder both apply to code:

1. Strip the StarCoderData header: `^(?:<(?:reponame|filename|gh_stars)>[^\n]*\n)+`.
2. Drop files that do not parse with `ast.parse` under Python 3.
3. Drop files under 200 bytes or over 100KB (already applied at fetch time for Stack-Edu).

And for the budget, using rough bytes-per-token figures (about 3.5 for code with a code-aware tokenizer, about 4.5 for English): 40GB of Stack-Edu is about 11B tokens, 20GB of StarCoderData is about 5B after filtering, half of FineWeb-Edu's 10BT sample is 5B, and the 10 Cosmopedia shards are 3B. That is the 20B token, 65% Python mix from the plan, with headroom. Chapter 2 replaces the estimates with real token counts.

## Final inventory

`inspect_data.py` over everything on disk, after the fetch:

```
source          files         docs      GB    mean     p50      p90      p99  parses
python_edu        155   15,499,954   41.10    2652    1324     5922    20579   90.8%
python_stack       20    4,361,580   20.60    4724    1883    10505    42204   48.8%
fineweb_edu        14    9,672,101   45.97    4753    2951     8913    35323    nan%
cosmopedia         10    3,762,890   14.11    3750    3454     5627     8095    nan%
```

The `python_stack` parse rate is before the header strip (94% after, per the sample above); `python_edu` is after the size filter. 122GB of text, 15.5M Python files from Stack-Edu, and the `parses` column is what the shard builder will enforce.

## Next

Chapter 2 builds the tokenizer, trains it on a sample of this corpus, and measures how many tokens a line of Python costs.
