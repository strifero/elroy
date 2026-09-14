# The plan

This is the working plan for Elroy, written before the first line of model code. It records what we decided and why, so that when a later chapter does something odd you can trace it back to a reason. It will be edited as reality intervenes; the edits are part of the story.

## Goal

Train a language model from scratch that can answer basic programming questions and write short Python programs, and document the whole process so that someone with one or two GPUs can reproduce it.

Teaching comes first, capability second. Every choice below was made by asking "does this make the guide clearer or the model better, and is the trade worth it?"

## What "from scratch" means here

The tokenizer, the model, the training loop, the distributed wiring, the sampler and the fine-tuning code are all written by hand in plain PyTorch. We do not use `transformers`, `Trainer`, `accelerate`, `tiktoken` or `tokenizers`.

We do use libraries for things that are not the lesson: `pyarrow` and `datasets` to read parquet files from Hugging Face, `numpy` for the token shards on disk, `regex` for the tokenizer's pre-tokenization pattern (Python's `re` lacks the Unicode classes we need), and `huggingface_hub` to publish the result.

## The model

Elroy-350M is a Llama-style decoder: RMSNorm, rotary position embeddings, SwiGLU feed-forward, no biases, tied input and output embeddings.

| | Elroy-mini | Elroy-350M |
|---|---|---|
| Layers | 8 | 24 |
| Width | 512 | 1024 |
| Heads | 8 | 16 |
| FFN hidden | 1408 | 3072 |
| Context | 1024 | 2048 |
| Vocabulary | 32,768 | 32,768 |
| Parameters | about 42M | about 360M |
| Training tokens | 1B | 20B |
| Hardware | one small GPU, hours | two RTX A4500, about 9 days |

We started this plan at GPT-2 small (124M, learned positions, GELU). We moved to the Llama recipe once the goal became "as good at coding as this hardware allows" rather than "the simplest possible model." RMSNorm, RoPE and SwiGLU each cost about a paragraph to explain and are what every current small model uses, so a reader who finishes this guide recognizes the shape of real models.

## Why 350M and 20B tokens

Three things move coding ability at this scale, in order: how many tokens of good code the model sees, how good that code is, and how many parameters it has. The Codex paper's scaling curve puts 300M parameters at about 13% on HumanEval and 2.5B at 21%, so parameters matter, but tokens and quality matter more per dollar.

360M parameters is the largest model that trains comfortably with plain data-parallel on two 20GB cards. Weights, gradients and AdamW state in mixed precision are about 6GB per GPU, leaving room for a large enough micro-batch to keep utilization high without activation checkpointing. Going to 1B would require sharding the optimizer state across GPUs (FSDP), which is another concept to teach, and would need 50B+ tokens to be worth it, which is beyond what two workstation GPUs can do in a month.

Compute check: training FLOPs are about 6 x parameters x tokens, so 6 x 360M x 20B is 4.3e19. Two A4500s are rated at 94 TFLOPS of dense bf16 each; the Chapter 0 smoke test measures 78. At a realistic 35% model FLOPs utilization of the measured figure that is about 55 TFLOPS sustained, so about 220 hours, or 9 days. If the validation loss still has slope at 20B tokens, the run resumes to 30B. That is a decision for day 8, not now.

## Data

About 20B tokens, roughly 65% Python and 35% English.

| Source | Share | Why |
|---|---|---|
| Stack-Edu, Python subset | 45% | Classifier-filtered educational code, the same recipe SmolLM2 used. The quality core. |
| StarCoderData, Python subset | 20% | Permissively licensed code from The Stack, for volume and variety. |
| FineWeb-Edu | 25% | So the model can read a question. Educational filter keeps it dense. |
| Cosmopedia v2 | 10% | Synthetic textbook-style explanations; small models learn explanation from it. |

The last 10% of training is an anneal: the learning rate decays to zero while the mix shifts to the highest-quality Python plus a slice of instruction-style data. MiniCPM and SmolLM both showed this phase is worth far more than its token count.

Stack Overflow would be ideal question-and-answer data. It is CC-BY-SA, which would force share-alike terms on the weights, so it stays out.

Stack-Edu is an index, not a corpus: each row is a Software Heritage blob id plus a quality score, and the file contents are fetched from the public `softwareheritage` S3 bucket (anonymous HTTPS works, no AWS account needed). Chapter 1 writes the fetcher. About 82% of Stack-Edu's Python is "no license detected" and 18% carries a permissive license; this is the same basis The Stack v2 and StarCoder2 trained on, and we follow it, with a `--permissive-only` flag for anyone who wants the stricter set. StarCoderData is gated behind a click-through on Hugging Face; accept the terms and set `HF_TOKEN`.

Raw parquet lands in `data/raw/` and is tokenized into uint16 shards under `data/shards/`. The shards are about 40GB for 20B tokens.

## Tokenizer

A byte-level BPE with a 32k vocabulary, trained by our own code on a sample of the mixed corpus. The pre-tokenization pattern keeps runs of spaces together (so an eight-space Python indent is one token, not eight) and splits digits individually. Chapter 2 measures tokens per line of Python before and after; that number is the whole argument for training your own tokenizer.

## Training objective

Next-token prediction, plus fill-in-the-middle on half of the code documents. FIM is three special tokens and a document shuffle, costs nothing at training time, and is what lets the model complete code inside a file rather than only at the end.

## Post-training

A supervised fine-tune on coding instructions: Evol-Instruct-Code and OSS-Instruct, filtered to Python and to examples that fit in 2048 tokens, plus a few hundred hand-written pairs for identity and beginner concept explanations. Loss is masked on the prompt. A preference-tuning (DPO) chapter is a stretch goal.

## Evaluation

HumanEval, HumanEval+ and MBPP pass@1 through a sandboxed executor, reported for the base and chat checkpoints. HellaSwag as an English sanity check. Per-domain validation loss (Python and English separately) throughout training. All numbers are published, including the disappointing ones.

Honest expectation: low-to-mid teens on HumanEval after SFT. Elroy will write `fizzbuzz`, reverse a linked list and read a CSV; it will not refactor your service.

## Hardware and where things run

The main run is on st-ai-1: two RTX A4500 (20GB each, Ampere, NVLink), AMD Ryzen 9 7900, 32GB RAM, 1TB NVMe, Ubuntu 22.04, driver 595, PyTorch 2.11 with CUDA 12.8 wheels. Other GPUs in the rack (an RTX 8000 and two V100s) do not join the main run, since data-parallel training moves at the speed of the slowest card. They run tokenizer training, the mini config, SFT experiments and evals in parallel so the main run is never interrupted.

Per-hardware tuning (micro-batch, attention backend, compile) is measured on the actual cards in Chapter 4, not copied from someone else's recipe.

## Release

The repo is the source of truth. Each chapter is also published as a blog post on strifetech.com as it is finished. Weights go to Hugging Face as `elroy-mini`, `elroy-350m-base` and `elroy-350m-chat`, Apache-2.0, with a model card that lists every data source and its license.

## Phases

| Phase | What | Effort |
|---|---|---|
| 0 | Environment on st-ai-1, repo scaffold, smoke test, dataset staging | one session |
| 1 | Tokenizer, model, training loop, FIM, validated on the mini config | three sessions |
| 2 | The 350M run | 9 to 11 days, unattended, daily check |
| 3 | Anneal, SFT, evals | two sessions |
| 4 | Release, model card, blog conversion | one session |

Docs are written alongside the code in each phase, not afterwards. The write-up is the product.

## Decision log

- 2026-09-14: Started. Chose pure PyTorch, GPT-2 small, 124M, general Q&A.
- 2026-09-14: Retargeted to programming questions and code generation. Python only for the published run; the data pipeline treats languages as a config list.
- 2026-09-14: Scaled to 350M / 20B tokens / Llama-style architecture / FIM / anneal. Ruled out 1B for this run (needs FSDP and 50B+ tokens).
- 2026-09-14: Phase 0 done. Measured 78 TFLOPS per A4500 and 41 GiB/s NVLink; revised the run estimate from 8 to 9 days. Dropped smollm-corpus python-edu (same blob-index scheme as Stack-Edu, older source); Stack-Edu Python is the sole educational code source.
- 2026-09-14: Chapter 1 findings. Stack-Edu Python is 25M files / 67GB, fetched by score via anonymous S3. StarCoderData carries `<reponame>/<filename>/<gh_stars>` header tags in 48% of files; strip them. Only Python-3-parseable files are kept (drops about 6% of code, mostly Python 2).
