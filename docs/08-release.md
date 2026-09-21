# Chapter 8: Release

By the end of this chapter the two models are folders anyone can download and run with `torch` and one Python file, the model card says what is in them and what they cannot do, and there is a chat page on the LAN that Chapter 7's checkpoint answers from.

## What a release is

A checkpoint from Chapter 4 is a pickle of a dict: the weights, the optimizer state, the loader positions, the RNG states, 4.3GB, and it only loads if this repository's classes are importable. None of that belongs in a release. A release is the weights in a format that does not execute code when loaded, the configuration needed to rebuild the module structure, the tokenizer, and enough code to run all three without the training repository.

```bash
python scripts/export_hf.py --ckpt checkpoints/elroy-350m/base-final.pt --out release/elroy-350m-base
python scripts/export_hf.py --ckpt checkpoints/elroy-350m-chat/chat-final.pt --out release/elroy-350m-chat --chat --evals results/chat/evals.json
```

Each folder contains five files:

```
model.safetensors    721MB   the parameters in bf16, the tied embedding stored once
config.json                  n_layer, d_model, n_head, d_ff, seq_len, vocab_size, rope_theta, variant
tokenizer.json       413KB   merges, specials and the pre-tokenization pattern from Chapter 2
elroy_min.py                 model, tokenizer and sampler in one file, no dependency on this repo
README.md                    the model card
```

Three choices in there are worth a sentence each. The weights are bf16 because that is the precision they were trained in under autocast and it halves the download; the fp32 master copy in the checkpoint carries no information the model ever used in a forward pass. The embedding is stored once and `config.json` records the tie, so loading is `load_state_dict(strict=False)` plus one assignment. And the format is safetensors rather than a `.pt` file because a `.pt` is a pickle, and loading a pickle from the internet runs whatever code is in it; safetensors is a header and a byte array.

## elroy_min.py

`elroy/elroy_min.py` is the model, the tokenizer's encoder and decoder, and `generate`, in 215 lines, with no import from the `elroy` package. It is copied into every release folder. `tests/test_min.py` loads the same checkpoint through the training code and through `elroy_min` and asserts the logits are bit-identical, which is the test that matters: a from-scratch reimplementation of RoPE or RMSNorm that is off by a transpose produces plausible text and wrong numbers.

```python
from elroy_min import load, generate, chat_prompt
model, tok = load("release/elroy-350m-chat")
ids = tok.encode(chat_prompt("Write a function that returns the n-th Fibonacci number."))
print(tok.decode(generate(model, ids, stop_ids={tok.special_ids["<|end|>"], tok.eot})))
```

There is no KV cache in it, on purpose, for the reason Chapter 6 gives. Anyone who wants one has the attention code in front of them.

## The model card

`export_hf.py` writes the card from the config and a JSON of benchmark scores. It states the architecture and size, the token count and mix, the hardware and days, the tokenizer, the licenses of every dataset that touched the weights, and a limitations section that is the summary of Chapters 5 and 7 rather than boilerplate: confident mistakes in plausible code, little outside Python, no recent libraries, no refusals. A card that only lists what the model does well is advertising.

## Serving it

`scripts/serve.py` is the page in the screenshots and the API behind it:

```bash
python scripts/serve.py --ckpt checkpoints/elroy-350m-chat/chat-final.pt --port 8100
```

Standard library only. `GET /` returns a single-page chat client; `POST /v1/chat/completions` speaks the OpenAI chat format, streaming or not, so Open WebUI, `curl` and the `openai` package all work against it unchanged. The request's message list is laid out in the Chapter 7 template (system, then alternating user and assistant turns, then `<|assistant|>`), generation stops at `<|end|>`, and a lock serializes requests because there is no batching. `POST /reload` re-reads the checkpoint, which is how the page picked up each new SFT checkpoint while Chapter 7 was still training.

Two behaviours in the server are worth knowing about because they are policy, not model. It refuses to emit a stop token in the first three positions, after "hi" produced an empty reply (Chapter 7 explains why). And it drops empty assistant turns from the history before building the prompt, so one empty reply does not poison the rest of a conversation. Neither changes what the model knows; both change what a person sees, and a description of a model that leaves them out is describing something else.

## Publishing

```bash
python scripts/export_hf.py ... --push strifero/elroy-350m-base
python scripts/export_hf.py ... --push strifero/elroy-350m-chat
```

`--push` creates the Hub repository and uploads the folder. The weights are Apache-2.0, matching the code, and the card lists every dataset license so that a reader can make their own call about the "no license detected" share of Stack-Edu that Chapter 1 discusses.

Published on September 20, 2026: [strifero/elroy-350m-base](https://huggingface.co/strifero/elroy-350m-base) and [strifero/elroy-350m-chat](https://huggingface.co/strifero/elroy-350m-chat), 721MB each. The repository went public the same day.

## What to do next

The obvious sequel is the run this one measured the need for: every validation curve still had slope at 20B tokens, and Chapter 5's numbers say another 10B would have been worth a few points. Beyond that, in rough order of value per hour: a KV cache in `elroy_min.py`, a few hundred more handwritten examples (greetings, refusals, "I do not know"), a second language, and a 1B-parameter Elroy on four GPUs, which needs FSDP and is where this write-up stops being enough.
