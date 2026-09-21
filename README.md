# Elroy

Elroy is a small language model for Python, built from scratch in plain PyTorch and documented end to end so that you can build one too.

Everything here is hand written: the byte-level BPE tokenizer, the transformer, the training loop, the multi-GPU wiring, the sampler, and the supervised fine-tune that turns a text predictor into something that answers programming questions. There is no `transformers`, no `Trainer`, no hidden magic. The only libraries in the hot path are PyTorch and NumPy.

The published model, Elroy-350M, is a 24-layer Llama-style decoder trained on about 20B tokens of educational Python and English on two workstation GPUs over six days. It is not a replacement for a frontier coding assistant. It writes short, correct Python functions, explains beginner concepts, and completes code in the middle of a file. The point is that you can see every line that made it do that.

## Who this is for

You can read Python and you have used PyTorch at least once. You do not need to have trained a model before. Every chapter ends with a script you can run and a number you can check against ours.

## Hardware tiers

| Tier | What you have | What you can do |
|------|---------------|-----------------|
| Laptop | CPU only, or a small GPU | Chapters 0 to 4 in full, plus the `mini` config for a few hours. Elroy-mini writes `fizzbuzz`. |
| One GPU | A single 12GB+ card | Everything, at `mini` scale in an afternoon, or the full run in a few weeks. |
| Two GPUs | Two 20GB+ cards with NVLink or fast PCIe | The published Elroy-350M run in six days. |

If you only want to use Elroy, skip to Chapter 8, download the weights, and run `scripts/serve.py` to chat with it in a browser.

## Chapters

| # | Chapter | You end up with |
|---|---------|-----------------|
| 0 | [Setup](docs/00-setup.md) | A working environment and a GPU smoke test |
| 1 | [Data](docs/01-data.md) | A mixed Python and English corpus, inspected and understood |
| 2 | [Tokenizer](docs/02-tokenizer.md) | A hand-written BPE trainer and a 32k code-aware vocabulary |
| 3 | [Model](docs/03-model.md) | A Llama-style transformer whose parameter count matches the arithmetic |
| 4 | [Training loop](docs/04-training.md) | A resumable, multi-GPU training loop with fill-in-the-middle |
| 5 | [The run](docs/05-run.md) | The real run: loss curves, what broke, samples, and benchmark numbers |
| 6 | [Sampling](docs/06-sampling.md) | Temperature, top-k, top-p, and a REPL that runs the code it writes |
| 7 | [Teaching it to answer](docs/07-sft.md) | Supervised fine-tuning on coding instructions |
| 8 | [Release](docs/08-release.md) | Exported weights, a model card, an inference-only script, and a chat server |

The full plan, with the reasoning behind each decision, is in [docs/PLAN.md](docs/PLAN.md).

## Layout

```
elroy/
  elroy/            the package: tokenizer, model, data, train, sample, sft
  configs/          mini.yaml (laptop scale) and elroy-350m.yaml (the published run)
  scripts/          download, tokenize, evaluate, export, smoke tests
  docs/             one markdown chapter per phase
```

## License

Code is Apache-2.0. Released weights are Apache-2.0. Training data licenses are listed in Chapter 1 and in the model card.

Built by Sean Trifero, Strife Technologies.
