# Chapter 6: Sampling

By the end of this chapter you can talk to the base model: give it the start of a function and watch it finish it, give it the code on both sides of a gap and watch it fill the gap, and understand the three knobs that decide whether the output is boring, useful, or unhinged.

## What the model actually gives you

A forward pass on a sequence returns, at every position, a vector of 32,768 numbers: one logit per token in the vocabulary. Softmax turns the last position's logits into a probability distribution over what comes next. Generation is: pick a token from that distribution, append it, run the model again, repeat until a stop token or a length limit.

Everything in `elroy/sample.py` is about the word *pick*.

## The knobs

**Temperature** divides the logits before the softmax. At `T = 1` you sample from the model's own distribution. Below 1 the distribution sharpens: likely tokens become more likely, unlikely ones vanish. Above 1 it flattens. At `T = 0` (greedy) you always take the argmax, which is deterministic and has a characteristic failure: the model finds a locally plausible loop and never leaves it (`self.__return self.__return self.__return` in one of our early mini-model samples).

**Top-k** keeps only the `k` highest-probability tokens and renormalises. **Top-p** (nucleus sampling) keeps the smallest set of tokens whose probabilities sum to `p`, so the cutoff adapts: when the model is confident the set is one or two tokens, when it is unsure the set is large. Top-p is the one that matters in practice.

For code, the usual starting point is `T = 0.2`, `top_p = 0.95`: almost greedy, with just enough randomness to escape loops. The benchmark numbers in Chapter 5 use `T = 0` because that is how pass@1 is conventionally reported for small models.

## Stopping

The base model has no idea when a function is finished; it will happily start the next one. `CODE_STOPS` is a list of strings that mean "you are done": a new top-level `def` or `class`, an `if __name__`, a `print(` at column 0. Generation stops when the decoded output contains one, and `truncate_at` cuts the text at the first occurrence. `<|endoftext|>` also stops generation, since the model learned in pretraining that it ends documents.

## No KV cache, on purpose

Each new token re-runs the whole forward pass over everything so far, so generating `n` tokens costs `O(n^2)`. A KV cache keeps the attention keys and values from previous steps so each new token costs one position's worth of work. It is a well-understood optimisation and it is the first thing anyone adds to a real inference server. It is left out here because it doubles the size of the attention code for no conceptual gain, and because 350M parameters at a few hundred tokens is a second or two on a GPU without it.

## Fill in the middle

```bash
python -m elroy.sample --ckpt checkpoints/elroy-350m/latest.pt --fim "def add(a, b):\n" "\n\nprint(add(2, 3))\n"
```

This lays out the prompt exactly as Chapter 4's FIM transformation did during training, `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`, and the model produces the middle. Because half of the code the model saw in pretraining was in this form, it needs no fine-tuning to do it.

## The REPL

```bash
python -m elroy.sample --ckpt checkpoints/elroy-350m/latest.pt --repl
```

Type a prompt, get a continuation. The base model is a text continuer, not an assistant, so prompts should look like the start of a Python file: a signature and a docstring, a comment describing what comes next, an import. Chapter 7 is what turns this into something you can ask a question.

## Samples from Elroy-350M base

```
(filled in from the trained model)
```

## Next

Chapter 7 teaches the model the difference between continuing text and answering a question.
