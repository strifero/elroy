"""Generation. Chapter 6.

    python -m elroy.sample --ckpt checkpoints/elroy-350m/latest.pt --prompt "def fizzbuzz(n):"
    python -m elroy.sample --ckpt ... --repl          # interactive
    python -m elroy.sample --ckpt ... --fim "def add(a, b):\\n" "\\n\\nprint(add(2, 3))"

The model gives a distribution over the next token. Sampling turns that into a
choice, repeatedly:

  temperature   divide the logits by T before softmax. T < 1 sharpens (safer,
                more repetitive), T > 1 flattens (more varied, more mistakes).
  top-k         keep only the k most likely tokens, renormalise.
  top-p         keep the smallest set of tokens whose probability sums to p.
  greedy        T = 0: always the argmax. Deterministic, prone to loops.

For code, low temperature (0.2) with top-p 0.95 is the usual starting point;
HumanEval-style evals at pass@1 use exactly that.

There is no KV cache here. Each new token re-runs the full forward pass over
the prompt plus everything generated so far, so generation is O(n^2) in the
output length. For a 350M model and a few hundred tokens that is a second or
two on a GPU and it keeps the code honest; a cache is an optimisation, not a
concept, and it is left as an exercise the release chapter points to.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from elroy.model import Elroy, ModelConfig
from elroy.tokenizer import Tokenizer


def load_model(ckpt_path: str, device: torch.device) -> tuple[Elroy, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Elroy(ModelConfig(**cfg["model"])).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def generate(model: Elroy, ids: list[int], max_new_tokens: int = 256, temperature: float = 0.2,
             top_k: int = 0, top_p: float = 0.95, stop_ids: set[int] | None = None,
             stop_strs: list[str] | None = None, tok: Tokenizer | None = None,
             device: torch.device = torch.device("cpu")) -> list[int]:
    """Sample continuation token ids for a prompt (as ids). Returns only the new tokens."""
    stop_ids = stop_ids or set()
    out: list[int] = []
    x = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = model.cfg.seq_len
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -seq_len:])
        logits = logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < kth] = float("-inf")
            if 0 < top_p < 1:
                sorted_logits, order = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # drop everything after the cumulative mass passes p (keep at least one)
                drop = cum - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[drop] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        if nxt in stop_ids:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if stop_strs and tok is not None:
            text = tok.decode(out)
            if any(s in text for s in stop_strs):
                break
    return out


def truncate_at(text: str, stop_strs: list[str]) -> str:
    cut = len(text)
    for s in stop_strs:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


# Stop sequences for "complete this function" prompts: anything that starts a new
# top-level definition means the function body is finished.
CODE_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--fim", nargs=2, metavar=("PREFIX", "SUFFIX"), default=None)
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tok = Tokenizer.load(args.tokenizer)
    model, _ = load_model(args.ckpt, device)

    def run(prompt_ids: list[int], stops: set[int]) -> str:
        new = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
                       stop_ids=stops, device=device)
        return tok.decode(new)

    if args.fim:
        prefix, suffix = (s.encode().decode("unicode_escape") for s in args.fim)
        ids = tok.encode(f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>")
        middle = run(ids, {tok.eot})
        print(prefix + middle + suffix)
        return
    if args.prompt is not None:
        prompt = args.prompt.encode().decode("unicode_escape")
        print(prompt + run(tok.encode_ordinary(prompt), {tok.eot}))
        return
    if args.repl:
        print("Elroy REPL. Type a prompt; the model continues it. Ctrl-D to quit.")
        for line in sys.stdin:
            prompt = line.rstrip("\n").encode().decode("unicode_escape")
            print(prompt + run(tok.encode_ordinary(prompt), {tok.eot}))
            print("-" * 60)
        return
    ap.error("give --prompt, --fim, or --repl")


if __name__ == "__main__":
    main()
