"""Chapter 5: a fixed set of prompts through a checkpoint, for the write-up.

    python scripts/sample_base.py --ckpt checkpoints/elroy-350m/base-final.pt > results/base/samples.md

Same prompts every time so milestones can be compared side by side.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elroy.sample import CODE_STOPS, generate, load_model, truncate_at  # noqa: E402
from elroy.tokenizer import Tokenizer  # noqa: E402

PROMPTS = [
    ("fizzbuzz", "def fizzbuzz(n):\n"),
    ("is_prime", "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is a prime number.\"\"\"\n"),
    ("reverse_list", "class Node:\n    def __init__(self, val, next=None):\n        self.val = val\n        self.next = next\n\n\ndef reverse_linked_list(head):\n"),
    ("word_count", "def word_counts(text: str) -> dict[str, int]:\n    \"\"\"Count how many times each word appears in text, ignoring case.\"\"\"\n"),
    ("csv_column", "import csv\n\n\ndef read_column(path: str, column: str) -> list[str]:\n    \"\"\"Return every value in the named column of a CSV file.\"\"\"\n"),
    ("question", "Q: What is the difference between a list and a tuple in Python?\nA:"),
    ("english", "The city of Providence, Rhode Island, is"),
    ("prose_prompt", "Write a Python function that checks whether a number is prime.\n"),
]

FIM = [
    ("fim_body", "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n", "        out.append(str(i))\n    return out\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, cfg = load_model(args.ckpt, device)
    tok = Tokenizer.load(args.tokenizer)
    print(f"# Samples from {args.ckpt} (temperature {args.temperature}, top-p 0.95, seed 0)\n")
    for name, prompt in PROMPTS:
        ids = tok.encode_ordinary(prompt)
        out = generate(model, ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                       top_p=0.95, stop_ids={tok.eot}, device=device)
        text = tok.decode(out)
        if name not in ("question", "english"):
            text = truncate_at(text, CODE_STOPS)
        print(f"## {name}\n\n```\n{prompt}{text}\n```\n")
    for name, prefix, suffix in FIM:
        s = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        ids = tok.encode(s)
        out = generate(model, ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                       top_p=0.95, stop_ids={tok.eot, tok.special_ids["<|fim_suffix|>"], tok.special_ids["<|fim_middle|>"]}, device=device)
        middle = tok.decode(out)
        print(f"## {name} (fill in the middle; the model wrote the marked part)\n\n```\n{prefix}### >>>\n{middle}### <<<\n{suffix}```\n")


if __name__ == "__main__":
    main()
