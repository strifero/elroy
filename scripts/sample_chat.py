"""Chapter 7: a fixed set of questions through the chat checkpoint, for the write-up.

    python scripts/sample_chat.py --ckpt checkpoints/elroy-350m-chat/chat-final.pt > results/chat/samples.md

Same prompts every time; several are the ones the base model failed in Chapter 5.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elroy.chat import build_prompt  # noqa: E402
from elroy.sample import generate, load_model  # noqa: E402
from elroy.tokenizer import Tokenizer  # noqa: E402

PROMPTS = [
    ("who", "Who are you?"),
    ("limits", "What can you not do?"),
    ("prime_prose", "Write a Python function that checks whether a number is prime."),
    ("list_tuple", "What is the difference between a list and a tuple in Python?"),
    ("word_counts", "Write a function word_counts(text) that returns a dict of how many times each word appears in text, ignoring case."),
    ("csv", "How do I read one column from a CSV file in Python?"),
    ("france", "What is the capital of France?"),
    ("providence", "Tell me about Providence, Rhode Island."),
    ("error", "I get 'TypeError: can only concatenate str (not \"int\") to str' when I run print(\"Total: \" + 5). What is wrong?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, cfg = load_model(args.ckpt, device)
    tok = Tokenizer.load(args.tokenizer)
    stop = {tok.eot, tok.special_ids["<|end|>"]}
    print(f"# Chat samples from {args.ckpt} (temperature {args.temperature}, top-p 0.95, seed 0)\n")
    for name, q in PROMPTS:
        ids = tok.encode(build_prompt(q))
        out = generate(model, ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                       top_p=0.95, stop_ids=stop, device=device)
        print(f"## {name}\n\n**user:** {q}\n\n**elroy:** {tok.decode(out).strip()}\n")


if __name__ == "__main__":
    main()
