"""Chapter 7, step 1: build the instruction-tuning set.

    python scripts/prepare_sft.py --tokenizer data/tokenizer.json --out data/sft

Sources, both permissively licensed and both on the Hub:

  ise-uiuc/Magicoder-OSS-Instruct-75K   (MIT)         problem/solution pairs generated
                                                       from real open-source snippets
  theblackcat102/evol-codealpaca-v1     (Apache-2.0)  Evol-Instruct style coding tasks

plus sft/handwritten.jsonl, a few dozen pairs we wrote ourselves (repeated): who
Elroy is, what it can and cannot do, and beginner concept explanations.

Filtering: keep examples whose reply contains a Python code block (or no code
block and a short prose answer), whose code parses under Python 3, and which
fit in 2048 tokens under the chat template. Everything is written as one
tokenized .npy of ids and one of loss masks, padded per example, plus a
manifest with the counts, so elroy.sft can memory-map it.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.chat import extract_code, format_example  # noqa: E402
from elroy.data import parses_py3  # noqa: E402
from elroy.tokenizer import Tokenizer  # noqa: E402


def looks_python(reply: str) -> bool:
    if "```" not in reply:
        return False
    code = extract_code(reply)
    if not code.strip():
        return False
    fence_lang = reply.split("```", 1)[1].split("\n", 1)[0].strip().lower()
    if fence_lang and fence_lang not in ("python", "py", "python3"):
        return False
    return parses_py3(code)


def load_pairs(limit: int) -> list[tuple[str, str, str]]:
    from datasets import load_dataset
    pairs = []
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    for r in ds:
        if r.get("lang", "python") == "python":
            pairs.append(("oss_instruct", r["problem"], r["solution"]))
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    for r in ds:
        pairs.append(("evol_codealpaca", r["instruction"], r["output"]))
    if limit:
        random.Random(0).shuffle(pairs)
        pairs = pairs[:limit]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--handwritten-repeats", type=int, default=4,
                    help="the identity/concept set is small; repeat it so it is not drowned out")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--handwritten", default="sft/handwritten.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = Tokenizer.load(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    examples: list[tuple[list[int], list[int]]] = []

    def add(source: str, user: str, assistant: str, check: bool) -> None:
        c = counts.setdefault(source, {"seen": 0, "kept": 0})
        c["seen"] += 1
        if check and not looks_python(assistant):
            return
        ex = format_example(tok, user.strip(), assistant.strip(), max_len=args.max_len)
        if ex is None:
            return
        c["kept"] += 1
        examples.append(ex)

    for source, user, assistant in load_pairs(args.limit):
        add(source, user, assistant, check=True)

    hw = args.handwritten
    if os.path.exists(hw):
        rows = [json.loads(l) for l in open(hw) if l.strip()]
        for _ in range(args.handwritten_repeats):
            for r in rows:
                add("handwritten", r["user"], r["assistant"], check=False)
    else:
        print(f"no {hw}; skipping the handwritten set")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = int(len(examples) * args.val_frac)
    for split, exs in (("val", examples[:n_val]), ("train", examples[n_val:])):
        lengths = np.array([len(ids) for ids, _ in exs], dtype=np.int32)
        ids = np.zeros((len(exs), args.max_len), dtype=np.uint16)
        mask = np.zeros((len(exs), args.max_len), dtype=np.uint8)
        for i, (x, m) in enumerate(exs):
            ids[i, : len(x)] = x
            mask[i, : len(m)] = m
        np.save(os.path.join(args.out, f"{split}_ids.npy"), ids)
        np.save(os.path.join(args.out, f"{split}_mask.npy"), mask)
        np.save(os.path.join(args.out, f"{split}_len.npy"), lengths)
        print(f"{split}: {len(exs):,} examples, mean {lengths.mean():.0f} tokens, "
              f"{int(mask.sum()):,} supervised tokens")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"counts": counts, "train": len(examples) - n_val, "val": n_val,
                   "max_len": args.max_len}, f, indent=2)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
