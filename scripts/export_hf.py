"""Chapter 8: package a checkpoint for release.

    python scripts/export_hf.py --ckpt checkpoints/elroy-350m/latest.pt --out release/elroy-350m-base
    python scripts/export_hf.py --ckpt checkpoints/elroy-350m-chat/latest.pt --out release/elroy-350m-chat --chat
    python scripts/export_hf.py ... --push strifero/elroy-350m-base

Writes a self-contained folder:

    model.safetensors     the weights, bf16, tied embedding stored once
    config.json           the ModelConfig fields plus which variant this is
    tokenizer.json        Elroy's own tokenizer file (merges + specials + pattern)
    README.md             the model card, filled in from the training manifest
    elroy_min.py          model + tokenizer + generate in one file with no dependency on this repo

Nothing here uses the transformers library; the card explains how to load with
elroy_min.py and torch alone.
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.sample import load_model  # noqa: E402

CARD = """---
license: apache-2.0
language:
- en
- code
tags:
- python
- code
- from-scratch
- elroy
---

# {name}

{name} is a {params}M-parameter Llama-style decoder trained from scratch on about 20B tokens of educational Python and English by Strife Technologies, as a fully documented teaching project. Every line of the tokenizer, model, training loop and fine-tuning code is in the repository, with a chapter explaining it:

https://github.com/strifero/elroy

{variant_blurb}

## Use

No `transformers` needed. `elroy_min.py` in this repo holds the model, the tokenizer and a sampler in one file:

```python
import torch
from elroy_min import load, generate

model, tok = load(".")          # this folder
ids = tok.encode({prompt_example})
out = generate(model, ids, max_new_tokens=128, temperature=0.2, top_p=0.95, stop_ids={{tok.eot}})
print(tok.decode(out))
```

## Training

| | |
|---|---|
| Parameters | {params}M ({layers} layers, width {width}, {heads} heads, context {ctx}) |
| Tokens | about 20B: 45% Stack-Edu Python, 20% StarCoderData Python, 25% FineWeb-Edu, 10% Cosmopedia-v2 |
| Objective | next token, plus fill-in-the-middle on half of the code |
| Hardware | two NVIDIA RTX A4500 (20GB), about 6 days |
| Tokenizer | own byte-level BPE, 32,768 entries, code-aware pre-tokenization |

## Evaluation

{eval_table}

## Data and licenses

Stack-Edu (HuggingFaceTB, Python subset; files with a permissive or no detected license, as in The Stack v2), StarCoderData (bigcode, permissive), FineWeb-Edu (ODC-By), Cosmopedia-v2 (ODC-By). Fine-tuning data: Magicoder OSS-Instruct (MIT), evol-codealpaca-v1 (Apache-2.0), and a handwritten set in the repository.

## Limitations

This is a small model. It writes short, mostly correct Python functions and explains beginner concepts. It makes confident mistakes, knows little outside Python, has no knowledge of recent libraries, and has not been trained to refuse anything. Run what it writes before you rely on it.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--evals", default=None, help="JSON file of {benchmark: score} to put in the card")
    ap.add_argument("--push", default=None, help="Hub repo id to upload to (requires hf auth login)")
    args = ap.parse_args()

    from safetensors.torch import save_file

    device = torch.device("cpu")
    model, cfg = load_model(args.ckpt, device)
    os.makedirs(args.out, exist_ok=True)
    name = args.name or os.path.basename(args.out.rstrip("/"))

    # tied embedding: save one copy, record the tie in config
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    mcfg = dict(cfg["model"])
    mcfg["variant"] = "chat" if args.chat else "base"
    mcfg["name"] = name
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(mcfg, f, indent=2)
    shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "elroy", "elroy_min.py"),
                os.path.join(args.out, "elroy_min.py"))

    evals = json.load(open(args.evals)) if args.evals else {}
    eval_table = "| Benchmark | pass@1 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in evals.items()) \
        if evals else "(see the repository)"
    if args.chat:
        blurb = ("This is the **chat** variant: fine-tuned on coding instructions with the template "
                 "`<|system|>...<|end|><|user|>...<|end|><|assistant|>`. Use `elroy_min.chat_prompt(question)` "
                 "to build prompts and stop on `<|end|>`.")
        prompt_example = 'chat_prompt("Write a function that checks whether a number is prime.")'
    else:
        blurb = ("This is the **base** variant: a text continuer. Prompt it with the start of a Python file "
                 "(a signature and docstring, a comment, an import). It also completes gaps with "
                 "`<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`.")
        prompt_example = '"def is_prime(n):\\n    \\"\\"\\"Return True if n is prime.\\"\\"\\"\\n"'
    card = CARD.format(name=name, params=round(model.num_params() / 1e6), variant_blurb=blurb,
                       prompt_example=prompt_example, layers=mcfg["n_layer"], width=mcfg["d_model"],
                       heads=mcfg["n_head"], ctx=mcfg["seq_len"], eval_table=eval_table)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(card)
    print(f"wrote {args.out}: {sum(v.numel() for v in state.values()) / 1e6:.1f}M params in safetensors")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
