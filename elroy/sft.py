"""Supervised fine-tuning. Chapter 7.

    python -m elroy.sft --base checkpoints/elroy-350m/latest.pt --data data/sft --out checkpoints/elroy-350m-chat

Same model, same loss, three differences from pretraining:

  1. The data is (prompt, reply) pairs laid out with the chat template.
  2. The loss is masked to the reply tokens (elroy.model.Elroy.forward takes a
     loss_mask), so the model is not trained to reproduce questions.
  3. A much smaller learning rate for a couple of epochs. The base model already
     knows Python; this just teaches it the format and the manners.

Examples are padded to the longest example in each batch; batches are drawn
from length buckets so padding stays small. Checkpoints use the same format as
pretraining, so elroy.sample and scripts/eval_code.py load them unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from elroy.model import Elroy, ModelConfig


class SFTData:
    def __init__(self, path: str, split: str, seed: int):
        self.ids = np.load(os.path.join(path, f"{split}_ids.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(path, f"{split}_mask.npy"), mmap_mode="r")
        self.lens = np.load(os.path.join(path, f"{split}_len.npy"))
        self.rng = np.random.default_rng(seed)
        self.order = np.argsort(self.lens)   # length buckets: neighbours have similar lengths

    def __len__(self) -> int:
        return len(self.lens)

    def batches(self, batch_size: int, shuffle: bool = True):
        """Yield (ids, targets, mask) batches, each trimmed to its longest example."""
        starts = np.arange(0, len(self.order), batch_size)
        if shuffle:
            self.rng.shuffle(starts)
        for s in starts:
            idx = np.sort(self.order[s: s + batch_size])
            L = int(self.lens[idx].max())
            x = torch.from_numpy(np.asarray(self.ids[idx, :L], dtype=np.int64))
            m = torch.from_numpy(np.asarray(self.mask[idx, :L], dtype=np.float32))
            # predict token t+1 from tokens <= t; the mask applies to the *target* position
            yield x[:, :-1], x[:, 1:], m[:, 1:]


@torch.no_grad()
def evaluate(model, data: SFTData, device, batch_size: int, dtype) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y, m in data.batches(batch_size, shuffle=False):
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, loss = model(x, y, loss_mask=m)
        tot += loss.item() * m.sum().item()
        n += m.sum().item()
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pretraining checkpoint")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    ck = torch.load(args.base, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Elroy(ModelConfig(**cfg["model"])).to(device)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    raw = model
    if not args.no_compile and device.type == "cuda":
        model = torch.compile(model, dynamic=True)   # batch shapes vary, so allow dynamic shapes

    train, val = SFTData(args.data, "train", args.seed), SFTData(args.data, "val", args.seed)
    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(raw.param_groups(args.weight_decay), lr=args.lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"SFT: {len(train):,} train / {len(val):,} val examples, {total_steps} steps "
          f"({args.epochs} epochs), lr {args.lr}", flush=True)
    print(f"val loss before: {evaluate(model, val, device, args.batch_size, dtype):.4f}", flush=True)

    step, micro, t0 = 0, 0, time.time()
    model.train()
    while step < total_steps:
        for x, y, m in train.batches(args.batch_size):
            if step >= total_steps:
                break
            # warmup then cosine to zero
            lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
                0.5 * args.lr * (1 + math.cos(math.pi * (step - args.warmup) / max(1, total_steps - args.warmup)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y, loss_mask=m)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:      # step the optimizer every grad_accum micro-batches
                continue
            gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr, "grad_norm": round(float(gn), 3)}
            if step % 10 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f} lr {lr:.2e} gn {float(gn):.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                rec["val"] = round(evaluate(model, val, device, args.batch_size, dtype), 4)
                print(f"  val loss {rec['val']:.4f}", flush=True)
                torch.save({"model": raw.state_dict(), "config": cfg, "step": step, "sft_args": vars(args)},
                           os.path.join(args.out, "latest.pt"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
    print("done")


if __name__ == "__main__":
    main()
