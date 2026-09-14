"""The training loop. Chapter 4.

    python -m elroy.train --config configs/mini.yaml
    torchrun --standalone --nproc_per_node=2 -m elroy.train --config configs/elroy-350m.yaml

One optimizer step = grad_accum micro-batches per GPU, summed across GPUs, so
that tokens_per_step tokens contribute to every update regardless of how many
GPUs you have or how many sequences fit in memory at once.

Learning rate: linear warmup, cosine decay to min_lr over the main phase, then
a linear decay to zero over the final anneal_fraction of training while the
data mix switches to anneal_mix (the "WSD" shape that MiniCPM and SmolLM use).

Checkpoints hold everything needed to resume bit-for-bit: model, optimizer,
step, the loader position on every rank, and the RNG states. Kill the run at
any point and start it again with the same command; it picks up at the last
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from elroy.loader import MixLoader, load_val
from elroy.model import Elroy, ModelConfig


# ------------------------------------------------------------------- helpers
def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local}")
        torch.cuda.set_device(device)
    else:
        rank, world, local = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def lr_at(step: int, total: int, cfg: dict) -> float:
    lr, warmup = cfg["lr"], cfg["warmup_steps"]
    min_lr = lr * cfg["min_lr_ratio"]
    anneal_start = int(total * (1 - cfg["anneal_fraction"]))
    if step < warmup:
        return lr * (step + 1) / warmup
    if step < anneal_start:
        frac = (step - warmup) / max(1, anneal_start - warmup)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))
    frac = (step - anneal_start) / max(1, total - anneal_start)
    return min_lr * (1 - frac)


@torch.no_grad()
def evaluate(model, val_sets: dict[str, torch.Tensor], device, micro_batch: int, rank: int, world: int,
             dtype) -> dict[str, float]:
    model.eval()
    out = {}
    for name, windows in val_sets.items():
        mine = windows[rank::world]
        total, n = torch.zeros((), device=device), 0
        for i in range(0, len(mine), micro_batch):
            b = mine[i: i + micro_batch].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                _, loss = model(b[:, :-1], b[:, 1:])
            total += loss.float() * len(b)
            n += len(b)
        cnt = torch.tensor(float(n), device=device)
        if world > 1:
            dist.all_reduce(total)
            dist.all_reduce(cnt)
        out[name] = (total / cnt).item()
    model.train()
    return out


def save_checkpoint(path: str, raw_model, optimizer, step: int, tokens: int, loader_states: list,
                    cfg: dict, rng_states: list) -> None:
    tmp = path + ".tmp"
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "tokens": tokens, "loaders": loader_states, "config": cfg,
                "rng": rng_states}, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", default=None, help="override data.shards_dir")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for tests and sweeps)")
    ap.add_argument("--micro-batch", type=int, default=0, help="override train.micro_batch")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--peak-tflops", type=float, default=78.0, help="per-GPU bf16 matmul peak, for MFU")
    ap.add_argument("--eval-windows", type=int, default=64, help="val windows per source")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    if args.micro_batch:
        tcfg["micro_batch"] = args.micro_batch
    shards_dir = args.shards or dcfg["shards_dir"]

    rank, world, local, device = setup_distributed()
    master = rank == 0
    torch.manual_seed(tcfg["seed"] + rank)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[tcfg["dtype"]]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seq_len, micro = mcfg["seq_len"], tcfg["micro_batch"]
    tokens_per_micro = micro * seq_len * world
    assert tcfg["tokens_per_step"] % tokens_per_micro == 0, \
        f"tokens_per_step {tcfg['tokens_per_step']} must be a multiple of micro_batch*seq_len*world={tokens_per_micro}"
    grad_accum = tcfg["tokens_per_step"] // tokens_per_micro
    total_steps = tcfg["total_tokens"] // tcfg["tokens_per_step"]
    anneal_start = int(total_steps * (1 - tcfg["anneal_fraction"]))
    run_dir = os.path.join(args.out, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)

    model = Elroy(ModelConfig(**mcfg)).to(device)
    n_params = model.num_params()
    flops_per_token = model.flops_per_token()
    if master:
        print(f"{cfg['name']}: {n_params / 1e6:.1f}M params, {world} GPU(s), micro_batch {micro} x seq {seq_len} "
              f"x accum {grad_accum} = {tcfg['tokens_per_step']:,} tokens/step, {total_steps:,} steps, "
              f"anneal from step {anneal_start:,}", flush=True)

    optimizer = torch.optim.AdamW(model.param_groups(tcfg["weight_decay"]), lr=tcfg["lr"],
                                  betas=tuple(tcfg["betas"]), fused=device.type == "cuda")
    loader = MixLoader(shards_dir, dcfg["mix"], seq_len, micro, rank, world, tcfg["seed"], device)
    if "anneal_mix" in dcfg:
        loader.check_mix(dcfg["anneal_mix"])
    if master:
        avail = loader.tokens_available()
        print("tokens available per source (this rank): " +
              ", ".join(f"{k} {v / 1e9:.2f}B" for k, v in avail.items()), flush=True)
    val_sets = {n: load_val(shards_dir, n, seq_len, args.eval_windows) for n in loader.names}

    # ---- resume
    step, tokens_seen = 0, 0
    latest = os.path.join(run_dir, "latest.pt")
    if not args.no_resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, tokens_seen = ck["step"], ck["tokens"]
        if len(ck["loaders"]) == world:
            loader.load_state(ck["loaders"][rank])
            torch.set_rng_state(ck["rng"][rank]["cpu"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state(ck["rng"][rank]["cuda"].cpu(), device)
        elif master:
            print(f"checkpoint was written by {len(ck['loaders'])} rank(s), now {world}: "
                  f"model and optimizer restored, data order restarts from a fresh permutation", flush=True)
        if master:
            print(f"resumed from {latest} at step {step:,} ({tokens_seen / 1e9:.3f}B tokens)", flush=True)
    if step >= anneal_start and "anneal_mix" in dcfg:
        loader.set_mix(dcfg["anneal_mix"])

    raw_model = model
    if tcfg.get("compile", True) and not args.no_compile:
        model = torch.compile(model)
    if world > 1:
        model = DDP(model, device_ids=[local])

    log_f = open(os.path.join(run_dir, "log.jsonl"), "a") if master else None
    stop_at = min(total_steps, step + args.max_steps) if args.max_steps else total_steps
    model.train()
    t_last = time.time()
    while step < stop_at:
        if step == anneal_start and "anneal_mix" in dcfg:
            loader.set_mix(dcfg["anneal_mix"])
            if master:
                print(f"step {step}: switching to anneal mix {dcfg['anneal_mix']}", flush=True)
        lr = lr_at(step, total_steps, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_acc = torch.zeros((), device=device)
        for k in range(grad_accum):
            x, y = loader.next_batch()
            if world > 1:
                model.require_backward_grad_sync = (k == grad_accum - 1)  # sync grads once per step
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_acc += loss.detach().float() / grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tcfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tcfg["tokens_per_step"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.time()
        dt, t_last = now - t_last, now
        if world > 1:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        tok_s = tcfg["tokens_per_step"] / dt
        mfu = flops_per_token * tok_s / (args.peak_tflops * 1e12 * world)
        rec = {"step": step, "tokens": tokens_seen, "loss": round(loss_acc.item(), 4), "lr": lr,
               "grad_norm": round(float(grad_norm), 3), "dt": round(dt, 3), "tok_s": int(tok_s),
               "mfu": round(mfu, 3)}
        if device.type == "cuda":
            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)

        if step % tcfg["eval_every"] == 0 or step == stop_at:
            val = evaluate(model, val_sets, device, micro, rank, world, dtype)
            rec["val"] = {k: round(v, 4) for k, v in val.items()}
            t_last = time.time()  # do not count eval time against the next step
        if master:
            line = (f"step {step:6d}/{total_steps} loss {rec['loss']:.4f} lr {lr:.2e} gn {rec['grad_norm']:.2f} "
                    f"| {dt * 1e3:6.0f} ms {tok_s / 1e3:6.1f}k tok/s mfu {mfu:.1%}"
                    + (f" mem {rec['peak_mem_gb']:.1f}G" if "peak_mem_gb" in rec else ""))
            if "val" in rec:
                line += " | val " + " ".join(f"{k} {v:.3f}" for k, v in rec["val"].items())
            print(line, flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        if step % tcfg["checkpoint_every"] == 0 or step == stop_at:
            loader_states = [None] * world
            rng = {"cpu": torch.get_rng_state()}
            if device.type == "cuda":
                rng["cuda"] = torch.cuda.get_rng_state(device)
            rng_states = [None] * world
            if world > 1:
                dist.all_gather_object(loader_states, loader.state())
                dist.all_gather_object(rng_states, rng)
            else:
                loader_states, rng_states = [loader.state()], [rng]
            if master:
                save_checkpoint(latest, raw_model, optimizer, step, tokens_seen, loader_states, cfg, rng_states)
                milestone = tcfg.get("milestone_every_tokens", 0)
                if milestone and tokens_seen % milestone < tcfg["tokens_per_step"]:
                    os.link(latest, os.path.join(run_dir, f"step-{step:07d}.pt"))
                print(f"checkpoint saved at step {step}", flush=True)
            t_last = time.time()

    if log_f:
        log_f.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
