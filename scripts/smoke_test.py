"""Chapter 0 smoke test.

Run on one GPU:
    python scripts/smoke_test.py

Run on every GPU in the box (this is how training will run too):
    torchrun --standalone --nproc_per_node=2 scripts/smoke_test.py

It reports, per GPU: name, memory, whether bf16 works, and a rough dense bf16
matmul throughput in TFLOPS. With more than one process it also does an NCCL
all-reduce so you know the cards can talk to each other. Nothing here is
Elroy-specific; it just proves the ground is solid before we build on it.
"""

import os
import time

import torch


def matmul_tflops(device: torch.device, n: int = 8192, iters: int = 20) -> float:
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    return 2 * n**3 * iters / dt / 1e12


def main() -> None:
    distributed = "RANK" in os.environ
    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0

    if not torch.cuda.is_available():
        print("No CUDA device. CPU is fine for the mini config, but this test is for GPUs.")
        return

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    cap = f"sm_{props.major}{props.minor}"
    bf16_ok = torch.cuda.is_bf16_supported()

    x = torch.ones(4, device=device, dtype=torch.bfloat16)
    assert (x + x).sum().item() == 8.0, "bf16 arithmetic failed"

    tflops = matmul_tflops(device)
    print(
        f"[rank {rank}/{world}] {props.name} {cap} "
        f"{props.total_memory / 2**30:.1f} GiB bf16={'yes' if bf16_ok else 'NO'} "
        f"torch={torch.__version__} dense bf16 matmul ~{tflops:.0f} TFLOPS"
    )

    if distributed:
        # The first collective pays for NCCL setup, so warm up before timing.
        warm = torch.ones(1024, device=device)
        for _ in range(3):
            torch.distributed.all_reduce(warm)
        t = torch.full((256, 1024, 1024), float(rank + 1), device=device)  # 1 GiB fp32
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch.distributed.all_reduce(t)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        expected = sum(range(1, world + 1))
        assert t[0, 0, 0].item() == expected, "all-reduce produced the wrong value"
        if rank == 0:
            print(f"NCCL all-reduce of 1 GiB across {world} GPUs: {dt * 1e3:.0f} ms "
                  f"({t.numel() * 4 / dt / 2**30:.1f} GiB/s)")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
