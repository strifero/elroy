# Chapter 0: Setup

By the end of this chapter you have a Python environment with a CUDA build of PyTorch, you know what your GPU can do in bf16, and you have run a two-process job with `torchrun`, which is exactly how training will run later.

## What a language model is, in three paragraphs

A language model is a function that takes a sequence of tokens and returns a probability for what the next token is. Tokens are chunks of text, usually a few characters long; Chapter 2 builds the thing that decides what those chunks are. Training means showing the model billions of real sequences and nudging its parameters so that the real next token gets a higher probability than it did before. That is the whole objective. Everything a trained model does, from completing a function to answering a question, is next-token prediction applied repeatedly.

The model we build is a transformer: a stack of identical blocks, each of which lets every position in the sequence look at every earlier position (attention) and then processes each position independently (a small feed-forward network). Chapter 3 writes it out. It has no notion of Python or English; it learns both from the data in Chapter 1 because that is what is in the data.

The reason this works at all is scale. A few hundred million parameters, a few tens of billions of tokens, and a week of GPU time turn a random initialization into something that writes working code. This guide is about getting from nothing to that, with every step visible.

## Hardware

Elroy-350M was trained on a workstation with two NVIDIA RTX A4500 GPUs (20GB each, connected with NVLink), an AMD Ryzen 9 7900, 32GB of RAM and a 1TB NVMe drive. The mini config trains on one GPU with 12GB or more, and the code runs on CPU for reading and debugging.

One warning about the machine itself: if it has been doing other things, clear them out first. Ours had an Ollama model pinned to both GPUs at boot, which took 30GB of the 40GB we have, and a ComfyUI server holding another 200MB. Anything that holds GPU memory will fight your training run; `nvidia-smi` shows who is holding what. We disabled the Ollama preload unit and will stop Ollama and ComfyUI entirely for the duration of the main run.

## Environment

We use [uv](https://docs.astral.sh/uv/) to make a Python 3.12 virtual environment and install PyTorch with CUDA 12.8 wheels. Any recent CUDA build of PyTorch 2.4+ works. Pip works too if you prefer it.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/strifetech/elroy.git
cd elroy
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

Check that PyTorch sees your GPUs:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

## The smoke test

`scripts/smoke_test.py` does three things per GPU: confirms bf16 arithmetic works, times a large bf16 matrix multiply to estimate throughput, and, when run under `torchrun` with more than one process, performs an NCCL all-reduce so you know the cards can exchange gradients.

```bash
python scripts/smoke_test.py
torchrun --standalone --nproc_per_node=2 scripts/smoke_test.py
```

Our numbers:

```
$ python scripts/smoke_test.py
[rank 0/1] NVIDIA RTX A4500 sm_86 19.6 GiB bf16=yes torch=2.11.0+cu128 dense bf16 matmul ~78 TFLOPS

$ torchrun --standalone --nproc_per_node=2 scripts/smoke_test.py
[rank 0/2] NVIDIA RTX A4500 sm_86 19.6 GiB bf16=yes torch=2.11.0+cu128 dense bf16 matmul ~78 TFLOPS
[rank 1/2] NVIDIA RTX A4500 sm_86 19.6 GiB bf16=yes torch=2.11.0+cu128 dense bf16 matmul ~75 TFLOPS
NCCL all-reduce of 1 GiB across 2 GPUs: 24 ms (40.9 GiB/s)
```

Two things to read off that. First, the spec sheet says an A4500 does 94 TFLOPS of dense bf16; we measure 78 on a big square matmul, and training will see less again. Plan from measured numbers, not the box. Second, 41 GiB/s between the cards means NVLink is doing the work (the four links report 14 GB/s each with `nvidia-smi nvlink -s`; plain PCIe 4.0 x16 would top out around 25 GB/s and usually lands well under that). The first all-reduce in a process is slow because NCCL is still setting up, so the script warms up before timing. If you forget that, you will conclude your interconnect is five times slower than it is; we did.

bf16 matters. Ampere and newer GPUs (sm_80+) support it natively, and it lets us train in half precision without the loss-scaling dance that fp16 requires. If your card reports `bf16=NO` (Turing, Volta, and older), the mini config still trains in fp16 with a gradient scaler; Chapter 4 covers the difference.

The matmul TFLOPS number is your ceiling. Real training reaches 30 to 45% of it. That fraction is called model FLOPs utilization (MFU) and Chapter 4 measures it for every config, because it is the number that turns "20 billion tokens" into "9 days."

## Storage

Token shards for the full run are about 40GB. Checkpoints for the 350M model are about 5GB each with optimizer state; we keep the last three and a milestone every 5B tokens. Raw parquet for the full data mix is another 60 to 80GB before tokenization. Budget 150GB of local disk. Everything lives under `data/` and `checkpoints/`, both ignored by git.

## Next

Chapter 1 downloads the data and looks at it, which is the part most guides skip and the part that decides what the model can do.
