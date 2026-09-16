# Chapter 4: The training loop

By the end of this chapter the corpus is on disk as token shards, the model is training on both GPUs at 70% of their measured peak, and you can kill the process at any moment and resume it exactly where it stopped. This chapter also answers the question the plan left open: how many days.

## Step 1: shards

```bash
python scripts/build_shards.py --tokenizer data/tokenizer.json --out data/shards
```

Training reads tokens, not text, so tokenization happens once, up front. For each source the script reads the parquet files, applies the Chapter 1 cleaning rules, tokenizes each document, appends `<|endoftext|>`, and writes the concatenated stream as `uint16` NumPy arrays of 100M tokens. The first 2M tokens of each source are split off as `val.npy`. A `manifest.json` records the counts.

Two details matter. The row groups are shuffled (seeded) before tokenization, because Stack-Edu was fetched best-score-first: without the shuffle, `val.npy` would be all score-5 files and training would march through the scores in order. And only the parent process touches parquet: it reads one row group at a time, hands 4,000-document slices to the worker pool, and a semaphore keeps at most two chunks per worker in flight. That last part cost two failed attempts. The first version gave each worker a whole row group to decode; Stack-Edu's row groups are 100,000 files, a worker holding the decoded strings plus their token lists was a couple of GB, and 24 of them tripped the OOM killer at 30GB. The second version had each worker re-read the row group and take a slice, which still left every worker holding pyarrow's decompression buffers for the whole group, about 1.3GB each, and died the same way seventeen minutes in. `Pool.imap` has its own trap: it drains its input on a feeder thread as fast as it can, so a generator that reads the corpus lazily would still end up with the whole corpus in the task queue. The semaphore is what bounds it. The final version runs at 6GB total.

Throughput is parent-bound: 3.8M tokens per second on Stack-Edu (short files, so the parent spends its time decoding strings and pickling), 10M on StarCoderData and Cosmopedia, 15M on FineWeb-Edu. The full corpus took 66 minutes and produced 52GB of shards:

```
source        docs         tokens   train shards
python_edu    13,725,239   9.687B    97
python_stack   3,768,950   4.340B    44
fineweb_edu    9,672,101  10.695B   107
cosmopedia     3,761,237   2.818B    29
total         30,927,527  27.540B   277
```

Against the Chapter 2 estimates (11B, 5B, 10B, 3B) the code sources came in lower: the parse filter and the header strip together dropped 11% of Stack-Edu's documents and 14% of StarCoderData's, and FIM adds three tokens per transformed file, not enough to matter. The budget still holds. At the 45/20/25/10 mix, 18B tokens of main phase uses 8.1B of `python_edu` and 3.6B of `python_stack`, and no source wraps.

### The anneal mix

The plan called for the final 10% of training to shift toward the best Python plus a slice of instruction-style data. The first config wrote that down literally, with a `sft_style` source in `anneal_mix`, and there is no such shard source: the instruction data belongs to Chapter 7, where it gets a loss mask, and was never built into pretraining shards. The loader would have accepted it anyway, dropping the unknown name and renormalizing over the other two, silently, at step 34,331. Two changes. `MixLoader.check_mix` now refuses a mix that names a source without shards, and `train.py` checks the anneal mix at startup so the failure is in the first second rather than five days in. And the anneal mix is now `python_edu` 0.60, `cosmopedia` 0.25, `fineweb_edu` 0.15. The `python_edu` share is capped by what is left after the main phase (1.59B of 2B), and Cosmopedia's textbook register is the closest thing in the corpus to the answering style Chapter 7 trains.

### Fill-in-the-middle

Half of the code documents are rearranged before tokenization:

```
PSM:  <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
SPM:  <|fim_prefix|><|fim_suffix|>{suffix}<|fim_middle|>{prefix}{middle}
```

Two random character positions split the file into prefix, middle and suffix. In PSM order the model sees the prefix, then the suffix, then has to produce the middle; SPM puts the suffix first. Half of the FIM documents use each. At inference, a user who wants a function body filled in constructs the same prompt and the model completes the middle. The cost is zero: it is the same next-token objective on a rearranged document. The benefit is that a model trained this way can be used inside an editor, not only at the end of a file. This is the recipe from the original FIM paper and from StarCoder.

## Step 2: the loader

`elroy/loader.py` memory-maps every shard, enumerates every non-overlapping window of `seq_len + 1` tokens per source, gives each GPU its share (window `i` belongs to rank `i % world_size`), and walks them in a seeded random order. Each row of a batch picks its source by the mix weights (45/20/25/10). The loader's whole state is a position per source plus an RNG, so a checkpoint can resume the exact sequence of batches.

If a source runs out it wraps to a second epoch with a new permutation and says so on stdout. With 20B tokens of training and a corpus a little larger than that, no source should wrap; if one does, the mix is off.

## Step 3: the loop

`elroy/train.py` is the whole thing, about 250 lines. The core is this:

```python
for k in range(grad_accum):
    x, y = loader.next_batch()
    model.require_backward_grad_sync = (k == grad_accum - 1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    (loss / grad_accum).backward()
grad_norm = clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

One optimizer step consumes `tokens_per_step` tokens, 524,288 for Elroy-350M, regardless of hardware. The GPU can only hold `micro_batch` sequences at a time, so the loop runs `grad_accum` forward-backward passes per GPU and lets the gradients add up before stepping. With two GPUs, `micro_batch` 4 and `seq_len` 2048, that is 32 micro-steps per optimizer step. The `require_backward_grad_sync` line tells DDP to exchange gradients between the GPUs only on the last micro-step rather than on every one, which matters: the exchange is 1.4GB.

Precision: parameters and optimizer state stay in fp32; the forward and backward run in bf16 under `autocast`. bf16 has the exponent range of fp32 and needs no loss scaling. RMSNorm and the cross-entropy cast back to fp32 internally because those are the two places where bf16's eight bits of mantissa hurt.

Optimizer: AdamW with `betas` (0.9, 0.95), weight decay 0.1 on the matrices and none on the norm gains, gradient clipping at 1.0. These are the GPT-3 / Llama defaults and there is no reason to touch them at this scale.

Learning rate: 1,000 steps of linear warmup to 4e-4, cosine decay to 4e-5 over the main phase, then a straight line to zero over the final 10% while the data mix switches to `anneal_mix`. The plan explains why the anneal exists; the loop just implements it by calling `loader.set_mix` at the boundary.

`torch.compile` wraps the model. It fuses the elementwise work (RMSNorm, RoPE, SiLU, the residual adds) into a handful of kernels and roughly doubles throughput. Compilation takes about a minute on the first step.

## Step 4: tune for the hardware you have

This is where the standing rule applies: measure on the actual cards, do not copy numbers from a different machine.

```bash
bash scripts/sweep_micro_batch.sh configs/elroy-350m.yaml data/shards-dev "4 8 16"
```

On the two A4500s:

| micro_batch | Result |
|---|---|
| 4 | 39.6k tokens/s, 13.2 s per step, 70% MFU, 18.9GB per GPU |
| 8 | out of memory |
| 16 | out of memory |

So `micro_batch` is 4. The 70% MFU is against the 78 TFLOPS we measured in Chapter 0, or 58% against the 94 on the spec sheet, and it is about as good as a dense model gets without hand-written kernels. The memory at 4 is almost entirely activations plus the logits: `4 x 2048 x 32768` logits in fp32 for the loss is 1GB on its own. Activation checkpointing would let `micro_batch` 8 fit at the cost of recomputing the forward pass, and would be slower; there is no reason to want it here.

For the mini config on one GPU: `micro_batch` 16 at `seq_len` 1024 uses 6.3GB and runs at 157k tokens/s, 61% MFU. A 12GB card is comfortable; an 8GB card should drop to 8.

## Step 5: how long

38,146 steps at 13.2 seconds is 140 hours, about 5.9 days, plus a few percent for evaluation and checkpoints. Call it 6 days. The plan said 9 based on an assumed 35% MFU; the measured 70% is what changed it.

## Checkpoints and resume

Every 250 steps (about 55 minutes) the loop writes `checkpoints/elroy-350m/latest.pt`: model, optimizer, step, the loader position on every rank, and the RNG states, about 4.3GB. Every 5B tokens the current checkpoint is also hard-linked as a permanent `step-NNNNNNN.pt`. The file is written to a temporary name and renamed, so a crash mid-write cannot corrupt the last good one.

Resuming is the same command with no extra flags. We tested it: run 6 steps, stop, run 8 more, and the step counter, learning rate and loss continue as if nothing happened. Resuming on a different number of GPUs restores the model and optimizer but restarts the data order from a fresh permutation, and says so.

That test was on the mini config, and the first real resume of the 350M run, at step 13,000, ran one step and died with CUDA out of memory. The checkpoint was loaded with `map_location=device`, straight onto the GPU. `load_state_dict` copies the tensors into the parameters that already live there and the optimizer casts its state to match, so the loaded dict is a second copy of everything, 4.3GB, and it stayed alive because the variable holding it was never dropped. At 17.3GB of a 20GB card there is no room for a second copy of the model and optimizer. The fix is two lines: load to CPU, and `del` the dict once its contents have been copied. The mini config never showed it because 42M parameters times two fits anywhere. Test resume at the scale you are going to run.

A second bug surfaced the same day: the milestone check tested whether the current step had crossed a 5B-token boundary, but checkpoints are only written every 250 steps and 5B tokens is step 9,537, so the condition could never be true on a step that writes a checkpoint. The 5B milestone was lost; the check now fires on the first checkpoint past each boundary, and the step-12,750 checkpoint (6.7B tokens) stands in for it.

## Evaluation during training

Every 250 steps the loop computes the loss on 64 held-out windows per source. Four numbers, not one: `python_edu`, `python_stack`, `fineweb_edu`, `cosmopedia`. Watching them separately is what tells you whether the code is improving or whether the English is carrying the average. The log is a JSONL file with one record per step, which Chapter 5 turns into curves.

## The 10-minute check

Before spending six days, spend ten minutes:

```bash
python scripts/build_shards.py --tokenizer data/tokenizer.json --out data/shards-dev --max-docs 3000 --max-files 3
python -m elroy.train --config configs/mini.yaml --shards data/shards-dev --max-steps 200
```

The loss should fall from about 10.4 to under 7 in the first couple of hundred steps of the mini config. If it does not move, something upstream is wrong and no amount of GPU time will fix it. Ours did:

```
step      1/3814 loss 10.4905 lr 2.00e-06 gn 3.36 |   2856 ms   91.8k tok/s mfu 35.9% mem 6.0G
step     25/3814 loss 9.5680 lr 5.00e-05 gn 1.33 |   1690 ms  155.1k tok/s mfu 60.7% mem 6.3G
step     50/3814 loss 8.4633 lr 1.00e-04 gn 1.18 |   1702 ms  154.0k tok/s mfu 60.3% mem 6.3G
step    100/3814 loss 6.7787 lr 2.00e-04 gn 1.55 |   1709 ms  153.4k tok/s mfu 60.0% mem 6.3G
step    150/3814 loss 5.7986 lr 3.00e-04 gn 1.32 |   1710 ms  153.3k tok/s mfu 60.0% mem 6.3G
step    200/3814 loss 5.3194 lr 4.00e-04 gn 0.89 |   1711 ms  153.2k tok/s mfu 60.0% mem 6.3G | val python_edu 4.873 python_stack 5.738 fineweb_edu 5.956 cosmopedia 6.177
```

Six minutes, 52M tokens, loss from 10.49 to 5.32, and the per-source validation losses already say something: the Python sources are easier than the English ones, and Stack-Edu (beginner-heavy, repetitive) is the easiest of all. The first step is slow because `torch.compile` is compiling.

## Next

Chapter 5 is the real run: what the curves looked like, what broke, and what the model could do at each milestone.
