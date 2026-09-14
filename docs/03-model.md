# Chapter 3: The model

By the end of this chapter you have `elroy/model.py`, about 250 lines that define Elroy end to end, and a test that proves the parameter count matches the arithmetic, the fast attention path matches the slow one, and the model cannot see the future.

## The shape of it

```
tokens (B, T)
  -> embedding                          (B, T, d_model)
  -> Block x n_layer:
       x = x + Attention(RMSNorm(x))
       x = x + SwiGLU(RMSNorm(x))
  -> RMSNorm
  -> logits = x @ embedding.T            (B, T, vocab)
```

`B` is the batch, `T` the sequence length (2048 for Elroy-350M), `d_model` the width of the residual stream (1024). Every block reads from the stream, computes something, and adds it back. Nothing is ever overwritten, which is why gradients flow cleanly through 24 layers.

This is the Llama recipe. Compared to GPT-2 it swaps LayerNorm for RMSNorm, learned position embeddings for rotary ones, GELU for SwiGLU, and drops every bias. Each of those is a paragraph below. None of them is essential, all of them are standard, and a reader who finishes this chapter will recognise the shape of every open model released since 2023.

## RMSNorm

```python
xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
return (xf * self.weight.float()).type_as(x)
```

Divide each vector by its root-mean-square, then multiply by a learned per-channel gain. LayerNorm also subtracts the mean and adds a bias; dropping both saves a little compute and changes nothing measurable. We compute it in fp32 even when the rest of the model runs in bf16, because the sum of squares of a 1024-vector is exactly the kind of number that loses precision in 16 bits.

## Rotary position embeddings

Attention on its own is order-blind: shuffle the tokens and the attention scores shuffle with them. The model needs to know where things are. GPT-2 added a learned vector per position to the embeddings. RoPE instead rotates the query and key vectors by an angle proportional to their position, in pairs of dimensions, at a different frequency per pair:

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
freqs = torch.outer(positions, inv_freq)      # (T, head_dim / 2)
```

Pair 0 rotates fast (one radian per position), the last pair rotates slowly (one radian per ten thousand positions). When a rotated query at position `m` meets a rotated key at position `n`, the dot product depends only on `m - n`, because the two rotations compose into a single rotation by the difference. The test checks exactly this: the same two vectors placed at positions (5, 8) and at (20, 23) produce the same score.

The practical consequence is that the model learns relative offsets ("the token three back") rather than absolute slots ("position 1,847"), which is both what code needs and what lets a model generalise a little past its training length.

## Attention

```python
q, k, v = self.qkv(x).split(C, dim=2)
q = q.view(B, T, n_head, head_dim).transpose(1, 2)     # (B, H, T, D)
...
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

One matrix produces queries, keys and values; they are split into 16 heads of 64 dimensions each so that different heads can attend to different things. The math, written out in `naive_attention`, is:

```python
scores = q @ k.transpose(-2, -1) / sqrt(D)      # how much does each position want each other position
scores = scores.masked_fill(~tril, -inf)        # a position may only look backward
return scores.softmax(-1) @ v                   # weighted sum of what it found
```

The mask is the whole reason a language model works: position `t` computes its output from positions `0..t` only, so the prediction for token `t+1` cannot cheat by reading it. The test flips one token in the middle of a sequence and asserts that every logit before it is bit-for-bit unchanged.

Training uses `scaled_dot_product_attention`, which on an Ampere GPU dispatches to a fused flash-attention kernel. It computes the same thing without ever materialising the `T x T` score matrix, which at `T = 2048` and 16 heads would be 128MB per sequence in bf16 and would dominate memory. The test asserts that the fast path and the naive path agree to 1e-5.

## SwiGLU

```python
return self.down(F.silu(self.gate(x)) * self.up(x))
```

The feed-forward network is where a transformer stores most of what it knows. The classic version is two matrices with a GELU between them. SwiGLU uses three: `gate` and `up` both project to the hidden width, they are multiplied elementwise after a SiLU on the gate, and `down` projects back. Because it has three matrices, the hidden width is set to 3x `d_model` rather than 4x to keep the parameter count comparable. Every model since PaLM has found it slightly better for the same parameters, and that is the entire justification.

## Tied embeddings

The input embedding matrix (`vocab x d_model`, 33.5M parameters at our size) is reused as the output projection. The intuition is that the vector that means "this is the token `def`" on the way in is a reasonable vector to compare against on the way out. It saves 10% of the parameters at 350M and is standard for small models. One side effect shows up in the test: at initialization, the logit for the *current* token is `<norm(e), e>`, which is a little larger than the logits for random other tokens, so the model starts with a slight bias toward repeating its input and the initial next-token loss is a hair above `ln(vocab)` rather than exactly at it.

## Initialization

All matrices are drawn from a normal with standard deviation 0.02. The two matrices that write into the residual stream in each block (`attn.proj` and `mlp.down`) are scaled down by `1 / sqrt(2 x n_layer)`, so that the stream does not grow with depth as 48 residual additions pile up. This is the GPT-2 trick and it is the difference between a stable start and a loss that spikes in the first hundred steps.

## Counting the parameters

Per block:

| Piece | Parameters |
|---|---|
| q, k, v, out projections | 4 x d_model x d_model |
| gate, up, down | 3 x d_model x d_ff |
| two RMSNorm gains | 2 x d_model |

Plus the embedding (`vocab x d_model`) and the final norm (`d_model`).

For Elroy-350M (`d_model` 1024, `d_ff` 3072, 24 layers, 32,768 vocab): 4.19M + 9.44M + 2K per block, times 24, is 327.2M; plus 33.5M for the embedding; total 360.8M. Elroy-mini (512 / 1408 / 8 layers) is 42.5M. `count_params_by_formula` is that table in code, and the first line of the test asserts the model agrees with it.

## FLOPs per token

Training costs about `6 x params` FLOPs per token: the forward pass is `2 x params` (one multiply and one add per weight), and the backward pass is twice that. Attention adds `12 x n_layer x d_model x T` for the score and value products, which at `T = 2048` is another 0.6 GFLOP. Elroy-350M comes to 2.77 GFLOP per token; 20B tokens is 5.5e19. `flops_per_token()` is what the training loop divides into measured throughput to report MFU.

## Running the test

```bash
python tests/test_model.py
ok: 164,672 params match formula; sdpa==naive; causal; init loss 10.425 vs ln(V) 10.397; rope relative
```

## Next

Chapter 4 feeds this model tokens as fast as the GPUs can take them, which turns out to be mostly a question of batch shapes and one call to `torch.compile`.
