"""Model checks. Run with:  python tests/test_model.py

1. Parameter count equals the formula from the chapter.
2. Fast attention (SDPA) matches the written-out naive attention.
3. Causality: changing a token never changes logits at earlier positions.
4. Loss at initialization is close to ln(vocab_size): the model starts out
   near-uniform.
5. RoPE gives a dot product that depends only on relative position.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elroy.model import Elroy, ModelConfig, apply_rope, count_params_by_formula, rope_cos_sin  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=1000, n_layer=2, n_head=4, d_model=64, d_ff=176, seq_len=32)
    model = Elroy(cfg).eval()

    assert model.num_params() == count_params_by_formula(cfg), (model.num_params(), count_params_by_formula(cfg))

    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        fast, _ = model(idx)
        for b in model.blocks:
            b.attn.use_sdpa = False
        slow, _ = model(idx)
        for b in model.blocks:
            b.attn.use_sdpa = True
    assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()

    idx2 = idx.clone()
    idx2[:, 10] = (idx2[:, 10] + 1) % cfg.vocab_size
    with torch.no_grad():
        out2, _ = model(idx2)
    assert torch.allclose(fast[:, :10], out2[:, :10], atol=1e-5), "future token leaked into the past"
    assert not torch.allclose(fast[:, 10:], out2[:, 10:]), "changed token had no effect"

    big = ModelConfig(vocab_size=32768, n_layer=2, n_head=4, d_model=64, d_ff=176, seq_len=32)
    m2 = Elroy(big)
    x = torch.randint(0, big.vocab_size, (4, 33))
    _, loss = m2(x[:, :-1], x[:, 1:])   # next-token targets, as in training
    # Tied embeddings give the current token a slightly larger logit at init (its
    # logit is <norm(e), e>), which is the wrong answer for next-token prediction,
    # so the loss starts a hair above ln(V) rather than exactly at it.
    assert abs(loss.item() - math.log(big.vocab_size)) < 0.3, loss.item()

    cos, sin = rope_cos_sin(64, 16, 10000.0)
    q = torch.randn(1, 1, 64, 16)
    k = torch.randn(1, 1, 64, 16)
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    # same relative offset, different absolute positions, same vector content -> same score
    q2 = q.clone(); k2 = k.clone()
    q2[..., 20, :] = q[..., 5, :]
    k2[..., 23, :] = k[..., 8, :]
    q2r, k2r = apply_rope(q2, cos, sin), apply_rope(k2, cos, sin)
    s1 = (qr[..., 5, :] * kr[..., 8, :]).sum()
    s2 = (q2r[..., 20, :] * k2r[..., 23, :]).sum()
    assert torch.allclose(s1, s2, atol=1e-4), (s1.item(), s2.item())

    print(f"ok: {model.num_params():,} params match formula; sdpa==naive; causal; "
          f"init loss {loss.item():.3f} vs ln(V) {math.log(big.vocab_size):.3f}; rope relative")


if __name__ == "__main__":
    main()
