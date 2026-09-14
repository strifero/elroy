"""Elroy, single file: model, tokenizer and sampler for running a released checkpoint.

Needs torch, safetensors and regex. No transformers, no dependency on the
elroy package. This file is copied into every release folder by
scripts/export_hf.py; the code is the same as elroy/model.py and
elroy/tokenizer.py minus training.

    from elroy_min import load, generate, chat_prompt
    model, tok = load("path/to/elroy-350m-chat")
    ids = tok.encode(chat_prompt("Write fizzbuzz."))
    print(tok.decode(generate(model, ids, stop_ids={tok.special_ids["<|end|>"], tok.eot})))
"""

from __future__ import annotations

import json
import math
import os

import regex
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_RESERVED_SPECIALS = 16
SYSTEM = "You are Elroy, a small Python assistant built from scratch by Strife Technologies."


# ---------------------------------------------------------------- tokenizer
class Tokenizer:
    def __init__(self, merges, vocab_size, specials, pattern):
        self.vocab_size = vocab_size
        self._re = regex.compile(pattern)
        self.ranks = {tuple(p): 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        base = vocab_size - NUM_RESERVED_SPECIALS
        self.special_ids = {s: base + i for i, s in enumerate(specials)}
        self.id_to_special = {v: k for k, v in self.special_ids.items()}
        self._special_re = regex.compile("(" + "|".join(regex.escape(s) for s in specials) + ")")
        self._cache: dict[bytes, list[int]] = {}
        self.eot = self.special_ids["<|endoftext|>"]

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        d = json.load(open(path))
        return cls(d["merges"], d["vocab_size"], d["specials"], d["pattern"])

    def _bpe(self, word: bytes) -> list[int]:
        ids = list(word)
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            ids[best_i:best_i + 2] = [best_rank]
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        out = []
        for m in self._re.finditer(text):
            w = m.group().encode("utf-8")
            ids = self._cache.get(w)
            if ids is None:
                ids = self._cache[w] = self._bpe(w)
            out.extend(ids)
        return out

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            sid = self.special_ids.get(chunk)
            out.append(sid) if sid is not None else out.extend(self.encode_ordinary(chunk))
        return out

    def decode(self, ids) -> str:
        parts = [self.id_to_special[i].encode() if i in self.id_to_special else self.vocab[i] for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")


# -------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None].type_as(x), sin[None, None].type_as(x)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp = RMSNorm(d), Attention(d, h), RMSNorm(d), SwiGLU(d, ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class Elroy(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, h, ff, L, V, T = cfg["d_model"], cfg["n_head"], cfg["d_ff"], cfg["n_layer"], cfg["vocab_size"], cfg["seq_len"]
        self.seq_len = T
        self.embed = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d, h, ff) for _ in range(L))
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.embed.weight
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, d // h, 2).float() / (d // h)))
        fr = torch.outer(torch.arange(T).float(), inv)
        self.register_buffer("rope_cos", fr.cos(), persistent=False)
        self.register_buffer("rope_sin", fr.sin(), persistent=False)

    def forward(self, idx):
        T = idx.size(1)
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.rope_cos[:T], self.rope_sin[:T])
        return self.lm_head(self.norm(x))


def load(folder: str, device: str | None = None) -> tuple[Elroy, Tokenizer]:
    from safetensors.torch import load_file
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(folder, "config.json")))
    model = Elroy(cfg)
    state = load_file(os.path.join(folder, "model.safetensors"))
    model.load_state_dict(state, strict=False)   # lm_head.weight is tied, stored once
    model.to(device).eval()
    return model, Tokenizer.load(os.path.join(folder, "tokenizer.json"))


def chat_prompt(user: str, system: str = SYSTEM) -> str:
    return f"<|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>"


@torch.no_grad()
def generate(model: Elroy, ids: list[int], max_new_tokens: int = 256, temperature: float = 0.2,
             top_p: float = 0.95, stop_ids: set[int] | None = None) -> list[int]:
    device = next(model.parameters()).device
    stop_ids = stop_ids or set()
    x = torch.tensor([ids], device=device)
    out = []
    for _ in range(max_new_tokens):
        logits = model(x[:, -model.seq_len:])[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if 0 < top_p < 1:
                s, order = torch.sort(logits, descending=True)
                p = F.softmax(s, dim=-1)
                s[torch.cumsum(p, dim=-1) - p >= top_p] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(0, order, s)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        if nxt in stop_ids:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out


if __name__ == "__main__":
    import sys
    model, tok = load(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "def fizzbuzz(n):\n"
    ids = tok.encode(chat_prompt(prompt)) if json.load(open(os.path.join(sys.argv[1], "config.json"))).get("variant") == "chat" \
        else tok.encode_ordinary(prompt)
    print(tok.decode(generate(model, ids, stop_ids={tok.eot, tok.special_ids.get("<|end|>", -1)})))
