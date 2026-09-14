"""Streaming token loader over the shards from scripts/build_shards.py. Chapter 4.

Each source is a list of .npy files of uint16 tokens. A "window" is one
seq_len + 1 slice (inputs are the first seq_len tokens, targets the last). For
each source we enumerate every non-overlapping window across its shards, give
this rank its share (window i belongs to rank i % world_size), and walk them in
a seeded random order. Each batch row picks its source by the mix weights.

Every token is seen at most once per epoch and every rank sees a disjoint set.
If a source runs dry it wraps around with a new permutation; the loader reports
that so you know a source is being repeated.

State is four numbers per source (position in the permutation and the epoch) plus
the RNG, so a checkpoint can resume exactly where it left off. Shards are
memory-mapped; a 40GB corpus costs no RAM to open.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch


class SourceStream:
    def __init__(self, files: list[str], seq_len: int, rank: int, world: int, seed: int):
        self.files = files
        self.arrays = [np.load(f, mmap_mode="r") for f in files]
        self.seq_len = seq_len
        self.rank, self.world, self.seed = rank, world, seed
        stride = seq_len + 1
        # (shard index, start offset) for every window, then keep this rank's share
        shard_ids, starts = [], []
        for si, a in enumerate(self.arrays):
            n = (len(a) - 1) // stride  # windows fully inside the shard
            shard_ids.append(np.full(n, si, dtype=np.int32))
            starts.append(np.arange(n, dtype=np.int64) * stride)
        self.shard_ids = np.concatenate(shard_ids)[rank::world]
        self.starts = np.concatenate(starts)[rank::world]
        self.epoch = 0
        self.pos = 0
        self._perm = self._permutation()

    def _permutation(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.rank, self.epoch])
        return rng.permutation(len(self.starts))

    def __len__(self) -> int:
        return len(self.starts)

    def next(self) -> np.ndarray:
        if self.pos >= len(self._perm):
            self.epoch += 1
            self.pos = 0
            self._perm = self._permutation()
        i = self._perm[self.pos]
        self.pos += 1
        a = self.arrays[self.shard_ids[i]]
        s = self.starts[i]
        return np.asarray(a[s: s + self.seq_len + 1], dtype=np.int64)

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state(self, st: dict) -> None:
        self.epoch, self.pos = st["epoch"], st["pos"]
        self._perm = self._permutation()


class MixLoader:
    def __init__(self, shards_dir: str, mix: dict[str, float], seq_len: int, batch_size: int,
                 rank: int = 0, world: int = 1, seed: int = 0, device: str = "cpu"):
        names = [n for n, w in mix.items() if w > 0]
        total = sum(mix[n] for n in names)
        self.names = names
        self.weights = np.array([mix[n] / total for n in names])
        self.streams: dict[str, SourceStream] = {}
        for n in names:
            files = sorted(glob.glob(os.path.join(shards_dir, n, "train-*.npy")))
            if not files:
                raise FileNotFoundError(f"no train shards for source {n!r} under {shards_dir}")
            self.streams[n] = SourceStream(files, seq_len, rank, world, seed)
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        self.rng = np.random.default_rng([seed, rank, 12345])
        self.epochs_seen = {n: 0 for n in names}

    def check_mix(self, mix: dict[str, float]) -> None:
        """Refuse a mix that names a source with no shards. train.py checks the anneal
        mix at startup so a typo fails in the first second, not five days in; without
        this, set_mix would silently renormalize over the sources it does have."""
        unknown = [n for n, w in mix.items() if w > 0 and n not in self.streams]
        if unknown:
            raise KeyError(f"mix names sources with no shards: {unknown}; have {self.names}")

    def set_mix(self, mix: dict[str, float]) -> None:
        """Change the sampling weights mid-run (used for the anneal phase)."""
        self.check_mix(mix)
        total = sum(mix.get(n, 0.0) for n in self.names)
        self.weights = np.array([mix.get(n, 0.0) / total for n in self.names])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        picks = self.rng.choice(len(self.names), size=self.batch_size, p=self.weights)
        rows = []
        for p in picks:
            st = self.streams[self.names[p]]
            rows.append(st.next())
            if st.epoch != self.epochs_seen[self.names[p]]:
                self.epochs_seen[self.names[p]] = st.epoch
                print(f"[loader] source {self.names[p]} wrapped to epoch {st.epoch}", flush=True)
        buf = torch.from_numpy(np.stack(rows))
        x = buf[:, :-1].to(self.device, non_blocking=True)
        y = buf[:, 1:].to(self.device, non_blocking=True)
        return x, y

    def tokens_available(self) -> dict[str, int]:
        return {n: len(s) * self.seq_len for n, s in self.streams.items()}

    def state(self) -> dict:
        return {"streams": {n: s.state() for n, s in self.streams.items()},
                "rng": self.rng.bit_generator.state}

    def load_state(self, st: dict) -> None:
        for n, s in st["streams"].items():
            self.streams[n].load_state(s)
        self.rng.bit_generator.state = st["rng"]


def load_val(shards_dir: str, name: str, seq_len: int, max_windows: int) -> torch.Tensor:
    """Fixed validation windows for one source, shape (n, seq_len + 1)."""
    a = np.load(os.path.join(shards_dir, name, "val.npy"), mmap_mode="r")
    stride = seq_len + 1
    n = min(max_windows, (len(a) - 1) // stride)
    return torch.from_numpy(np.asarray(a[: n * stride], dtype=np.int64).reshape(n, stride))
