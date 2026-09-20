"""Chapter 5: figures from the training log.

    python scripts/plot_run.py --log checkpoints/elroy-350m/log.jsonl --out docs/img

Writes train-loss.png, val-loss.png, lr.png and prints the numbers the chapter quotes.
One series per colour, fixed colour order, direct end labels, hairline grid, no chart junk.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow
ANNEAL_START_TOKENS = 18e9


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def shade_anneal(ax):
    ax.axvspan(ANNEAL_START_TOKENS / 1e9, 20, color=GRID, alpha=0.5, lw=0)
    ax.text(19, ax.get_ylim()[1], "anneal", color=INK2, fontsize=8, ha="center", va="top")


def ema(x, alpha=0.02):
    out = np.empty_like(x)
    m = x[0]
    for i, v in enumerate(x):
        m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="checkpoints/elroy-350m/log.jsonl")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.log)]
    tok = np.array([r["tokens"] for r in rows]) / 1e9
    loss = np.array([r["loss"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    val = [(r["tokens"] / 1e9, r["val"]) for r in rows if r.get("val")]
    sources = list(val[0][1].keys())
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    # ---- figure 1: train loss
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    style(ax, "tokens (billions)", "training loss (log scale)")
    ax.set_yscale("log")
    ax.plot(tok, loss, color=SERIES[0], alpha=0.15, lw=0.6)
    sm = ema(loss)
    ax.plot(tok, sm, color=SERIES[0], lw=2, solid_capstyle="round")
    ax.set_yticks([1.5, 2, 3, 5, 10]); ax.set_yticklabels(["1.5", "2", "3", "5", "10"])
    ax.set_xlim(0, 20.6); ax.set_ylim(1.4, 11)
    shade_anneal(ax)
    ax.text(tok[-1] + 0.15, sm[-1], f"{sm[-1]:.2f}", color=INK, fontsize=9, va="center")
    ax.set_title("Elroy-350M training loss, 20B tokens", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "train-loss.png")); plt.close(fig)

    # ---- figure 2: validation loss per source
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    style(ax, "tokens (billions)", "validation loss (log scale)")
    ax.set_yscale("log")
    xs = [t for t, _ in val]
    for i, s in enumerate(sources):
        ys = [v[s] for _, v in val]
        ax.plot(xs, ys, color=SERIES[i], lw=2, label=s, solid_capstyle="round")
        ax.text(xs[-1] + 0.15, ys[-1], f"{s} {ys[-1]:.2f}", color=INK, fontsize=8.5, va="center")
    ax.set_yticks([1.2, 1.5, 2, 3, 5]); ax.set_yticklabels(["1.2", "1.5", "2", "3", "5"])
    ax.set_xlim(0, 24.5); ax.set_ylim(1.1, 6.5)
    shade_anneal(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    ax.set_title("Validation loss by source (64 held-out windows each, every 250 steps)", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "val-loss.png")); plt.close(fig)

    # ---- figure 3: learning rate
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=160)
    style(ax, "tokens (billions)", "learning rate")
    ax.plot(tok, lr, color=SERIES[0], lw=2)
    ax.set_xlim(0, 20.6); ax.set_ylim(0, 4.4e-4)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    shade_anneal(ax)
    ax.set_title("Learning rate: 1,000-step warmup, cosine to 4e-5, linear to zero over the last 10%", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "lr.png")); plt.close(fig)

    # ---- numbers for the chapter
    def at(tokens_b):
        i = int(np.argmin(np.abs(tok - tokens_b)))
        return rows[i]
    print("train loss (EMA) at 1B/5B/10B/15B/18B/20B:",
          [round(float(sm[int(np.argmin(np.abs(tok - b)))]), 3) for b in (1, 5, 10, 15, 18, 20)])
    for s in sources:
        ys = {round(t): v[s] for t, v in val}
        first = val[0][1][s]; pre = [v for t, v in val if t <= 18][-1][s]; last = val[-1][1][s]
        print(f"val {s:12s} first {first:.3f}  at 18B {pre:.3f}  final {last:.3f}  anneal gain {pre - last:+.3f}")
    mfu = np.array([r["mfu"] for r in rows[1:]]); dt = np.array([r["dt"] for r in rows[1:]])
    print(f"mean step {dt.mean():.2f}s, mean mfu {mfu.mean():.3f}, total tokens {rows[-1]['tokens'] / 1e9:.3f}B, steps {rows[-1]['step']}")
    gn = np.array([r["grad_norm"] for r in rows])
    print(f"grad norm max {gn.max():.2f} at step {rows[int(gn.argmax())]['step']}, median {np.median(gn):.3f}, spikes > 1.0 after warmup: {(gn[1000:] > 1.0).sum()}")


if __name__ == "__main__":
    main()
