"""Big-number stat slides (GPU-free) — the audience-centric half of the deck: one idea, one giant
number per image, minimal words. A class audience reads a number, not a paragraph. Numbers trace to
docs/report/README.md §3 (the 2026-06-24 Kaggle run); nothing here is fabricated.

  big_compare  — before -> after hero (e.g. 50% -> 89% routing; 55 -> 7 tool over-fires)
  big_single   — one hero number + context chips (e.g. 92% on unseen domains)
  scale        — three stat callouts + the measured reasoning-mode donut
"""
from __future__ import annotations

from pathlib import Path

SLATE = "#7a8699"   # the "before" / baseline tone (neutral, not alarming)
GREEN = "#1e8449"   # the "after" / ours tone (clearly better)
NAVY = "#1a2a4f"
GOLD = "#c8a24a"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _titlebar(fig, title):
    fig.text(0.5, 0.92, title, ha="center", va="top", fontsize=17, fontweight="bold", color=NAVY)
    fig.add_artist(_line(0.34, 0.66, 0.875))


def _line(x0, x1, y):
    from matplotlib.lines import Line2D
    ln = Line2D([x0, x1], [y, y], color=GOLD, lw=2.2)
    return ln


def big_compare(before: str, after: str, before_lbl: str, after_lbl: str,
                title: str, sub: str, out: Path) -> Path:
    """A before -> after hero: two giant numbers with an arrow, baseline muted, ours green."""
    plt = _plt()
    fig = plt.figure(figsize=(9.6, 5.4)); fig.patch.set_facecolor("white")
    _titlebar(fig, title)
    fig.text(0.27, 0.50, before, ha="center", va="center", fontsize=86, fontweight="bold", color=SLATE)
    fig.text(0.50, 0.52, "→", ha="center", va="center", fontsize=58, color="#bbb")
    fig.text(0.74, 0.50, after, ha="center", va="center", fontsize=92, fontweight="bold", color=GREEN)
    fig.text(0.27, 0.30, before_lbl, ha="center", va="top", fontsize=12, color=SLATE)
    fig.text(0.74, 0.30, after_lbl, ha="center", va="top", fontsize=12.5, color=GREEN, fontweight="bold")
    fig.text(0.5, 0.10, sub, ha="center", va="center", fontsize=11, color="#444")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def big_single(value: str, title: str, sub: str, chips: list[str], out: Path) -> Path:
    """One hero number (green) + context chips below (e.g. the unseen domains it was tested on)."""
    plt = _plt()
    fig = plt.figure(figsize=(9.6, 5.4)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    _titlebar(fig, title)
    fig.text(0.5, 0.56, value, ha="center", va="center", fontsize=120, fontweight="bold", color=GREEN)
    fig.text(0.5, 0.30, sub, ha="center", va="center", fontsize=12.5, color="#333")
    n = len(chips); cw, gap = 0.16, 0.025
    x0 = 0.5 - (n * cw + (n - 1) * gap) / 2
    from matplotlib.patches import FancyBboxPatch
    for i, c in enumerate(chips):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.13), cw, 0.075, boxstyle="round,pad=0.006",
                                    fc="#eef2f7", ec=NAVY, lw=1.2, transform=ax.transAxes))
        ax.text(x + cw / 2, 0.1675, c, ha="center", va="center", fontsize=11,
                color=NAVY, transform=ax.transAxes)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def floors_out(losses: list[float], out: Path, n_train: int = 72329, grad_accum: int = 16) -> Path:
    """The REAL training-loss curve (runs/full_train.log): loss drops fast then floors. The story
    beat 'it learns the harness in a fraction of one pass, so we don't train all 72K'. The examples-
    seen + fraction-of-epoch are computed from the measured update count, not asserted."""
    plt = _plt()
    fig = plt.figure(figsize=(9.6, 5.4)); fig.patch.set_facecolor("white")
    _titlebar(fig, "It learns fast — then floors out")
    ax = fig.add_axes([0.11, 0.20, 0.84, 0.52])
    xs = list(range(1, len(losses) + 1))
    ax.plot(xs, losses, color=NAVY, lw=1.6)
    floor = sum(losses[-20:]) / min(20, len(losses))
    ax.axhline(floor, color=GOLD, lw=1.4, ls="--")
    ax.text(len(losses) * 0.5, floor + 0.22, f"floors at ~{floor:.1f}", ha="center",
            fontsize=9.5, color=GOLD, fontweight="bold")
    ax.set_xlabel("training updates (optimizer steps)", fontsize=10)
    ax.set_ylabel("loss", fontsize=10)
    ax.set_ylim(0, max(losses) * 1.05)
    ax.set_xlim(1, len(losses))
    seen = len(losses) * grad_accum
    frac = seen / n_train
    fig.text(0.5, 0.085, f"{len(losses)} updates ≈ {seen/1000:.1f}K examples ≈ {frac:.0%} of ONE pass "
             f"through the {n_train/1000:.0f}K — so we never train the full set.",
             ha="center", fontsize=11, color="#333")
    fig.text(0.5, 0.025, "Why: ~85% of every example is the same harness contract — it nails the "
             "format fast; the rest it generalizes.", ha="center", fontsize=9.5, color="#666")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out
