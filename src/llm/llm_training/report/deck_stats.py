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


def _callout(fig, x, value, label):
    fig.text(x, 0.55, value, ha="center", va="center", fontsize=46, fontweight="bold", color=NAVY)
    fig.text(x, 0.40, label, ha="center", va="center", fontsize=11.5, color="#444")


def scale(n_train: int, modes: dict, out: Path) -> Path:
    """Three callouts (examples / model size / hardware) + a donut of the measured reasoning modes."""
    plt = _plt()
    fig = plt.figure(figsize=(10.2, 5.2)); fig.patch.set_facecolor("white")
    _titlebar(fig, "Trained cheap, at scale — and taught to think")
    _callout(fig, 0.17, f"{n_train/1000:.0f}K", "training examples")
    _callout(fig, 0.40, "4B", "params (Gemma-4 E4B)")
    _callout(fig, 0.63, "1", "free Kaggle T4 GPU")
    ax = fig.add_axes([0.75, 0.26, 0.19, 0.50])
    order = [m for m in ("fast", "think", "auto", "plan") if m in modes]
    vals = [modes[m] for m in order]
    cols = ["#aeb8c7", "#6f86b8", GREEN, GOLD]
    ax.pie(vals, colors=cols[:len(order)], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "reason\nmodes", ha="center", va="center", fontsize=9, color=NAVY)
    pct = {m: modes[m] * 100 // sum(vals) for m in order}
    fig.text(0.845, 0.17, "   ".join(f"{m} {pct[m]}%" for m in order[:2]), ha="center",
             fontsize=8.5, color="#444")
    fig.text(0.845, 0.125, "   ".join(f"{m} {pct[m]}%" for m in order[2:]), ha="center",
             fontsize=8.5, color="#444")
    fig.text(0.40, 0.115, "3 of 4 examples are general-purpose by design — chess is just the demo domain.",
             ha="center", fontsize=10.5, color="#555")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out
