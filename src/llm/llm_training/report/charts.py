"""Matplotlib builders for the report charts (GPU-free). Each takes plain data + an output path and
writes a PNG. Kept deterministic and dependency-light so they render the same on Kaggle or locally.
CLI builds the three that need no GPU (layer-contribution, corpus, training timeline) from
chart_data; the per-version trend + per-slice bars are written by report.version_eval after the run.
  python -m llm_training.report.charts [--out docs/findings/report_assets]"""
from __future__ import annotations

import argparse
from pathlib import Path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def layer_contribution(cond: dict, out: Path, title: str = "Routing: harness vs SFT-weights (val)") -> Path:
    """Grouped bars: each condition x {verb, macro-prec, exact-name}. Shows what the harness
    contract buys (vs base no-harness) and what the trained weights add on top."""
    plt = _plt()
    labels = list(cond)
    metrics = [("verb", "verb accuracy"), ("macro", "macro precision"), ("exact", "exact-name")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.25
    for i, (k, name) in enumerate(metrics):
        xs = [j + i * w for j in range(len(labels))]
        vals = [cond[l].get(k, 0.0) for l in labels]
        bars = ax.bar(xs, vals, width=w, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([j + w for j in range(len(labels))], labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)
    return out


def corpus_composition(stats: dict, out: Path) -> Path:
    """Two panels: reasoning-mode mix (the fast/think/auto/plan contract) + top-12 slice sizes."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    modes = stats["modes"]
    order = [m for m in ("fast", "think", "auto", "plan") if m in modes] + \
            [m for m in modes if m not in ("fast", "think", "auto", "plan")]
    a1.bar(order, [modes[m] for m in order])
    for i, m in enumerate(order):
        a1.text(i, modes[m], f"{modes[m]:,}", ha="center", va="bottom", fontsize=8)
    a1.set_title(f"Reasoning-mode mix (train, n={stats['n_train']:,})")
    a1.set_ylabel("rows")
    top = sorted(stats["slices"].items(), key=lambda x: -x[1])[:12]
    names = [k.replace("V1_", "")[:18] for k, _ in top]
    a2.barh(names[::-1], [v for _, v in top][::-1])
    a2.set_title(f"Top slices by size ({stats['n_slices']} slices total)")
    a2.set_xlabel("rows")
    fig.suptitle(f"SFT corpus v1.2 — design mix ~75% general / 25% chess · train {stats['n_train']:,} / val {stats['n_val']:,}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)
    return out


def per_slice_bars(slice_acc: dict, out: Path, title: str = "Per-slice routing accuracy (v4 adapter)") -> Path:
    """Horizontal bars of per-slice exact routing accuracy (verb+name). slice_acc: {slice: 0..1}."""
    plt = _plt()
    items = sorted(slice_acc.items(), key=lambda x: x[1])
    names = [k.replace("V1_", "")[:22] for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.26 * len(items))))
    bars = ax.barh(names, vals, color=["#c0392b" if v < 0.5 else "#2980b9" for v in vals])
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("exact routing accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)
    return out


def training_timeline(versions: list, out: Path) -> Path:
    """The v2->v3->v4 story: a line over versions with the measured verb accuracy where available,
    each point annotated with the bug it fixed. Versions without a measured number sit on the line
    with the annotation only (honest — no fabricated metric)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    xs = list(range(len(versions)))
    ys = [v.get("verb") for v in versions]
    have = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(have) >= 2:
        ax.plot([x for x, _ in have], [y for _, y in have], "-o", color="#2980b9", zorder=3)
    last = len(versions) - 1
    for x, v in zip(xs, versions):
        y = v.get("verb")
        if y is not None:
            ax.scatter([x], [y], color="#2980b9", zorder=4)
            ax.text(x, y + 0.03, f"{y:.0%}", ha="center", fontsize=9, fontweight="bold")
        yb = (y if y is not None else 0.5)
        # keep edge boxes inside the canvas: first anchors left, last anchors right, middle centers
        ha = "left" if x == 0 else "right" if x == last else "center"
        dx = 8 if x == 0 else -8 if x == last else 0
        ax.annotate(f"why: {v['why']}\nfix: {v['fix']}", (x, yb), textcoords="offset points",
                    xytext=(dx, -34), ha=ha, fontsize=6.5,
                    bbox=dict(boxstyle="round", fc="#fef9e7", ec="#d4ac0d"))
    ax.set_xticks(xs, [f"{v['label']}\n{v['date']}" for v in versions])
    ax.set_xlim(-0.6, last + 0.6)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("routing verb accuracy (val)")
    ax.set_title("E4B harness — diagnose -> fix -> measure across retrains")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    from llm_training.report import chart_data as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D.REPO / "docs" / "findings" / "report_assets"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    layer_contribution(D.COND_VAL, outdir / "chart-layer-contribution.png")
    corpus_composition(D.corpus_stats(), outdir / "chart-corpus-composition.png")
    training_timeline(D.VERSIONS, outdir / "chart-training-timeline.png")
    print(f"\nGPU-free report charts in {outdir}", flush=True)


if __name__ == "__main__":
    main()
