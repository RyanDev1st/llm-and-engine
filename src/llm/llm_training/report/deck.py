"""Presentation-deck slide builders (GPU-free). The CONCEPT slides (pipeline, two-verb idea) live
here; the big-number STAT slides live in deck_stats. `main()` renders the whole curated 4-minute set
in one call. Every number traces to docs/report/README.md §3 (the 2026-06-24 Kaggle run) — nothing
fabricated. Design rule: ONE idea per slide, few words, the number does the talking.
  python -m llm_training.report.deck [--out docs/findings/report_assets]"""
from __future__ import annotations

import argparse
from pathlib import Path

NAVY = "#1a2a4f"
GOLD = "#c8a24a"
# Real adapter routing confusion (gold rows -> predicted cols), native fair test, n=142 (README §3a).
ADAPTER_CM = {"skill": {"skill": 104, "tool": 7, "none": 6},
              "tool": {"skill": 0, "tool": 22, "none": 3},
              "none": {"skill": 0, "tool": 0, "none": 0}}
CM_CAPTION = ("Held-out val, n=142. Rows = the right call, columns = what it chose. The diagonal is "
              "126/142 = 88.7% correct\n(vs 49.6% for the un-tuned base). It rarely confuses "
              "“load a skill” with “call a tool.”")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _stage(ax, x, y, w, h, head, phrase, fc, ec):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=2.0))
    ax.text(x + w / 2, y + h * 0.66, head, ha="center", va="center", fontsize=13,
            fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h * 0.32, phrase, ha="center", va="center", fontsize=9.5, color="#333")


def pipeline(out: Path) -> Path:
    """Four stages, one phrase each. Hook: free GPU to train, your own GPU to run."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11.5, 3.0)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [("DATA", "72K examples", "#fdf3d0", GOLD),
              ("TRAIN", "free Kaggle T4", "#dbe7f5", "#2471a3"),
              ("ADAPTER", "tiny, base frozen", "#d5f5e3", "#1e8449"),
              ("SERVE", "your own GPU", "#fadbd8", "#a93226")]
    w, gap, y, h = 0.215, 0.047, 0.30, 0.46
    for i, (head, phrase, fc, ec) in enumerate(stages):
        x = 0.01 + i * (w + gap)
        _stage(ax, x, y, w, h, head, phrase, fc, ec)
        if i < len(stages) - 1:
            ax.annotate("", (x + w + gap, y + h / 2), (x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#888"))
    ax.text(0.5, 0.95, "How it's built", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.07, "Fine-tune a 4B model on a free GPU → run it locally.",
            ha="center", fontsize=10.5, color="#555")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def two_verbs(out: Path) -> Path:
    """The whole idea on one slide: a SKILL loads knowledge, a TOOL takes action. Few words."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11.0, 3.8)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.93, "The idea: it picks the right move — two verbs",
            ha="center", fontsize=16, fontweight="bold", color=NAVY)
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.05, 0.34), 0.42, 0.42, boxstyle="round,pad=0.014",
                                fc="#d5f5e3", ec="#1e8449", lw=2.2))
    ax.add_patch(FancyBboxPatch((0.53, 0.34), 0.42, 0.42, boxstyle="round,pad=0.014",
                                fc="#dbe7f5", ec="#2471a3", lw=2.2))
    ax.text(0.26, 0.66, "SKILL", ha="center", fontsize=18, fontweight="bold", color="#1e8449")
    ax.text(0.26, 0.50, "loads know-how\ninto its head", ha="center", va="center",
            fontsize=11.5, color="#222")
    ax.text(0.74, 0.66, "TOOL", ha="center", fontsize=18, fontweight="bold", color="#2471a3")
    ax.text(0.74, 0.50, "runs a real function,\nthen explains the result", ha="center", va="center",
            fontsize=11.5, color="#222")
    ax.text(0.5, 0.16, "The skills + tools are listed in the prompt and change every time —",
            ha="center", fontsize=10.5, color="#555")
    ax.text(0.5, 0.07, "so it learns to operate ANY toolset, not memorize chess.",
            ha="center", fontsize=10.5, color="#555", fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    from llm_training.report import chart_data as D, charts, deck_stats as S, ppt_charts
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D.REPO / "docs" / "findings" / "report_assets"))
    args = ap.parse_args()
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    st = D.corpus_stats()
    # Pipeline + idea (concept)
    pipeline(o / "slide-pipeline.png")
    two_verbs(o / "slide-two-verbs.png")
    # Data scale (measured)
    S.scale(st["n_train"], st["modes"], o / "slide-scale.png")
    # Benchmark hero stats (README §3)
    S.big_compare("50%", "89%", "base model", "after fine-tuning", "Does the training help? Routing accuracy",
                  "Held-out tests it never saw (n=142). The core win.", o / "slide-win-routing.png")
    S.big_compare("55", "7", "base model", "ours", "It learned restraint — tool over-fires",
                  "Times it grabbed a tool when it should have just loaded a skill. Lower is better.",
                  o / "slide-win-restraint.png")
    S.big_single("92%", "It generalizes to domains it never trained on",
                 "of tasks completed on real cooking / music / wellness / tax prompts (n=60, 95% grounded).",
                 ["cooking", "music", "wellness", "tax"], o / "slide-generalizes.png")
    # Proof (technical / backup)
    ppt_charts.confusion_matrix(ADAPTER_CM, ["skill", "tool", "none"], o / "slide-confusion-adapter.png",
                                CM_CAPTION, title="Where it routes right — E4B v4 adapter (val)")
    # Supporting design charts
    charts.corpus_composition(st, o / "chart-corpus-composition.png")
    charts.training_timeline(D.VERSIONS, o / "chart-training-timeline.png")
    print(f"\nDeck rendered into {o}", flush=True)


if __name__ == "__main__":
    main()
