"""Presentation-deck slide builders (GPU-free) — the three the report charts lacked: the training+
serving PIPELINE, a DATA-anatomy concept card (the two-verb harness contract), and the base->adapter
RECALIBRATION before/after bars. Plus a `main()` that renders the WHOLE curated deck (these three +
the real confusion matrix + cross-model lines + corpus + timeline) into report_assets in one call, so
a 4-minute talk's images come from one command. Every number traces to the 2026-06-24 Kaggle run
(see docs/report/README.md §3); nothing here is fabricated.
  python -m llm_training.report.deck [--out docs/findings/report_assets]"""
from __future__ import annotations

import argparse
from pathlib import Path

# Measured base->adapter recalibration (docs/report/README.md §3a, native n=142).
RECAL = {"routing verb": (0.496, 0.887), "tool F1": (0.42, 0.81), "skill F1": (0.56, 0.94)}
# Real adapter routing confusion (gold rows -> predicted cols), native fair test, n=142.
ADAPTER_CM = {"skill": {"skill": 104, "tool": 7, "none": 6},
              "tool": {"skill": 0, "tool": 22, "none": 3},
              "none": {"skill": 0, "tool": 0, "none": 0}}
CM_CAPTION = (
    "Held-out val, n=142 — fair native-mode test (each row scored in its trained reasoning mode).\n"
    "Rows = the correct verb; columns = what the model chose. 126/142 = 88.7% on the diagonal,\n"
    "vs 49.6% for the un-tuned base. The fine-tune taught it to LOAD A SKILL before acting and only\n"
    "call a TOOL when one is needed. Tool F1 0.81 (was 0.42), skill F1 0.94 (was 0.56).")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _box(ax, x, y, w, h, lines, fc, ec):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.6))
    head, *body = lines
    ax.text(x + w / 2, y + h - 0.05, head, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h - 0.16, "\n".join(body), ha="center", va="top",
            fontsize=8.2, color="#222")


def pipeline(out: Path) -> Path:
    """Left->right flow: SFT data -> QLoRA train (Kaggle T4) -> LoRA adapter -> local GGUF serve."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11.5, 3.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (["1 · DATA", "v1.2 SFT corpus", "~73k rows · 2 verbs", "75% general / 25% chess"], "#fdf3d0", "#b7950b"),
        (["2 · TRAIN", "QLoRA · Unsloth", "Gemma-4 E4B (nf4 4-bit)", "Kaggle T4 · seq 1664"], "#d6eaf8", "#2471a3"),
        (["3 · ADAPTER", "LoRA (all-linear)", "base frozen", "merge -> bf16"], "#d5f5e3", "#1e8449"),
        (["4 · SERVE", "GGUF q4/q5/q6", "local RTX 4060", "+ vision mmproj"], "#fadbd8", "#a93226"),
    ]
    w, gap, y, h = 0.215, 0.047, 0.30, 0.50
    for i, (lines, fc, ec) in enumerate(stages):
        x = 0.01 + i * (w + gap)
        _box(ax, x, y, w, h, lines, fc, ec)
        if i < len(stages) - 1:
            ax.annotate("", (x + w + gap, y + h / 2), (x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#555"))
    ax.text(0.5, 0.94, "Pipeline — train once on a free T4, serve locally",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.06, "A general agentic harness: the model picks among the skills + tools in its "
            "prompt and reasons to a goal — chess is one demo domain of many.",
            ha="center", fontsize=8.6, color="#555")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def data_anatomy(out: Path) -> Path:
    """Concept card: the two-verb contract every row teaches (skill = load guidance, tool = act)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "What every training row teaches: two verbs, one action per step",
            ha="center", fontsize=13, fontweight="bold")
    _box(ax, 0.04, 0.40, 0.43, 0.40,
         ["<skill> NAME </skill>", "loads a listed skill's guidance", "into context (progressive",
          "disclosure) — it does NOT act"], "#d5f5e3", "#1e8449")
    _box(ax, 0.53, 0.40, 0.43, 0.40,
         ["<tool> NAME args </tool>", "calls a real function and gets",
          "a result the model must narrate", "(never computes it itself)"], "#d6eaf8", "#2471a3")
    ax.text(0.5, 0.30, "Reasoning modes:  fast  (no <think>)   ·   think  (every step)   ·   "
            "auto  (only on hard decisions)", ha="center", fontsize=9.5, color="#333")
    ax.text(0.5, 0.18, "user → <think> plan → <skill> load → <tool> act → grounded answer",
            ha="center", fontsize=10.5, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round", fc="#fef9e7", ec="#d4ac0d"))
    ax.text(0.5, 0.05, "The skills + tools are LISTED in the prompt and vary per row — so the model "
            "learns to operate ANY catalog, not memorize chess.", ha="center", fontsize=8.6, color="#555")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def recalibration(cond: dict, out: Path) -> Path:
    """Grouped before/after bars: base vs v4 adapter on verb / tool F1 / skill F1 (the fine-tune win)."""
    plt = _plt()
    labels = list(cond)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    w = 0.34
    xs = list(range(len(labels)))
    base = [cond[l][0] for l in labels]
    adpt = [cond[l][1] for l in labels]
    b1 = ax.bar([x - w / 2 for x in xs], base, w, label="E4B base + harness", color="#c0392b")
    b2 = ax.bar([x + w / 2 for x in xs], adpt, w, label="E4B v4 adapter (ours)", color="#1e8449")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + w / 2, bar.get_height() + 0.01, f"{bar.get_height():.0%}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs, labels, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (held-out val, n=142)")
    ax.set_title("The fine-tuning win — base over-fires tools; the adapter routes correctly")
    ax.legend(fontsize=9, loc="upper left")
    fig.text(0.5, 0.005, "Tool false-positives 55 → 7 · the base fired a tool when it should have "
             "loaded a skill; the adapter learned the difference.", ha="center", fontsize=8.4, color="#555")
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    from llm_training.report import chart_data as D, charts, ppt_charts
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D.REPO / "docs" / "findings" / "report_assets"))
    args = ap.parse_args()
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    pipeline(o / "slide-pipeline.png")
    data_anatomy(o / "slide-data-anatomy.png")
    recalibration(RECAL, o / "slide-recalibration.png")
    ppt_charts.confusion_matrix(ADAPTER_CM, ["skill", "tool", "none"],
                                o / "slide-confusion-adapter.png", CM_CAPTION,
                                title="Routing verb confusion — E4B v4 adapter (held-out val)")
    ppt_charts.model_lines(D.MODELS, o / "slide-model-lines.png")
    charts.corpus_composition(D.corpus_stats(), o / "chart-corpus-composition.png")
    charts.training_timeline(D.VERSIONS, o / "chart-training-timeline.png")
    print(f"\nDeck rendered into {o}", flush=True)


if __name__ == "__main__":
    main()
