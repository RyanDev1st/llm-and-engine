"""Presentation deck — FACTUAL/DATA visuals for the talk arc. Brand slides are AI-gen (prompts in
slide-visuals.md). Main() renders the ordered set. All numbers from real artifacts (README §3).
  python -m llm_training.report.deck [--out docs/findings/report_assets]"""
from __future__ import annotations

import argparse
from pathlib import Path

NAVY = "#1a2a4f"
GOLD = "#c8a24a"
GREEN = "#1e8449"
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


def _stage(ax, x, y, w, h, head, lines, fc, ec):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=2.0))
    ax.text(x + w / 2, y + h * 0.74, head, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h * 0.34, "\n".join(lines), ha="center", va="center",
            fontsize=8.8, color="#333", linespacing=1.35)


def pipeline(out: Path) -> Path:
    """How we trained + serve it: 3-stage flow with the REAL infra, then the key knobs + why."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11.5, 5.0)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "How it was trained — and served", ha="center", fontsize=16,
            fontweight="bold", color=NAVY)
    stages = [("TRAIN", ["Kaggle 2× T4 (free)", "3 accounts · ~135 GPU-h", "over weeks"], "#dbe7f5", "#2471a3"),
              ("ADAPTER", ["tiny LoRA on top", "base stays frozen", "→ merge to serve"], "#d5f5e3", "#1e8449"),
              ("SERVE", ["Colab 1× T4 (live site)", "or local GGUF", "on your own GPU"], "#fadbd8", "#a93226")]
    w, gap, y, h = 0.27, 0.065, 0.52, 0.34
    for i, (head, lines, fc, ec) in enumerate(stages):
        x = 0.045 + i * (w + gap)
        _stage(ax, x, y, w, h, head, lines, fc, ec)
        if i < len(stages) - 1:
            ax.annotate("", (x + w + gap, y + h / 2), (x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#888"))
    # the knobs + brief why (the "max seq, ranks, explain why" the script asks for)
    knobs = [("QLoRA · 4-bit nf4", "fit a 4B model on a free T4"),
             ("max seq 1664", "longest example is 1655 tok — don't truncate the reasoning"),
             ("rank 16 · all-linear", "enough to learn the format, small enough to fit"),
             ("loss-weight ×8 on tags", "the harness tags must beat the base model's habits")]
    ax.text(0.5, 0.40, "key settings (and why)", ha="center", fontsize=10.5,
            fontweight="bold", color=GOLD)
    yk = 0.32
    for name, why in knobs:
        ax.text(0.30, yk, name, ha="right", fontsize=9.5, fontweight="bold", color=NAVY)
        ax.text(0.33, yk, why, ha="left", fontsize=9.2, color="#555")
        yk -= 0.075
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def thinks(out: Path) -> Path:
    """THE ACTION LOOP — a clean 5-step horizontal call flow with a loop-back arc.
    Label-only nodes, no paragraphs. Real trace from training."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11.5, 4.2)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.92, "The call flow — one action at a time",
            ha="center", fontsize=18, fontweight="bold", color=NAVY)
    # 5 wide nodes, left → right, with generous gaps
    nodes = [("User asks", "#f0f0f0"), ("<skill>\nload guidance", "#d5f5e3"),
             ("<tool>\ncall function", "#dbe7f5"), ("read\nresult", "#eef2f7"),
             ("grounded\nanswer", GOLD)]
    n, w, gap, y, h = len(nodes), 0.148, 0.040, 0.48, 0.28
    x0 = 0.5 - (n * w + (n - 1) * gap) / 2
    for i, (label, fc) in enumerate(nodes):
        x = x0 + i * (w + gap)
        ec = NAVY if fc != GOLD else GOLD
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010", fc=fc, ec=ec, lw=2.2))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color=NAVY if fc != GOLD else "white", linespacing=1.2)
        if i < n - 1:
            x1 = x + w + gap; yc = y + h/2
            ax.annotate("", (x1, yc), (x + w, yc),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#999"))
    # loop-back arc
    import matplotlib.patches as mpatches
    x4, x1b = x0 + 4*(w + gap) + w/2, x0 + w/2
    ax.add_patch(mpatches.FancyArrowPatch((x4, y - 0.02), (x1b, y - 0.02),
                connectionstyle="arc3,rad=-0.28", arrowstyle="-|>", lw=2.2, color=GREEN,
                mutation_scale=20))
    ax.text(0.5, y - 0.25, "loop: decide → act → read → decide again",
            ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
    # bottom
    ax.text(0.5, 0.05, "The same verbs, the same contract — only the skills + tools "
            "LISTED in the prompt, which change every request.",
            ha="center", fontsize=10, color="#555")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out

def modes(out: Path) -> Path:
    """4 clean cards. Big name, one rule, one example. Generous whitespace."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(11.5, 4.8)); fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "One model — four reasoning modes", ha="center",
            fontsize=18, fontweight="bold", color=NAVY)
    cards = [
        ("FAST", "#7a8699", "no <goal>\nno <think>", "“push Ba6”"),
        ("THINK", "#2471a3", "<goal> once\n<think> every step", "“yo, e6 for me”"),
        ("AUTO", "#1e8449", "<goal> once\n<think> only on\nhard choices", "“play Qa4+ pls”"),
        ("PLAN", "#c8a24a", "<goal> all\n<plan> checklist", "“debug + break down”"),
    ]
    w, gap, y0, h = 0.225, 0.027, 0.34, 0.55; x0 = 0.5 - (4 * w + 3 * gap) / 2
    for i, (name, ec, rule, ex) in enumerate(cards):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.014",
                    fc="white", ec=ec, lw=3.0))
        ax.text(x + w/2, y0 + h - 0.06, name, ha="center", va="top",
                fontsize=18, fontweight="bold", color=ec)
        sep_y = y0 + h - 0.20
        ax.add_artist(Line2D([x + 0.06, x + w - 0.06], [sep_y, sep_y], color=ec, lw=1.2))
        ax.text(x + w/2, sep_y - 0.04, rule, ha="center", va="top",
                fontsize=10, color="#333", linespacing=1.5)
        ax.text(x + w/2, y0 + 0.05, ex, ha="center", va="bottom",
                fontsize=8.5, color="#888", fontstyle="italic")
    # bottom line
    fig.text(0.5, 0.09, "Same contract: <skill> loads guidance · <tool> calls a function · one per step.",
             ha="center", fontsize=11, color="#333")
    fig.text(0.5, 0.03, "A 4B model — we trained the reasoning IN, and the restraint to not over-think.",
             ha="center", fontsize=9.5, color="#555")
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
    # Files are NUMBERED by talk position (see docs/report/slide-visuals.md) so they sort in
    # presentation order and interleave with the presenter's slides: 01-02 = AI-gen brand
    # (meet-the-model, it's-local), 07 = the live chats (presenter supplies). This renders only the
    # FACTUAL / DATA visuals.
    # 03 · the contract: two verbs, one action per step (the FACTUAL loop)
    thinks(o / "03-how-it-works.png")
    # 03b · the four reasoning modes (fast / think / auto / plan)
    modes(o / "03b-reasoning-modes.png")
    # 04 · how trained + served (real knobs + infra)
    pipeline(o / "04-how-trained.png")
    # 05 · the data (slices + mode mix)
    charts.corpus_composition(st, o / "05-the-data.png")
    # 06 · floors out fast (REAL loss curve)
    losses = D.load_train_losses()
    if losses:
        S.floors_out(losses, o / "06-floors-out.png", st["n_train"])
    else:
        print("no runs/full_train.log -> skipping floors-out slide (no fabricated curve)", flush=True)
    # 7 · (chats shown live from the Kaggle run) — not rendered here
    # 8 · does it work — TWO results, framed as TWO DIFFERENT QUESTIONS so they don't read as
    # comparable. EXACT numbers (88.7 / 91.7), never rounded, so they match the confusion backup and
    # don't look fudged. The 55->7 / F1 detail lives ONLY on the confusion backup (no redundant slide).
    # 08 · comparison chart: base+harness vs adapter+harness (grouped bars, native numbers)
    comparison_labels = {
        "E4B base\n+ harness": {"verb": 0.496},
        "E4B v4 adapter\n+ harness": {"verb": 0.887},
    }
    S.comparison(comparison_labels, o / "08-result-comparison.png",
                 title="Does fine-tuning help? Verb accuracy (held-out val, n=142)")
    # 09 · generalization (different test — framed as Q2 to avoid "91.7 vs 88.7 = drop" confusion)
    S.big_single("91.7%", "Q2 · Can it finish a task in a domain it never trained on?",
                 "tasks completed on real cooking / music / wellness / tax prompts (n=60, 95% grounded) — "
                 "a DIFFERENT test from Q1, so this isn't a drop from 88.7%.",
                 ["cooking", "music", "wellness", "tax"], o / "09-result-generalizes.png")
    # BACKUP (not in the main flow): the per-class proof. Same 88.7% as the comparison.
    ppt_charts.confusion_matrix(ADAPTER_CM, ["skill", "tool", "none"], o / "backup-confusion.png",
                                CM_CAPTION, title="Where it routes right — E4B v4 adapter (val)")
    print(f"\nStory deck rendered into {o}", flush=True)


if __name__ == "__main__":
    main()
