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
    """THE ACTION LOOP — Ultra-minimal text, purely visual flowchart."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#11141a")
    ax.set_facecolor("#11141a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.94, "The Agent Harness & Thinking Loop", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#ffffff")
    ax.add_artist(Line2D([0.38, 0.62], [0.89, 0.89], color="#c8a24a", lw=2.0))

    # Box dimensions & centers
    cx_L, cx_R = 0.28, 0.78
    y_top, y_mid, y_bot = 0.80, 0.46, 0.12
    
    # 1. User Prompt
    w1, h1 = 0.36, 0.12
    card1 = FancyBboxPatch((cx_L - w1/2, y_top - h1/2), w1, h1, boxstyle="round,pad=0.01", fc="#1c202a", ec="#7a8699", lw=1.5)
    ax.add_patch(card1)
    ax.text(cx_L, y_top + 0.02, "1. USER PROMPT", ha="center", va="center", fontsize=12, fontweight="bold", color="#ffffff")
    ax.text(cx_L, y_top - 0.025, "“find a good move”", ha="center", va="center", fontsize=11, color="#dcdde1", fontstyle="italic")

    # 2. Agent Model
    w2, h2 = 0.36, 0.26
    card2 = FancyBboxPatch((cx_L - w2/2, y_mid - h2/2), w2, h2, boxstyle="round,pad=0.01", fc="#15221c", ec="#2ecc71", lw=2.0)
    ax.add_patch(card2)
    ax.text(cx_L, y_mid + 0.08, "2. AGENT", ha="center", va="center", fontsize=14, fontweight="bold", color="#2ecc71")
    
    agent_text = "<goal>\n<think>\n<skill> or <tool>"
    ax.text(cx_L, y_mid - 0.03, agent_text, ha="center", va="center", fontsize=12, color="#ffffff", fontweight="bold", linespacing=1.8)

    # 3. Harness
    w3, h3 = 0.36, 0.26
    card3 = FancyBboxPatch((cx_R - w3/2, y_mid - h3/2), w3, h3, boxstyle="round,pad=0.01", fc="#1a202c", ec="#3498db", lw=1.5)
    ax.add_patch(card3)
    ax.text(cx_R, y_mid + 0.08, "3. HARNESS", ha="center", va="center", fontsize=14, fontweight="bold", color="#3498db")
    
    harness_text = "Injects Skill\nRuns Code\nReturns Data"
    ax.text(cx_R, y_mid - 0.03, harness_text, ha="center", va="center", fontsize=12, color="#ffffff", fontweight="bold", linespacing=1.8)

    # 4. Final Reply
    w4, h4 = 0.36, 0.12
    card4 = FancyBboxPatch((cx_L - w4/2, y_bot - h4/2), w4, h4, boxstyle="round,pad=0.01", fc="#2c2518", ec="#c8a24a", lw=1.5)
    ax.add_patch(card4)
    ax.text(cx_L, y_bot + 0.02, "4. GROUNDED REPLY", ha="center", va="center", fontsize=12, fontweight="bold", color="#c8a24a")
    ax.text(cx_L, y_bot - 0.025, "“Play Ba6”", ha="center", va="center", fontsize=11, color="#dcdde1", fontstyle="italic")

    # Arrows
    def _draw_arrow(start, end, color="#7a8699"):
        arrow = mpatches.FancyArrowPatch(start, end, arrowstyle="-|>", lw=2.0, color=color, mutation_scale=16)
        ax.add_patch(arrow)

    # User -> Agent
    _draw_arrow((cx_L, y_top - h1/2), (cx_L, y_mid + h2/2), "#7a8699")
    
    # Agent -> Harness (Top Half)
    arr_y1 = y_mid + 0.05
    _draw_arrow((cx_L + w2/2, arr_y1), (cx_R - w3/2, arr_y1), "#2ecc71")
    ax.text((cx_L + cx_R)/2, arr_y1 + 0.02, "Action", ha="center", va="bottom", fontsize=11, color="#2ecc71", fontweight="bold")
    
    # Harness -> Agent (Bottom Half)
    arr_y2 = y_mid - 0.05
    _draw_arrow((cx_R - w3/2, arr_y2), (cx_L + w2/2, arr_y2), "#3498db")
    ax.text((cx_L + cx_R)/2, arr_y2 - 0.02, "Data", ha="center", va="top", fontsize=11, color="#3498db", fontweight="bold")
    
    # Agent -> Reply
    _draw_arrow((cx_L, y_mid - h2/2), (cx_L, y_bot + h4/2), "#c8a24a")
    
    # Add a visual "Looping" indicator
    loop_arc = mpatches.FancyArrowPatch((cx_L - w2/2 - 0.01, y_mid - 0.06), (cx_L - w2/2 - 0.01, y_mid + 0.06), 
                                         connectionstyle="arc3,rad=-0.8", arrowstyle="-|>", lw=2.0, color="#2ecc71", mutation_scale=16)
    ax.add_patch(loop_arc)
    ax.text(cx_L - w2/2 - 0.05, y_mid, "LOOP", ha="right", va="center", fontsize=11, fontweight="bold", color="#2ecc71", rotation=90)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def modes(out: Path) -> Path:
    """4 clean cards. Big name, one rule, one example. Generous whitespace."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    fig.patch.set_facecolor("#11141a")
    ax.set_facecolor("#11141a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "One Model — Four Reasoning Modes", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#ffffff")
    ax.add_artist(Line2D([0.38, 0.62], [0.88, 0.88], color="#c8a24a", lw=2.0))

    cards = [
        ("FAST", "#7a8699", "No <goal>\nNo <think>", "“push Ba6”", "Direct action", ["Prompt", "Act"]),
        ("THINK", "#3498db", "Use <goal>\n<think> every step", "“yo, e6 for me”", "Multi-turn analysis", ["Goal", "Think", "Tag"]),
        ("AUTO", "#2ecc71", "Use <goal>\n<think> when hard", "“play Qa4+ pls”", "Speed & depth", ["Goal", "Quiet", "Think*"]),
        ("PLAN", "#c8a24a", "Use <goal>\n<plan> checklist", "“debug + break down”", "Compound tasks", ["Goal", "Plan", "Tick"]),
    ]

    w, gap, y0, h = 0.22, 0.02, 0.22, 0.56
    x0 = 0.5 - (4 * w + 3 * gap) / 2

    for i, (name, ec, rule, ex, desc, steps) in enumerate(cards):
        x = x0 + i * (w + gap)
        # Card Background
        card = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.01", fc="#161a23", ec=ec, lw=2.0)
        ax.add_patch(card)

        # Header Title
        ax.text(x + w/2, y0 + h - 0.06, name, ha="center", va="top",
                fontsize=15, fontweight="bold", color=ec)
        
        # Rule text
        ax.text(x + w/2, y0 + h - 0.16, rule, ha="center", va="top",
                fontsize=9.2, color="#ffffff", linespacing=1.3)

        # Draw mini sequence flow (simplified)
        yc = y0 + h * 0.42
        n_steps = len(steps)
        xs = [x + w * (0.2 + 0.6 * j / max(1, n_steps - 1)) for j in range(n_steps)]
        box_w, box_h = 0.05, 0.05
        
        for j, step_lbl in enumerate(steps):
            box_x = xs[j] - box_w / 2
            box_y = yc - box_h / 2
            step_box = FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="circle,pad=0.01" if j==0 else "round,pad=0.01",
                                      fc="#1e2530", ec=ec, lw=1.2)
            ax.add_patch(step_box)
            ax.text(xs[j], yc, step_lbl, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#ffffff")
            
            if j < n_steps - 1:
                arrow = mpatches.FancyArrowPatch((xs[j] + box_w/2 + 0.01, yc), (xs[j+1] - box_w/2 - 0.01, yc),
                                                 arrowstyle="-|>", lw=1.2, color="#777", mutation_scale=10)
                ax.add_patch(arrow)

        # Mode description
        ax.text(x + w/2, y0 + 0.10, desc, ha="center", va="bottom",
                fontsize=9, color="#7a8699")

        # Example text
        ax.text(x + w/2, y0 + 0.04, ex, ha="center", va="bottom",
                fontsize=8.5, color=ec, fontstyle="italic")

    # bottom line
    fig.text(0.5, 0.10, "Same contract: <skill> loads guidance · <tool> calls a function · one per step.",
             ha="center", fontsize=10.5, color="#dcdde1")
    fig.text(0.5, 0.04, "A 4B model — we trained the reasoning IN, and the restraint to not over-think.",
             ha="center", fontsize=9.5, color="#7a8699")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
