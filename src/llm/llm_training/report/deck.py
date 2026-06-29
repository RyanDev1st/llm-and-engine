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
    stages = [("TRAIN", ["Kaggle 2× T4 (free)", "DDP training", "38h per session"], "#dbe7f5", "#2471a3"),
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
             ("DDP · 2× GPUs", "1.81x speedup, 38-hour iteration sessions to reach v4"),
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
    ax.text(0.5, 0.96, "The Agent Harness & Thinking Loop", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#ffffff")
    ax.add_artist(Line2D([0.38, 0.62], [0.91, 0.91], color="#c8a24a", lw=2.0))

    # Box dimensions
    w, h = 0.40, 0.11
    cx = 0.5

    nodes = [
        (0.85, "1. USER PROMPT", "“What's the best move?”", "#1c202a", "#7a8699", True),
        (0.70, "2. AGENT: REASONING", "Sets <goal>, uses <think>", "#15221c", "#2ecc71", False),
        (0.55, "3. AGENT: XML CALL", "Selects <skill> or <tool>", "#15221c", "#2ecc71", False),
        (0.40, "4. HARNESS: EXECUTE", "Executes parsed function", "#1a202c", "#3498db", False),
        (0.25, "5. HARNESS: RESULT", "Returns state to agent", "#1a202c", "#3498db", False),
        (0.10, "6. GROUNDED REPLY", "“Qxc5 is the move… Should I go deeper?”", "#2c2518", "#c8a24a", True),
    ]

    for y, title, text, fc, ec, is_italic in nodes:
        card = FancyBboxPatch((cx - w/2, y - h/2), w, h, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.5)
        ax.add_patch(card)
        ax.text(cx, y + 0.015, title, ha="center", va="center", fontsize=11, fontweight="bold", color=ec)
        
        txt_color = "#ffffff" if is_italic else "#8395a7"
        txt_size = 10.5 if is_italic else 9
        
        ax.text(cx, y - 0.020, text, ha="center", va="center", fontsize=txt_size, color=txt_color, 
                fontstyle="italic" if is_italic else "normal")

    def _draw_down_arrow(y_start, y_end, color):
        arrow = mpatches.FancyArrowPatch((cx, y_start), (cx, y_end), arrowstyle="-|>", lw=2.0, color=color, mutation_scale=16)
        ax.add_patch(arrow)

    # Down arrows
    _draw_down_arrow(0.85 - h/2, 0.70 + h/2, "#7a8699")
    _draw_down_arrow(0.70 - h/2, 0.55 + h/2, "#2ecc71")
    _draw_down_arrow(0.55 - h/2, 0.40 + h/2, "#3498db")
    _draw_down_arrow(0.40 - h/2, 0.25 + h/2, "#3498db")

    # Minimal Check Node (> 6 turns?)
    chk_x, chk_y = 0.15, 0.25
    chk_w, chk_h = 0.12, 0.06
    chk_box = FancyBboxPatch((chk_x - chk_w/2, chk_y - chk_h/2), chk_w, chk_h, boxstyle="round,pad=0.01", fc="#1a202c", ec="#7a8699", lw=1.5)
    ax.add_patch(chk_box)
    ax.text(chk_x, chk_y, "> 6 turns?", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#dcdde1")

    # Path 1: Node 5 -> Check Node
    left_x = cx - w/2
    ax.plot([left_x, chk_x + chk_w/2], [0.25, 0.25], color="#3498db", lw=2.0)
    ax.add_patch(mpatches.FancyArrowPatch((left_x - 0.02, 0.25), (chk_x + chk_w/2, 0.25), arrowstyle="-|>", lw=2.0, color="#3498db", mutation_scale=16))

    # Path 2: Check Node (No) -> Node 2
    ax.plot([chk_x, chk_x, left_x - 0.02], [chk_y + chk_h/2, 0.70, 0.70], color="#3498db", lw=2.0)
    ax.add_patch(mpatches.FancyArrowPatch((left_x - 0.02, 0.70), (left_x, 0.70), arrowstyle="-|>", lw=2.0, color="#3498db", mutation_scale=16))
    ax.text(chk_x - 0.015, 0.50, "No", ha="right", va="center", fontsize=10.5, color="#3498db", fontweight="bold")

    # Path 3: Check Node (Yes) -> Node 6
    ax.plot([chk_x, chk_x, left_x - 0.02], [chk_y - chk_h/2, 0.10, 0.10], color="#e74c3c", lw=2.0)
    ax.add_patch(mpatches.FancyArrowPatch((left_x - 0.02, 0.10), (left_x, 0.10), arrowstyle="-|>", lw=2.0, color="#e74c3c", mutation_scale=16))
    ax.text(chk_x - 0.015, 0.175, "Yes", ha="right", va="center", fontsize=10.5, color="#e74c3c", fontweight="bold")

    # Right Done Arrow (2 -> 6)
    right_x = cx + w/2
    ax.plot([right_x, 0.85, 0.85, right_x + 0.02], [0.70, 0.70, 0.10, 0.10], color="#c8a24a", lw=2.0)
    ax.add_patch(mpatches.FancyArrowPatch((right_x + 0.02, 0.10), (right_x, 0.10), arrowstyle="-|>", lw=2.0, color="#c8a24a", mutation_scale=16))
    ax.text(0.87, (0.70 + 0.10)/2, "Goal hit, reply", ha="center", va="center", rotation=270, fontsize=10.5, color="#c8a24a", fontweight="bold")

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
        ("FAST", "#7a8699", "No <goal>\nNo <think>", "“play e4”", "Direct move command", ["Prompt", "Act"]),
        ("THINK", "#3498db", "Use <goal>\n<think> every step", "“explain why my move failed”", "Coaching & analysis", ["Goal", "Think", "Tag"]),
        ("AUTO", "#2ecc71", "Use <goal>\n<think> when hard", "“make a move, check threats”", "Balanced gameplay", ["Goal", "Quiet", "Think*"]),
        ("PLAN", "#c8a24a", "Use <goal>\n<plan> checklist", "“audit game + run python sim”", "Multi-step validation", ["Goal", "Plan", "Tick"]),
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


def data_factory(out: Path) -> Path:
    """THE DATA FACTORY — How we built 73k examples from hand-written parts."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.patch.set_facecolor("#11141a")
    ax.set_facecolor("#11141a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.88, "Where data from?", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#ffffff")
    ax.text(0.5, 0.80, "“We didn’t just grab it from nowhere, its ours”", ha="center",
            fontsize=12, color="#c8a24a", fontstyle="italic")

    # Buckets
    buckets = [
        ("Question", "6", "#3498db", "“spot anything wrong”"),
        ("Style", "6", "#2ecc71", "slang (yo ___)"),
        ("Scenario", "3", "#9b59b6", "auth hole bug"),
        ("Mode", "3", "#e67e22", "fast / think"),
        ("Closer", "10", "#e74c3c", "“jump to fix?”")
    ]
    w, h = 0.13, 0.21
    gap = 0.04
    start_x = 0.5 - (len(buckets) * w + (len(buckets)-1) * gap) / 2
    
    y_buckets = 0.48
    for i, (name, count, color, example) in enumerate(buckets):
        x = start_x + i * (w + gap)
        box = FancyBboxPatch((x, y_buckets), w, h, boxstyle="round,pad=0.01", fc="#1c202a", ec=color, lw=2.0)
        ax.add_patch(box)
        ax.text(x + w/2, y_buckets + h - 0.04, name, ha="center", va="top", fontsize=12, fontweight="bold", color="#ffffff")
        ax.text(x + w/2, y_buckets + h/2 - 0.01, count, ha="center", va="center", fontsize=32, fontweight="bold", color=color)

        # One-liner short examples BELOW the buckets
        ax.text(x + w/2, y_buckets - 0.05, example, ha="center", va="top", fontsize=10.5, color="#8395a7", fontstyle="italic")

        if i < len(buckets) - 1:
            ax.text(x + w + gap/2, y_buckets + h/2, "×", ha="center", va="center", fontsize=24, color="#7a8699")

    # Bottom equation and card details
    ax.text(0.5, 0.22, "6 × 6 × 3 × 3 × 10 = 3,240 for one card (skill)", ha="center", fontsize=14, fontweight="bold", color="#dcdde1")

    ax.text(0.5, 0.11, "20 General Skills   +   5 Chess Skills", ha="center", fontsize=15, fontweight="bold", color="#2ecc71")
    ax.text(0.5, 0.04, "(Chess uses real Stockfish self-play positions stopped at random depths)", ha="center", fontsize=11, color="#8395a7", fontstyle="italic")

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
    # BACKUP: The Data Factory diagram
    data_factory(o / "backup-data-factory.png")
    print(f"\nStory deck rendered into {o}", flush=True)


if __name__ == "__main__":
    main()
