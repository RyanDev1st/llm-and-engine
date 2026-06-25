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
    """THE ACTION LOOP — a clean 8-step vertical staircase flowchart.
    Light text, dark charcoal background, clear columns and flow arrows.
    Real trace from training."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    # Setup figure with dark background
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    fig.patch.set_facecolor("#11141a")
    ax.set_facecolor("#11141a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.94, "The Agent Harness & Thinking Loop", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#ffffff")
    ax.add_artist(Line2D([0.38, 0.62], [0.90, 0.90], color="#c8a24a", lw=2.0))

    # Column Headers
    ax.text(0.26, 0.86, "USER / HARNESS (ENVIRONMENT)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#7a8699")
    ax.text(0.74, 0.86, "AGENT MODEL (INLINE LOOP)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#c8a24a")

    # Coordinates
    w, h = 0.32, 0.125
    cx1, cx2 = 0.26, 0.74
    y1, y2, y3, y4 = 0.70, 0.51, 0.32, 0.13

    # Helper function to draw a card
    def _draw_card(x_center, y, title, body, bg_color, border_color, title_color):
        x = x_center - w / 2
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=bg_color, ec=border_color, lw=1.5)
        ax.add_patch(card)
        ax.text(x_center, y + h * 0.76, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color)
        ax.text(x_center, y + h * 0.34, body, ha="center", va="center", fontsize=7.2, color="#dcdde1", linespacing=1.2)

    # Helper to draw a direct arrow
    def _draw_arrow(start, end, color="#7a8699", connection_style=None):
        if connection_style:
            arrow = mpatches.FancyArrowPatch(start, end, connectionstyle=connection_style,
                                             arrowstyle="-|>", lw=1.5, color=color, mutation_scale=12)
        else:
            arrow = mpatches.FancyArrowPatch(start, end, arrowstyle="-|>", lw=1.5, color=color, mutation_scale=12)
        ax.add_patch(arrow)

    # Row 1: Step 1 (User Input) -> Step 2 (Commit Goal)
    _draw_card(cx1, y1, "1. USER PROMPT (Input)",
               "Human sends a request\ne.g., 'check threats and find a move'",
               "#1c202a", "#7a8699", "#ffffff")
    _draw_card(cx2, y1, "2. COMMIT OBJECTIVE",
               "<goal>threats; best move</goal>\nGoal held in state to prevent early stopping",
               "#1a202c", "#3498db", "#3498db")

    # Row 2: Step 3 (Routing Decision) -> Step 4 (Skill Catalog)
    _draw_card(cx2, y2, "3. ROUTING DECISION",
               "<think>need guidelines; load skill</think>\nEmits tags: <skill>threats</skill>",
               "#15221c", "#27ae60", "#2ecc71")
    _draw_card(cx1, y2, "4. HARNESS DISCLOSURE",
               "Harness retrieves and returns skill body\n(Instructions injected dynamically into context)",
               "#1c202a", "#7a8699", "#7a8699")

    # Row 3: Step 5 (Tool Call) -> Step 6 (Tool Subprocess)
    _draw_card(cx2, y3, "5. TOOL CALL",
               "<think>skill loaded; get eval data</think>\nEmits tags: <tool>eval fen=...</tool>",
               "#15221c", "#27ae60", "#2ecc71")
    _draw_card(cx1, y3, "6. HARNESS EXECUTION",
               "Harness runs engine or sandboxed python script\nReturns grounded DATA (e.g. '+1.4' / move)",
               "#1c202a", "#7a8699", "#7a8699")

    # Row 4: Step 7 (Goal-Met Check) -> Step 8 (Grounded Answer)
    _draw_card(cx2, y4, "7. GOAL-MET SELF CHECK",
               "<think>goal met; reply now</think>\nChecks off checklist, exits thinking loop",
               "#15221c", "#27ae60", "#2ecc71")
    _draw_card(cx1, y4, "8. GROUNDED REPLY (Output)",
               "Plain text response to user: 'Ba6'\nNo tags, purely grounded in tool results",
               "#2c2518", "#c8a24a", "#c8a24a")

    # Draw arrows
    # Arrow 1: User Prompt -> Goal Commit
    _draw_arrow((cx1 + w/2, y1 + h/2), (cx2 - w/2, y1 + h/2), color="#7a8699")
    # Arrow 2: Goal Commit -> Routing Decision
    _draw_arrow((cx2, y1), (cx2, y2 + h), color="#3498db")
    # Arrow 3: Routing Decision -> Skill Catalog
    _draw_arrow((cx2 - w/2, y2 + h/2), (cx1 + w/2, y2 + h/2), color="#27ae60")
    # Arrow 4: Skill Catalog -> Tool Call
    _draw_arrow((cx1, y2), (cx2, y3 + h), color="#7a8699")
    # Arrow 5: Tool Call -> Harness Execution
    _draw_arrow((cx2 - w/2, y3 + h/2), (cx1 + w/2, y3 + h/2), color="#27ae60")
    # Arrow 6: Harness Execution -> Goal-Met Check
    _draw_arrow((cx1, y3), (cx2, y4 + h), color="#7a8699")
    # Arrow 7: Goal-Met Check -> Grounded Reply
    _draw_arrow((cx2 - w/2, y4 + h/2), (cx1 + w/2, y4 + h/2), color="#c8a24a")

    # Multi-step loop arc
    _draw_arrow((cx1 - w/2, y3 + h/2), (cx2 + w/2, y3 + h/2), color="#2ecc71", connection_style="arc3,rad=-0.45")
    ax.text(0.5, y3 + h * 1.5, "loop: decide → act → read → decide again",
            ha="center", va="center", fontsize=8.2, color="#2ecc71", fontweight="bold")

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
    ax.add_artist(Line2D([0.38, 0.62], [0.90, 0.90], color="#c8a24a", lw=2.0))

    cards = [
        ("FAST", "#7a8699", "No <goal>\nNo <think> tags", "“push Ba6”", "Direct action path", ["In", "Act", "Out"]),
        ("THINK", "#3498db", "<goal> once\n<think> every step", "“yo, e6 for me”", "Deep multi-turn analysis", ["Goal", "Think", "Tag"]),
        ("AUTO", "#2ecc71", "<goal> once\n<think> only on\nhard choices", "“play Qa4+ pls”", "Balanced speed & depth", ["Goal", "Silent", "Think*"]),
        ("PLAN", "#c8a24a", "<goal> all asks\n<plan> checklist", "“debug + break down”", "For compound tasks", ["Goal", "Plan", "Check"]),
    ]

    w, gap, y0, h = 0.225, 0.024, 0.22, 0.60
    x0 = 0.5 - (4 * w + 3 * gap) / 2

    for i, (name, ec, rule, ex, desc, steps) in enumerate(cards):
        x = x0 + i * (w + gap)
        # Card Background
        card = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.012", fc="#161a23", ec=ec, lw=2.0)
        ax.add_patch(card)

        # Header Title
        ax.text(x + w/2, y0 + h - 0.05, name, ha="center", va="top",
                fontsize=16, fontweight="bold", color=ec)
        
        # Rule text
        ax.text(x + w/2, y0 + h - 0.12, rule, ha="center", va="top",
                fontsize=9.2, color="#ffffff", linespacing=1.3)

        # Draw mini sequence flow
        yc = y0 + h * 0.44
        n_steps = len(steps)
        xs = [x + w * (0.16 + 0.68 * j / (n_steps - 1)) for j in range(n_steps)]
        box_w, box_h = 0.038, 0.046
        
        for j, step_lbl in enumerate(steps):
            box_x = xs[j] - box_w / 2
            box_y = yc - box_h / 2
            step_box = FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.002",
                                      fc="#1e2530", ec=ec, lw=1.0)
            ax.add_patch(step_box)
            ax.text(xs[j], yc, step_lbl, ha="center", va="center", fontsize=6.8, fontweight="bold", color="#ffffff")
            
            if j < n_steps - 1:
                arrow = mpatches.FancyArrowPatch((xs[j] + box_w/2 + 0.002, yc), (xs[j+1] - box_w/2 - 0.002, yc),
                                                 arrowstyle="-|>", lw=1.0, color="#555", mutation_scale=8)
                ax.add_patch(arrow)

        # Mode description
        ax.text(x + w/2, y0 + 0.12, desc, ha="center", va="bottom",
                fontsize=8.5, color="#7a8699")

        # Example text
        ax.text(x + w/2, y0 + 0.05, ex, ha="center", va="bottom",
                fontsize=8.5, color=ec, fontstyle="italic")

    # bottom line
    fig.text(0.5, 0.11, "Same contract: <skill> loads guidance · <tool> calls a function · one per step.",
             ha="center", fontsize=10.5, color="#dcdde1")
    fig.text(0.5, 0.05, "A 4B model — we trained the reasoning IN, and the restraint to not over-think.",
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
