"""PPT-ready chart builders for the report deck (GPU-free). These differ from `charts.py` in one
way: every image carries its OWN description text baked INTO the PNG, so a slide is one copy-paste
(no separate caption to keep in sync). Three assets:
  confusion_matrix  — the 3-class routing matrix + a plain-English legend + headline numbers below it
  model_lines       — performance across ALL models (E2B prior, E4B base, E4B nf4, E4B Q5_K_M, Q6_K)
  chat_card         — a captured chat section rendered as a slide card (prompt + reply + secs + tok/s)
All deterministic + dependency-light (matplotlib only) so they render the same on Kaggle or locally.
"""
from __future__ import annotations

import textwrap
from pathlib import Path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def confusion_matrix(cm: dict, classes: list[str], out: Path, caption: str,
                     title: str = "Routing verb confusion (held-out val)") -> Path:
    """The gold-vs-predicted matrix with `caption` rendered as a description block UNDER the heatmap,
    inside the same image — so the slide needs no separate legend. cm: {gold: {pred: count}}."""
    plt = _plt()
    mat = [[cm[g][p] for p in classes] for g in classes]
    fig = plt.figure(figsize=(5.4, 5.8))
    ax = fig.add_axes([0.16, 0.40, 0.72, 0.52])            # leave the lower ~38% for the caption
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(classes)), [f"pred\n{c}" for c in classes])
    ax.set_yticks(range(len(classes)), [f"gold {c}" for c in classes])
    thr = (max(max(r) for r in mat) / 2) or 1
    for g in range(len(classes)):
        for p in range(len(classes)):
            ax.text(p, g, mat[g][p], ha="center", va="center",
                    color="white" if mat[g][p] > thr else "black", fontsize=13)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.06, 0.30, caption, ha="left", va="top", fontsize=8.4, wrap=True,
             family="DejaVu Sans")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def model_lines(models: list[dict], out: Path,
                title: str = "Performance across models — E2B → E4B → GGUF quants") -> Path:
    """One line per quality metric across the ordered model axis; tok/s annotated under each model.
    A metric with no measured value at a model is skipped (the line bridges measured points only —
    no fabricated number), matching charts.training_timeline's honesty rule."""
    plt = _plt()
    metrics = [("verb", "routing verb acc"), ("completed", "task completion"),
               ("grounded", "grounded answer")]
    xs = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for key, name in metrics:
        pts = [(x, models[x].get(key)) for x in xs if models[x].get(key) is not None]
        if len(pts) >= 1:
            ax.plot([x for x, _ in pts], [y for _, y in pts], "-o", label=name, zorder=3)
            for x, y in pts:
                ax.text(x, y + 0.015, f"{y:.0%}", ha="center", fontsize=7.5)
    for x in xs:
        ts = models[x].get("tok_s")
        ax.text(x, -0.13, f"{ts:.0f} tok/s" if ts is not None else "tok/s n/a",
                ha="center", va="top", fontsize=7.2, color="#555", transform=ax.get_xaxis_transform())
    ax.set_xticks(xs, [m["label"] for m in models], fontsize=8)
    ax.set_xlim(-0.5, len(models) - 0.5)            # show ALL model slots even if a metric has gaps
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (held-out)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def chat_card(section_title: str, turns: list[dict], out: Path, subtitle: str = "") -> Path:
    """A captured chat SECTION as a single slide card: each turn shows the user prompt, a reply
    snippet, and its measured `secs`/`tok_s`. Pure layout over already-captured turns (no model)."""
    plt = _plt()
    rows = turns[:7]                                       # a slide holds ~7 turns legibly
    fig = plt.figure(figsize=(9.2, 1.05 + 1.18 * len(rows)))
    fig.text(0.04, 0.975, section_title, fontsize=13, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.04, 0.925, subtitle, fontsize=8.5, color="#555", va="top")
    y = 0.85                                              # leave the top ~15% for title + subtitle
    step = 0.85 / max(len(rows), 1)
    for t in rows:
        user = textwrap.shorten(t.get("prompt", ""), 96, placeholder=" …")
        reply = textwrap.fill(textwrap.shorten(t.get("reply", ""), 240, placeholder=" …"), 104)
        meta = f"{t.get('secs', 0):.1f}s · {t.get('gen_tokens', 0)} tok · {t.get('tok_s', 0):.0f} tok/s"
        fig.text(0.04, y, f"User: {user}", fontsize=9.5, fontweight="bold", va="top")
        fig.text(0.965, y, meta, fontsize=8, color="#1a6", ha="right", va="top")
        fig.text(0.04, y - step * 0.30, f"Coach: {reply}", fontsize=8.6, va="top", color="#222")
        y -= step
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out
