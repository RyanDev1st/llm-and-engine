"""CPU smoke gate for the report assets — renders the confusion matrix (with its description baked
in), the cross-model performance line chart, and a chat card, all from SEED/SAMPLE data with NO model
and NO GPU. The benchmark notebook runs this FIRST: in ~5s you confirm every renderer works and can
eyeball the layout, THEN trust the hours-long GPU cells and step away. The real cells overwrite with
real numbers under non-SAMPLE filenames, so the two never collide.
  python -m llm_training.report.gate [--out docs/findings/report_assets]"""
from __future__ import annotations

import argparse
from pathlib import Path

# A representative-shaped matrix (NOT measured — clearly labelled SAMPLE in the image caption).
_SAMPLE_CM = {"skill": {"skill": 210, "tool": 14, "none": 6},
              "tool": {"skill": 12, "tool": 240, "none": 8},
              "none": {"skill": 5, "tool": 7, "none": 170}}
_SAMPLE_TURNS = [
    {"prompt": "yo hows my position looking", "reply": "Roughly equal — you're a touch better; "
     "finish developing before you push pawns.", "secs": 3.4, "gen_tokens": 71, "tok_s": 20.9},
    {"prompt": "whats the best move", "reply": "Nf3 — it develops and eyes e5.", "secs": 2.1,
     "gen_tokens": 38, "tok_s": 18.1},
]


def cpu_smoke(outdir: str | Path) -> list[Path]:
    """Render all three PPT assets from seed data; assert each is a real PNG. Returns the paths."""
    from llm_training.eval_confusion import confusion_caption
    from llm_training.report import chart_data as D
    from llm_training.report import chat_suites, ppt_charts
    chat_suites.validate(chat_suites.PLAIN_CHATS)          # the hand-written suites are well-formed
    chat_suites.validate(chat_suites.WEB_CHATS)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = "SAMPLE (seed data — the real run overwrites this).\n\n" + confusion_caption(_SAMPLE_CM, 50, 68)
    paths = [
        ppt_charts.confusion_matrix(_SAMPLE_CM, ["skill", "tool", "none"],
                                    outdir / "SAMPLE-confusion.png", cap),
        ppt_charts.model_lines(D.MODELS, outdir / "SAMPLE-model-lines.png"),
        ppt_charts.chat_card("SAMPLE — Section 1 (bare harness)", _SAMPLE_TURNS,
                             outdir / "SAMPLE-chat-card.png", "seed data — replaced by the real run"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000, f"render failed: {p}"
    return paths


def main() -> None:
    from llm_training.report import chart_data as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(D.REPO / "docs" / "findings" / "report_assets"))
    args = ap.parse_args()
    paths = cpu_smoke(args.out)
    print("GATE OK -- report renderers all produced an image:")
    for p in paths:
        print(f"  [ok] {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
