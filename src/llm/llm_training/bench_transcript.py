"""Capture a REAL end-to-end agent transcript for the report — no fabrication. Runs the full
CoachLoop (the real serve loop) on held-out STRESS prompts with the real life-skills bundle
enabled, so each turn actually routes -> loads a real skill body -> calls a real tool -> narrates
the real result. Records the goal/think/skill/tool steps + the final reply as a readable markdown
conversation, written to docs/findings/<date>-agent-transcript.md.

This is the artifact for the report's "the agent in action" section: it shows the harness loop
operating on domains ABSENT from training (cooking/music/wellness/tax), proving route-by-
description + grounded narration end to end. Run on a GPU box (Kaggle/Colab) with the adapter:
  python -m llm_training.bench_transcript --adapter <best>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_training.eval_confusion import _load_model  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

# A few representative held-out cases (mode for the system signal). Chosen to show: skill+tool
# chain on an unseen domain, messy phrasing, and a clean decline.
_PROMPTS = [
    ("scale my cookie recipe from 12 up to 30 servings", "auto"),
    ("wanna make like 3x the cookies lol how do i", "auto"),
    ("set a metronome to 120 bpm so i can practice", "auto"),
    ("convert 5 miles into kilometers", "fast"),
    ("stressed af rn need to chill n breathe for a bit", "auto"),
    ("what is the capital of France?", "auto"),
]


def _fmt_turn(prompt: str, mode: str, events: list[dict], result: dict) -> list[str]:
    """One prompt's transcript block: the user turn, each captured step (goal/think/skill/tool +
    its real result), then the final reply — exactly what the agent produced."""
    L = [f"### [{mode}] User: {prompt}", ""]
    for ev in events:
        t = ev.get("type")
        if t == "goal":
            L.append(f"- 🎯 **goal** — {ev['content']}")
        elif t == "plan":
            L.append("- 📋 **plan**\n```\n" + ev["content"] + "\n```")
        elif t == "think":
            L.append(f"- 💭 *think* — {ev['content']}")
        elif t == "tool":
            verb = "skill" if ev.get("name") == "skill" else "tool"
            L.append(f"- 🔧 **{verb}** `{ev.get('call', '')}` → `{ev.get('result', '')[:160]}`")
    L += ["", f"**Coach:** {result.get('reply', '').strip()}", "", "---", ""]
    return L


def capture(model, prompts=_PROMPTS) -> str:
    """Run each prompt through a fresh CoachLoop (real loop, real life-skills bundle) and return
    the assembled markdown transcript. Each turn starts clean (no cross-turn state) so the block
    is self-contained for the report."""
    from backend.game import Game
    from backend.inference import CoachLoop
    from backend.tools import ToolExecutor
    pc = {"installed": ["life-skills"], "enabled": ["life-skills"], "marketplace": []}
    body: list[str] = []
    for prompt, mode in prompts:
        loop = CoachLoop(model, ToolExecutor(Game(), None, pc), plugin_context=pc)
        events: list[dict] = []
        result = loop.respond([], prompt, coverage=True, on_event=events.append, reasoning_mode=mode)
        body += _fmt_turn(prompt, mode, events, result)
        print(f"  captured: {prompt[:50]!r} -> {result.get('reply','')[:60]!r}", flush=True)
    return "\n".join(body)


def write_transcript(transcript: str, date_str: str) -> Path:
    out = REPO / "docs" / "findings" / f"{date_str}-agent-transcript.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("Parent: docs/reference/harness-architecture.md\n\n"
              "# Agent transcript — Gemma 4 E4B chess-coach on UNSEEN domains\n\n"
              "Real end-to-end runs of the serve loop (CoachLoop) on held-out prompts over the real\n"
              "`life-skills` bundle (cooking/music/wellness/tax — absent from training). Each turn:\n"
              "route by reading the in-context description -> load a real skill body -> call a real\n"
              "tool -> narrate the real result. Captured verbatim, not fabricated.\n\n")
    out.write_text(header + transcript + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel)")
    ap.add_argument("--server", default="http://127.0.0.1:7861", help="model service URL")
    args = ap.parse_args()
    from datetime import date
    model = _load_model(args)
    transcript = capture(model)
    out = write_transcript(transcript, f"{date.today():%Y-%m-%d}")
    print("\n" + transcript, flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
    from llm_training.clean_exit import flush_and_exit
    flush_and_exit()   # benign torch/CUDA exit-time SIGABRT must not fail the notebook run
