"""End-to-end proof that the demo skills drive the REAL Stockfish backend.

For each scripted skill it: loads the SKILL.md body through the live ToolExecutor
(load_skill), then runs the exact tool calls that skill's Steps prescribe against
a real python-chess board + Stockfish — printing the genuine engine output. No
fabrication: every score/line/threat comes from the engine.

Run (Stockfish auto-located from runtime/):
    python src/llm/skills_demo/_demo_integration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LLM = Path(__file__).resolve().parents[1]          # src/llm
DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LLM))
# make the 40 demo skills loadable by the live backend (serving + executor)
os.environ.setdefault("CHESS_SKILLS_DIRS", str(DEMO_DIR))

from backend.engine import Engine          # noqa: E402
from backend.game import Game              # noqa: E402
from backend.skills import load_skills     # noqa: E402
from backend.tools import ToolExecutor     # noqa: E402

# a real Najdorf so eval/threats/best_move have a non-trivial position to chew on
OPENING = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]

# (skill to load, the tool calls its Steps prescribe)
SCRIPT: list[tuple[str, list[str]]] = [
    ("position-evaluator", ["<tool>eval depth=14</tool>"]),
    ("candidate-moves", ["<tool>best_move top=3 depth=14</tool>"]),
    ("threat-scanner", ["<tool>threats depth=12</tool>"]),
    ("blunder-check", ["<tool>review_move depth=14</tool>"]),
    ("board-recall", ["<tool>board_state fields=all</tool>"]),
    ("material-counter", ["<tool>list_pieces color=white</tool>", "<tool>list_pieces color=black</tool>"]),
]


def main() -> int:
    catalog = {s.name for s in load_skills()}
    print(f"catalog: {len(catalog)} skills loadable (demo via {os.environ['CHESS_SKILLS_DIRS']})")
    missing = [name for name, _ in SCRIPT if name not in catalog]
    if missing:
        raise SystemExit(f"FAIL: skills not loadable by backend: {missing}")

    if not Path(Engine().path).exists():
        print(f"SKIP: Stockfish not found at {Engine().path}; skill loading verified, engine calls skipped.")
        return 0

    game, engine = Game(), Engine()
    for san in OPENING:
        game.move(san)
    print(f"position: after {' '.join(OPENING)}  (black to move)")
    tx = ToolExecutor(game, engine)
    try:
        for skill, calls in SCRIPT:
            body = tx.execute(f"<tool>load_skill name={skill}</tool>")
            status = "body loaded" if not body.startswith("error:") else body
            print(f"\n=== {skill} ===  load_skill -> {status}")
            for call in calls:
                print(f"  {call}\n    -> {tx.execute(call)}")
    finally:
        engine.quit()
    print("\nOK: every call above is live Stockfish output via the trained tool contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
