"""Integration: the skills_demo catalog loads through the live backend and its
skills drive the real Stockfish tool executor. The engine calls skip cleanly if
the Stockfish binary is absent (e.g. CI), but skill loading is always checked."""
from __future__ import annotations

from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "skills_demo"


def test_demo_catalog_loads_via_env(monkeypatch):
    from backend.skills import load_skills

    monkeypatch.setenv("CHESS_SKILLS_DIRS", str(DEMO_DIR))
    skills = load_skills()
    names = [s.name for s in skills]
    demo_slugs = {p.parent.name for p in DEMO_DIR.glob("*/SKILL.md")}

    assert len(demo_slugs) == 40
    assert demo_slugs <= set(names)              # all 40 served by the backend
    assert len(names) == len(set(names))         # dedup across roots
    assert all(s.description for s in skills)     # every served skill is routable


def test_demo_bodies_reference_official_tools(monkeypatch):
    from llm_dataset.v1.catalog import official_tools
    from backend.skills import load_skills

    tool_names = {t["name"] for t in official_tools()}
    monkeypatch.setenv("CHESS_SKILLS_DIRS", str(DEMO_DIR))
    demo = [s for s in load_skills() if s.name in {p.parent.name for p in DEMO_DIR.glob("*/SKILL.md")}]

    for skill in demo:
        referenced = {t for t in tool_names if t in skill.content}
        assert referenced, f"{skill.name} references no real backend tool"


def test_demo_skills_drive_stockfish(monkeypatch):
    from backend.engine import Engine

    if not Path(Engine().path).exists():
        pytest.skip("stockfish binary not present")

    from backend.game import Game
    from backend.tools import ToolExecutor

    monkeypatch.setenv("CHESS_SKILLS_DIRS", str(DEMO_DIR))
    game, engine = Game(), Engine()
    for san in ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"]:
        game.move(san)
    tx = ToolExecutor(game, engine)
    try:
        assert not tx.execute("<tool>load_skill name=position-evaluator</tool>").startswith("error:")
        assert tx.execute("<tool>eval depth=10</tool>").startswith("score:")
        assert tx.execute("<tool>best_move top=3 depth=10</tool>").startswith("best_moves:")
        assert tx.execute("<tool>threats depth=10</tool>").startswith("threats:")
        white = tx.execute("<tool>list_pieces color=white</tool>")
        black = tx.execute("<tool>list_pieces color=black</tool>")
        assert white != black                     # the material-counter arg fix
    finally:
        engine.quit()
