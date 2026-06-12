import chess

from backend.game import Game
from backend.thinking.prompts import board_facts, facts_summary, build_controller_system, build_narrator_system


def test_board_facts_reads_live_board():
    g = Game()
    bf = board_facts(g)
    assert "turn=white" in bf and "legal_moves=20" in bf and "last_move=none" in bf


def test_facts_summary_compacts_results():
    assert facts_summary([]) == "(none yet)"
    assert facts_summary([("eval", "score: +0.30")]) == "eval→score: +0.30"


def test_controller_system_has_manifest_and_outstanding():
    s = build_controller_system("", None, "best move and eval", "", ["eval"])
    assert "AVAILABLE TOOLS" in s          # full manifest present (it can route)
    assert "DONE" in s and "OUTSTANDING" in s and "eval" in s


def test_narrator_system_has_no_tool_manifest():
    s = build_narrator_system("")
    assert "AVAILABLE TOOLS" not in s      # narrator cannot route
    assert "grounded" in s.lower()
