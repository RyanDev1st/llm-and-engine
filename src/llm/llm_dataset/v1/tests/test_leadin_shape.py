"""Conversational shape (Stockfish-free): every assistant action turn is a
short lead-in sentence + exactly one <tool> call, coaching finals end with a
guiding question, and expected_tool_calls survives the lead-in prefix."""
import re

import chess

from llm_dataset.v1.renderer.chess import render_chess_row
from llm_dataset.v1.renderer.universality import render_universality_row
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row

_TOOL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")


class _FakePos:
    def __init__(self, fen: str) -> None:
        self.fen = fen
        b = chess.Board(fen)
        self.best_san = b.san(next(iter(b.legal_moves)))
        self.best_line_sans = [self.best_san, self.best_san]
        self.score_cp = 80
        self.score_kind = "cp"
        self.depth = 12
        self.threats_san = None
        self.mate = None


class _FakeAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _FakePos:
        return _FakePos(fen)


def _chess_rows(slice_name, n):
    return [render_chess_row(s, _FakeAnnotator()) for s in plan_scenarios({slice_name: n}, seed=99)]


def _assistant_tool_msgs(row):
    return [m["content"] for m in row["messages"]
            if m["role"] == "assistant" and "<tool>" in m["content"]]


def test_chess_action_turns_have_leadin_and_one_tool():
    for slice_name in ("A", "D", "E", "F", "G", "H"):
        for row in _chess_rows(slice_name, 4):
            assert validate_row(row) == [], (slice_name, validate_row(row))
            for content in _assistant_tool_msgs(row):
                assert not content.startswith("<tool>"), f"{slice_name}: no lead-in: {content!r}"
                assert len(_TOOL.findall(content)) == 1, f"{slice_name}: {content!r}"


def test_chess_coaching_finals_end_with_question():
    for slice_name in ("A", "B", "D", "E", "F", "G", "H"):
        for row in _chess_rows(slice_name, 3):
            final = row["messages"][-1]["content"]
            assert "<tool>" not in final
            assert final.rstrip().endswith("?"), (slice_name, final)


def test_knowledge_and_greeting_finals_stay_statements():
    for slice_name in ("I", "J", "K", "C"):
        for row in _chess_rows(slice_name, 2):
            assert not row["messages"][-1]["content"].rstrip().endswith("?")


def test_expected_tool_calls_survive_leadin():
    row = _chess_rows("D", 1)[0]
    assert row["expected_tool_calls"][:2] == ["load_skill", "board_state"]
    assert "eval" in row["expected_tool_calls"]


def test_universality_action_turns_have_leadin_and_one_tool():
    for slice_name in ("V1_E_board_grounding", "V1_G_multi_tool_budget",
                       "V1_H_error_recovery", "V1_K_adversarial_injection",
                       "V1_N_human_chat_skill_bridge"):
        row = render_universality_row(plan_scenarios({slice_name: 1}, seed=5)[0])
        assert validate_row(row) == [], (slice_name, validate_row(row))
        for content in _assistant_tool_msgs(row):
            assert not content.startswith("<tool>"), f"{slice_name}: {content!r}"
            assert len(_TOOL.findall(content)) == 1, f"{slice_name}: {content!r}"
        assert row["expected_tool_calls"], slice_name
