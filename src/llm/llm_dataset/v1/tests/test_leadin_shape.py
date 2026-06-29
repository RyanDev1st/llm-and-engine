"""Conversational shape (Stockfish-free): every assistant action turn carries
exactly ONE structured tool call (native: the tool-call turn has no prose — the
grounded narration is the FINAL turn), guiding questions are a bounded mix rather
than a final-answer monoculture, and expected_tool_calls survives the skill load
(load_skill is a native tool call, so expected_tool_calls is the TOOLS only)."""
import chess

from llm_dataset.v1.renderer.chess import render_chess_row
from llm_dataset.v1.renderer.tags import tool_calls_of
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row


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
        self.top_moves = ((self.best_san, self.score_cp), (self.best_san, self.score_cp - 30))


class _FakeAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _FakePos:
        return _FakePos(fen)


def _chess_rows(slice_name, n):
    return [render_chess_row(s, _FakeAnnotator()) for s in plan_scenarios({slice_name: n}, seed=99)]


def _action_msgs(row):
    return [m for m in row["messages"] if m["role"] == "assistant" and tool_calls_of(m)]


def test_chess_action_turns_have_exactly_one_tool():
    for slice_name in ("A", "D", "E", "F", "G", "H"):
        for row in _chess_rows(slice_name, 4):
            assert validate_row(row) == [], (slice_name, validate_row(row))
            for m in _action_msgs(row):
                assert len(tool_calls_of(m)) == 1, (slice_name, m)


def test_chess_coaching_finals_mix_direct_answers_and_guiding_questions():
    finals = []
    for slice_name in ("A", "B", "D", "E", "F", "G", "H"):
        for row in _chess_rows(slice_name, 12):
            final = row["messages"][-1]
            assert not tool_calls_of(final)          # final is a plain answer
            finals.append(final["content"].rstrip())
    question_count = sum(text.endswith("?") for text in finals)
    assert 0 < question_count <= len(finals) // 2


def test_knowledge_and_greeting_finals_stay_statements():
    for slice_name in ("I", "J", "K", "C"):
        for row in _chess_rows(slice_name, 2):
            assert not row["messages"][-1]["content"].rstrip().endswith("?")


def test_expected_tool_calls_survive_leadin():
    row = _chess_rows("D", 1)[0]
    # skills load via the native load_skill call, so expected_tool_calls is TOOLS only.
    assert row["expected_tool_calls"][0] == "board_state"
    assert "eval" in row["expected_tool_calls"]
    loaded = [tc["arguments"].get("name") for m in row["messages"] if m["role"] == "assistant"
              for tc in tool_calls_of(m) if tc["name"] == "load_skill"]
    assert "chess-coach" in loaded
