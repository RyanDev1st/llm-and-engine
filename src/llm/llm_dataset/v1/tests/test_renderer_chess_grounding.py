"""Stockfish-free grounding tests for slice-A (move) rows: the played move must
be legal in the row's FEN, the tool echo must match the real backend, and
board_state must report the real side to move. Catches the hardcoded-e4 bug."""
import re

import chess

from llm_dataset.v1.board_facts import move_echo
from llm_dataset.v1.renderer.chess import render_chess_row
from llm_dataset.v1.sampler import plan_scenarios

MOVE = re.compile(r"<tool>\s*move\s+san=([^\s<]+)\s*</tool>")


class _FakePos:
    """Stand-in for StockfishAnnotator output — no engine needed."""
    def __init__(self, fen: str) -> None:
        self.fen = fen
        b = chess.Board(fen)
        first_legal = next(iter(b.legal_moves))
        self.best_san = b.san(first_legal)
        self.best_line_sans = [self.best_san]
        self.score_cp = 12
        self.depth = 12
        self.threats_san = None
        self.mate = None


class _FakeAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _FakePos:
        return _FakePos(fen)


def _rows(n: int):
    scenarios = plan_scenarios({"A": n}, seed=2026)
    return [render_chess_row(s, _FakeAnnotator()) for s in scenarios]


def _played(row):
    for m in row["messages"]:
        if m["role"] == "assistant":
            hit = MOVE.findall(m["content"])
            if hit:
                return hit[-1]
    return None


def test_every_played_move_is_legal_in_its_fen():
    for row in _rows(25):
        san = _played(row)
        assert san is not None, "slice-A row emitted no move"
        chess.Board(row["position_fen"]).parse_san(san)  # raises if illegal


def test_tool_echo_after_move_matches_backend():
    for row in _rows(25):
        ms = row["messages"]
        i = next(idx for idx, m in enumerate(ms)
                 if m["role"] == "assistant" and "move san=" in m["content"])
        san = _played(row)
        assert ms[i + 1]["role"] == "tool"
        assert ms[i + 1]["content"] == move_echo(row["position_fen"], san)


def test_board_state_turn_matches_fen_side():
    for row in _rows(25):
        stm = "white" if row["position_fen"].split()[1] == "w" else "black"
        bs = [m["content"] for m in row["messages"]
              if m["role"] == "tool" and m["content"].startswith("board_state:")]
        for line in bs:
            assert f"turn={stm}" in line


def test_played_moves_are_diverse_not_monoculture():
    moves = {_played(r) for r in _rows(25)}
    assert len(moves) > 1, f"move monoculture: {moves}"


def test_slice_b_legal_moves_call_satisfies_required_square_arg():
    from llm_dataset.v1.renderer.chess import render_chess_row
    from llm_dataset.v1.validate import validate_row
    scenarios = plan_scenarios({"B": 8}, seed=2026)
    for s in scenarios:
        row = render_chess_row(s, _FakeAnnotator())
        bad = [v for v in validate_row(row) if v.rule == "args_match_schema"]
        assert not bad, bad
        calls = [m["content"] for m in row["messages"]
                 if m["role"] == "assistant" and "legal_moves" in m["content"]]
        assert calls and "square=" in calls[0]
