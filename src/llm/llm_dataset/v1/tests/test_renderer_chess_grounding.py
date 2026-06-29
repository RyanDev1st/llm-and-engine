"""Stockfish-free grounding tests for slice-A (move) rows: the played move must
be legal in the row's FEN, the tool echo must match the real backend, and
board_state must report the real side to move. Catches the hardcoded-e4 bug.

v5-native: actions are STRUCTURED tool_calls on the assistant message, read via
tags.tool_calls_of (not <tool>…</tool> text)."""
import chess

from llm_dataset.v1.board_facts import move_echo
from llm_dataset.v1.renderer.chess import render_chess_row
from llm_dataset.v1.renderer.tags import tool_calls_of
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row


class _FakePos:
    """Stand-in for StockfishAnnotator output — no engine needed."""
    def __init__(self, fen: str) -> None:
        self.fen = fen
        b = chess.Board(fen)
        first_legal = next(iter(b.legal_moves))
        self.best_san = b.san(first_legal)
        self.best_line_sans = [self.best_san]
        self.score_cp = 12
        self.score_kind = "cp"
        # top_moves: (san, cp) pairs — slice B/E now emit a best_move result, so the fake
        # must mirror AnnotatedPosition's field (it predated B calling the engine).
        self.top_moves = tuple((b.san(m), 12) for m in list(b.legal_moves)[:3])
        self.depth = 12
        self.threats_san = None
        self.mate = None


class _FakeAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _FakePos:
        return _FakePos(fen)


def _rows(n: int):
    scenarios = plan_scenarios({"A": n}, seed=2026)
    return [render_chess_row(s, _FakeAnnotator()) for s in scenarios]


def _move_calls(row):
    return [tc for m in row["messages"] if m["role"] == "assistant"
            for tc in tool_calls_of(m) if tc["name"] == "move"]


def _played(row):
    calls = _move_calls(row)
    return str(calls[-1]["arguments"].get("san")) if calls else None


def test_every_played_move_is_legal_in_its_fen():
    for row in _rows(25):
        san = _played(row)
        assert san is not None, "slice-A row emitted no move"
        chess.Board(row["position_fen"]).parse_san(san)  # raises if illegal


def test_tool_echo_after_move_matches_backend():
    for row in _rows(25):
        ms = row["messages"]
        i = next(idx for idx, m in enumerate(ms)
                 if m["role"] == "assistant" and any(tc["name"] == "move" for tc in tool_calls_of(m)))
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


class _MatePos:
    """A forced-mate annotation: score_cp is the MATE DISTANCE (not centipawns),
    score_kind='mate' — the case where the old renderer emitted a bogus pawn number."""
    def __init__(self, fen: str) -> None:
        self.fen = fen
        b = chess.Board(fen)
        legal = list(b.legal_moves)
        self.best_san = b.san(legal[0])
        self.best_line_sans = [b.san(m) for m in legal[:2]]
        self.score_cp = 2          # mate in 2 (white to move in our test FENs)
        self.score_kind = "mate"
        self.depth = 12
        self.threats_san = None
        self.top_moves = tuple((b.san(m), 100000) for m in legal[:3])


class _MateAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _MatePos:
        return _MatePos(fen)


def test_best_move_score_mate_grounds_and_is_noop_on_cp():
    from llm_dataset.v1.renderer.text import best_move_score, score_pawns

    class _W:  # mate in 3 for white
        score_kind, score_cp = "mate", 3

    class _B:  # mate in 2 for black
        score_kind, score_cp = "mate", -2

    class _Cp:  # non-mate must stay byte-identical to score_pawns (no spurious churn)
        score_kind, score_cp = "cp", 137
    assert best_move_score(_W()) == "mate in 3 for white"
    assert best_move_score(_B()) == "mate in 2 for black"
    assert best_move_score(_Cp()) == score_pawns(_Cp())


def test_e_slice_mate_final_is_tool_grounded():
    """The 'forced mate in N' final must have 'mate' in a tool result (the [G] preflight
    hole): both the top-form and series-form best_move results must carry it."""
    scenarios = plan_scenarios({"E": 24}, seed=2026)
    saw = False
    for s in scenarios:
        row = render_chess_row(s, _MateAnnotator())
        final = row["messages"][-1]["content"].lower()
        tools = " ".join(m["content"] for m in row["messages"] if m["role"] == "tool").lower()
        if "mate" in final:
            saw = True
            assert "mate in" in tools, f"mate claim not grounded in {row['id']}: tools={tools!r}"
    assert saw, "expected at least one mate final"


def test_slice_b_legal_moves_call_satisfies_required_square_arg():
    from llm_dataset.v1.validate import validate_row
    scenarios = plan_scenarios({"B": 8}, seed=2026)
    for s in scenarios:
        row = render_chess_row(s, _FakeAnnotator())
        bad = [v for v in validate_row(row) if v.rule == "args_match_schema"]
        assert not bad, bad
        calls = [tc for m in row["messages"] if m["role"] == "assistant"
                 for tc in tool_calls_of(m) if tc["name"] == "legal_moves"]
        assert calls and "square" in calls[0]["arguments"]


def _final(row):
    return row["messages"][-1]["content"].strip()


def test_chess_analysis_finals_are_not_question_monoculture():
    scenarios = plan_scenarios({"D": 12, "E": 12, "G": 12, "H": 12}, seed=2026)
    finals = [_final(render_chess_row(s, _FakeAnnotator())) for s in scenarios]
    question_finals = [text for text in finals if text.endswith("?")]
    assert len(question_finals) <= len(finals) // 2


def test_material_board_read_answers_directly_without_offer_closer():
    scenarios = plan_scenarios({"H": 8}, seed=2026)
    finals = [_final(render_chess_row(s, _FakeAnnotator())) for s in scenarios]
    assert finals
    assert not any(text.endswith("?") for text in finals)
    assert all("You've still got" in text for text in finals)


def test_thanks_and_greetings_are_direct_no_tool_replies():
    scenarios = plan_scenarios({"J": 24}, seed=2026)
    for s in scenarios:
        row = render_chess_row(s, _FakeAnnotator())
        calls = [tc for m in row["messages"] if m["role"] == "assistant" for tc in tool_calls_of(m)]
        assert calls == [], (row["id"], calls)
        assert row["selected_skills"] == []
        assert row["expected_tool_calls"] == []
        assert not row["messages"][-1]["content"].rstrip().endswith("?")
        assert validate_row(row) == []
