"""fetch_puzzle: real Lichess puzzle source. Tested OFFLINE — the network call is
monkeypatched, so CI never depends on Lichess. We verify (1) a good payload loads the
board + renders the solution SAN, (2) any failure falls back to the local bank, never
errors or hangs, (3) the dispatcher wires the tool."""
import chess

from backend import online_positions as op
from backend.game import Game
from backend.tools import ToolExecutor


# real Lichess daily-puzzle shape: fen = solver to move, solution = UCI moves.
_PAYLOAD = {"puzzle": {
    "id": "abcd1", "rating": 1822, "themes": ["middlegame", "crushing"],
    "fen": "2kr4/pp1r1pp1/5np1/8/4PBP1/5q1P/P5B1/5RK1 w - - 0 1",
    "solution": ["f1c1", "d7c7", "c1c7"], "lastMove": "d3f3"}}


def test_good_payload_loads_board_and_solution(monkeypatch):
    monkeypatch.setattr(op, "_fetch_puzzle_json", lambda timeout: _PAYLOAD)
    g = Game()
    out = op.fetch_puzzle(g)
    assert "lichess puzzle abcd1" in out and "rating 1822" in out
    assert "themes: middlegame, crushing" in out
    assert g.board.fen().split()[0] == _PAYLOAD["puzzle"]["fen"].split()[0]   # board set
    # f1c1 from that FEN is Rc1 in SAN — grounded, real answer
    assert "answer=Rc1" in out


def test_network_failure_falls_back_to_local_bank(monkeypatch):
    monkeypatch.setattr(op, "_fetch_puzzle_json", lambda timeout: None)
    g = Game()
    out = op.fetch_puzzle(g)
    assert out.startswith("note: online puzzle source unavailable")
    assert "fen=" in out and g.board.fen() != chess.STARTING_FEN   # local puzzle still set


def test_bad_fen_falls_back(monkeypatch):
    monkeypatch.setattr(op, "_fetch_puzzle_json",
                        lambda timeout: {"puzzle": {"fen": "not a fen", "solution": []}})
    g = Game()
    out = op.fetch_puzzle(g)
    assert "using a local puzzle" in out and g.board.fen() != chess.STARTING_FEN


def test_solution_san_is_blank_when_unparseable():
    assert op._solution_san("8/8/8/8/8/8/8/8 w - - 0 1", ["zzzz"]) == ""
    assert op._solution_san("", []) == ""


def test_executor_dispatches_fetch_puzzle(monkeypatch):
    monkeypatch.setattr(op, "_fetch_puzzle_json", lambda timeout: _PAYLOAD)
    g = Game()
    out = ToolExecutor(g, None).execute("<tool>fetch_puzzle</tool>")
    assert "lichess puzzle" in out and g.board.fen() != chess.STARTING_FEN
