"""Eval-bar caching: snapshot() ran a depth-18 Stockfish eval on EVERY call (switch/sync/move/poll),
which made session switching slow. A position's eval is identical for everyone, so it's cached by
(engine-choice, position FEN). Second look at the same board = cache hit, no engine call. CPU; a
counting fake stands in for the engine (no Stockfish needed)."""
import chess

from backend import eval_engines, state_api


class CountingEval:
    """Stands in for the bar engine; records how many real evals it computed."""
    def __init__(self):
        self.calls = 0

    def eval_white_cp(self, board, depth):
        self.calls += 1
        return ("cp", 35)


def _patch(monkeypatch, counter):
    state_api._EVAL_CACHE.clear()
    monkeypatch.setattr(eval_engines, "current", lambda: "stockfish")
    monkeypatch.setattr(eval_engines, "bar_engine", lambda eng: counter)


def test_second_eval_of_same_position_hits_cache(monkeypatch):
    c = CountingEval(); _patch(monkeypatch, c)
    board = chess.Board(); board.push_san("e4")        # a non-start position
    a = state_api.eval_bar(None, board)
    b = state_api.eval_bar(None, chess.Board(board.fen()))   # same position, fresh object
    assert a == b
    assert c.calls == 1                                # computed once, served from cache the 2nd time


def test_different_positions_each_compute(monkeypatch):
    c = CountingEval(); _patch(monkeypatch, c)
    b1 = chess.Board(); b1.push_san("e4")
    b2 = chess.Board(); b2.push_san("d4")
    state_api.eval_bar(None, b1)
    state_api.eval_bar(None, b2)
    assert c.calls == 2                                # distinct positions -> distinct evals


def test_start_and_gameover_never_call_engine(monkeypatch):
    c = CountingEval(); _patch(monkeypatch, c)
    state_api.eval_bar(None, chess.Board())            # startpos -> short-circuit 0.00
    fools = chess.Board()
    for u in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        fools.push_uci(u)                              # checkmate
    state_api.eval_bar(None, fools)                    # game over -> result text
    assert c.calls == 0
