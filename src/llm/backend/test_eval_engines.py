"""Selectable eval-bar engine (Stockfish | custom LiquidChess). The custom path needs
no Stockfish, so it's testable headless; state_api.eval_bar tags which engine produced
the value."""
import chess

from backend import eval_engines, state_api
from backend.game import Game


def _white_up_a_pawn() -> Game:
    g = Game()
    for s in ["e4", "d5", "exd5"]:
        g.move(s)
    return g


def test_engine_timeout_is_env_configurable(monkeypatch):
    # The completion eval runs many deep analyses per chess row; a configurable per-call timeout
    # lets a benchmark bound engine wait (CHESS_SF_TIMEOUT) without touching depth (which would
    # change the tool output). Construction is lazy (no Stockfish popen), so this is headless.
    from backend.engine import Engine
    monkeypatch.delenv("CHESS_SF_TIMEOUT", raising=False)
    assert Engine().timeout == 5.0                       # default unchanged for production
    monkeypatch.setenv("CHESS_SF_TIMEOUT", "2.0")
    assert Engine().timeout == 2.0                       # env override for the bench
    assert Engine(timeout=1.0).timeout == 1.0            # explicit arg still wins


def test_engine_toggle_and_available():
    assert "stockfish" in eval_engines.available() and "custom" in eval_engines.available()
    assert eval_engines.set_engine("custom") == "custom"
    assert eval_engines.current() == "custom"
    assert eval_engines.set_engine("bogus") == "custom"   # invalid ignored
    eval_engines.set_engine("stockfish")


def test_custom_evaluator_material_score():
    eval_engines.set_engine("custom")
    try:
        ev = state_api.eval_bar(None, _white_up_a_pawn().board)   # no Stockfish needed
        assert ev["engine"] == "custom"
        assert ev["kind"] == "cp" and ev["cp"] == 100 and ev["text"] == "+1.00"
        assert ev["bar"] > 50                                       # white favoured
    finally:
        eval_engines.set_engine("stockfish")


def test_custom_start_position_is_equal():
    eval_engines.set_engine("custom")
    try:
        ev = state_api.eval_bar(None, Game().board)
        assert ev["bar"] == 50 and ev["engine"] == "custom"
    finally:
        eval_engines.set_engine("stockfish")
