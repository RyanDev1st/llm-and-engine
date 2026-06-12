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
