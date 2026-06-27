"""what_if — the one new coach tool. Real Stockfish (no mocks): it must weigh a
candidate move against the engine's best, reject illegal/empty input, and NEVER
mutate the live board (it reasons on a copy)."""
from backend.coach_tools import what_if
from backend.engine import Engine
from backend.game import Game


def _game(sans):
    g = Game()
    for san in sans:
        assert g.move(san).startswith("success"), san
    return g


def test_what_if_calls_a_good_move_good():
    g = _game(["e4", "e5", "Nf3", "Nc6"])
    eng = Engine()
    try:
        out = what_if(g, eng, {"san": "Bb5", "depth": "12"})   # the Ruy Lopez — a fine move
        assert out.startswith("what_if:")
        assert "Bb5" in out
    finally:
        eng.quit()


def test_what_if_flags_a_bad_move_against_best():
    g = _game(["e4", "e5"])
    eng = Engine()
    try:
        out = what_if(g, eng, {"san": "Ke2", "depth": "12"})   # the Bongcloud — clearly worse than best
        assert out.startswith("what_if:")
        assert "best is" in out and "gives up" in out
    finally:
        eng.quit()


def test_what_if_rejects_illegal_and_empty():
    g = _game(["e4"])
    eng = Engine()
    try:
        assert what_if(g, eng, {"san": "Qh8", "depth": "12"}).startswith("error")
        assert what_if(g, eng, {"san": "", "depth": "12"}).startswith("error")
    finally:
        eng.quit()


def test_what_if_does_not_mutate_the_board():
    g = _game(["e4", "e5"])
    eng = Engine()
    fen_before = g.board.fen()
    try:
        what_if(g, eng, {"san": "Nf3", "depth": "12"})
        assert g.board.fen() == fen_before
    finally:
        eng.quit()
