"""Play-vs-engine opponent: the neural selector picks the side-to-move's move, with a random
fallback so the board never stalls. These tests force the deterministic paths (no torch load —
the real neural path is verified manually: loads nee_latest.pt, plays g1f3). Contract: choose()
returns a legal move, or a clear not-ok for a bad/finished position."""
import chess

from backend import opponent


def test_bad_fen_is_not_ok():
    assert opponent.choose("not a fen")["ok"] is False
    assert opponent.choose("")["ok"] is False


def test_game_over_is_not_ok():
    b = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:        # fool's mate
        b.push_uci(uci)
    assert b.is_game_over()
    assert opponent.choose(b.fen())["ok"] is False


def test_neural_move_used_when_selector_succeeds(monkeypatch):
    class Fake:
        def choose_move(self, board):
            return chess.Move.from_uci("e2e4")
    monkeypatch.setattr(opponent, "_selector", lambda: Fake())
    assert opponent.choose(chess.STARTING_FEN) == {"ok": True, "uci": "e2e4", "source": "neural"}


def test_random_fallback_when_selector_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("no torch / checkpoint")
    monkeypatch.setattr(opponent, "_selector", boom)
    out = opponent.choose(chess.STARTING_FEN)
    assert out["ok"] and out["source"] == "random"
    assert chess.Move.from_uci(out["uci"]) in chess.Board().legal_moves


def test_illegal_selector_move_falls_back_to_random(monkeypatch):
    class Bad:
        def choose_move(self, board):
            return chess.Move.from_uci("e2e5")          # not legal from the start
    monkeypatch.setattr(opponent, "_selector", lambda: Bad())
    out = opponent.choose(chess.STARTING_FEN)
    assert out["ok"] and out["source"] == "random"
