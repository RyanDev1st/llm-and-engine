"""Play-vs-engine opponent: the neural selector picks the side-to-move's move, with a random
fallback so the board never stalls. Contract: choose() returns a legal move, or a clear not-ok
for a bad/finished position. The monkeypatched tests force the deterministic branches; the REAL
load test (test_real_checkpoint_loads_and_plays_neural) exercises the actual checkpoint path so a
wrong _CKPT can't ship a silently-random engine again — that bug (parents[1] vs parents[2]) shipped
because every test mocked _selector and none touched the real path."""
import chess
import pytest

from backend import opponent


def test_checkpoint_path_points_at_the_real_engine_weights():
    # Regression: _CKPT must resolve to src/chess_engine/weights (parents[2]), NOT src/llm/... .
    # A wrong path makes torch.load raise -> silent random fallback -> a "stupid" engine.
    assert opponent._CKPT.exists(), f"checkpoint not at {opponent._CKPT} — wrong parents[] index?"


def test_real_checkpoint_loads_and_plays_neural():
    # No mocking: load the actual NeuralMoveSelector and confirm the served move is 'neural', not
    # the random fallback. Skips only if torch is genuinely absent (then random fallback is correct).
    pytest.importorskip("torch")
    if not opponent.available():
        pytest.skip("neural engine unavailable in this env (torch/checkpoint) — fallback is expected")
    out = opponent.choose(chess.STARTING_FEN)
    assert out["ok"] and out["source"] == "neural"
    assert chess.Move.from_uci(out["uci"]) in chess.Board().legal_moves


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
