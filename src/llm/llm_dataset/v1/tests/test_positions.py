import chess
from llm_dataset.v1.positions import PositionBank, load_default_bank, sample_position


def test_default_bank_returns_legal_positions():
    bank = load_default_bank()
    assert isinstance(bank, PositionBank)
    assert bank.count("opening") >= 200
    for category in ("opening", "middlegame", "endgame", "tactics", "terminal"):
        sample = sample_position(bank, category, seed=42)
        chess.Board(sample.fen)  # must parse


def test_sample_position_deterministic_with_seed():
    bank = load_default_bank()
    a = sample_position(bank, "opening", seed=1)
    b = sample_position(bank, "opening", seed=1)
    assert a.fen == b.fen
