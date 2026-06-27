"""move_facts must extract only TRUE move-effect facts — the grounding the 'why' is
built from. Each case is a hand-checked position so a wrong fact fails loudly."""
import chess

from llm_dataset.v1.move_facts import move_facts

START = chess.STARTING_FEN


def test_royal_fork_with_check():
    # Nb5-c7+ : the knight checks the e8 king AND hits the a8 rook (a royal fork).
    f = move_facts("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "Nc7+")
    assert f and f.piece == "knight"
    assert f.gives_check is True
    assert "a8" in f.forks and "rook" in f.fork_names
    assert "fork" in f.tactic_terms() and "check" in f.tactic_terms()


def test_absolute_pin_to_king():
    # Bf1-b5 : the bishop pins the d7 knight to the e8 king (one piece between, king beyond).
    f = move_facts("4k3/3n4/8/8/8/8/8/4KB2 w - - 0 1", "Bb5")
    assert f and f.piece == "bishop"
    assert f.pin_to_king == "d7" and f.pin_name == "knight"
    assert f.gives_check is False                 # blocked by the pinned knight -> not check
    assert "pin" in f.tactic_terms()


def test_develops_a_minor_piece():
    f = move_facts(START, "Nf3")
    assert f and f.piece == "knight"
    assert f.develops_minor is True
    assert f.is_capture is False and f.gives_check is False
    assert "develop" in f.tactic_terms()


def test_castling_flags_king_safety():
    f = move_facts("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O")
    assert f and f.is_castling is True
    assert "castle" in f.tactic_terms()


def test_free_capture_wins_material():
    # Rd1xd5 grabs an undefended bishop (black king on a8 can't recapture).
    f = move_facts("k7/8/8/3b4/8/8/8/3RK3 w - - 0 1", "Rxd5")
    assert f and f.is_capture is True
    assert f.captured == "bishop" and f.capture_square == "d5"
    assert f.wins_material is True
    assert "wins" in f.tactic_terms()


def test_defended_equal_capture_is_not_material_gain():
    # Qd2xd7+ is recapturable by the king (equal trade) -> not a material win.
    f = move_facts("3k4/3q4/8/8/8/8/3Q4/3K4 w - - 0 1", "Qxd7+")
    assert f and f.is_capture is True and f.captured == "queen"
    assert f.wins_material is False


def test_back_rank_mate():
    f = move_facts("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8#")
    assert f and f.is_mate is True and f.gives_check is True
    assert "mate" in f.tactic_terms()


def test_promotion():
    f = move_facts("8/P6k/8/8/8/8/8/6K1 w - - 0 1", "a8=Q")
    assert f and f.promotes == "queen" and f.piece == "pawn"


def test_attacks_the_queen_is_a_tempo():
    f = move_facts("4k3/3q4/8/8/8/8/4R3/4K3 w - - 0 1", "Rd2")
    assert f and f.attacks_queen is True
    assert "queen" in f.tactic_terms()


def test_illegal_move_returns_none():
    assert move_facts(START, "Qh5") is None        # blocked by the e2 pawn
    assert move_facts(START, "Ke2") is None
