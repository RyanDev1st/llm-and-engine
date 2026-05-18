from engine.research.board import BoardState
from engine.research.evaluation import pawn_structure, piece_activity


def test_pawn_structure_rewards_passed_pawns() -> None:
    board = BoardState.from_fen("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")

    assert pawn_structure(board) > 0


def test_pawn_structure_penalizes_doubled_isolated_pawns() -> None:
    board = BoardState.from_fen("4k3/8/8/8/8/4P3/4P3/4K3 w - - 0 1")

    assert pawn_structure(board) < 0


def test_pawn_structure_blocks_passed_pawn_bonus() -> None:
    board = BoardState.from_fen("4k3/8/4p3/4P3/8/8/8/4K3 w - - 0 1")

    assert pawn_structure(board) < 20


def test_piece_activity_rewards_center_minors() -> None:
    edge = BoardState.from_fen("4k3/8/8/8/N7/8/8/4K3 w - - 0 1")
    center = BoardState.from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")

    assert piece_activity(center) > piece_activity(edge)
