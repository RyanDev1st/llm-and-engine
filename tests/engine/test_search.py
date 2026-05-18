from engine.research import ChessEngine
from engine.research.board import BoardState
from engine.research.search import _move_priority


def test_move_priority_prefers_castling_over_quiet_king_move() -> None:
    board = BoardState.from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    engine = ChessEngine(board)

    assert _move_priority(engine, "e1g1") > _move_priority(engine, "e1f1")


def test_move_priority_prefers_queen_promotion() -> None:
    board = BoardState.from_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
    engine = ChessEngine(board)

    assert _move_priority(engine, "e7e8q") > _move_priority(engine, "e7e8n")


def test_move_priority_prefers_mate_in_one() -> None:
    board = BoardState.from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    engine = ChessEngine(board)

    assert _move_priority(engine, "f7g7") > _move_priority(engine, "f7h7")


def test_move_priority_prefers_checking_moves() -> None:
    board = BoardState.from_fen("4k3/8/8/8/8/8/8/2Q3K1 w - - 0 1")
    engine = ChessEngine(board)

    assert _move_priority(engine, "c1c8") > _move_priority(engine, "c1a1")
