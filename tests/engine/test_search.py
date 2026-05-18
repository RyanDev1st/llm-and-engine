from engine.research import ChessEngine
from engine.research.board import BoardState
from engine.research.search import _move_priority


def test_move_priority_prefers_checking_moves() -> None:
    board = BoardState.from_fen("4k3/8/8/8/8/8/8/2Q3K1 w - - 0 1")
    engine = ChessEngine(board)

    assert _move_priority(engine, "c1c8") > _move_priority(engine, "c1a1")
