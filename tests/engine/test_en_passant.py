from engine.research import ToolBackend
from engine.research.board import BoardState
from engine.research.engine import ChessEngine
from engine.research.notation import move_to_san


def test_double_pawn_push_sets_en_passant_target() -> None:
    engine = ChessEngine()

    result = engine.move("e2e4")

    assert result.ok
    assert engine.board.en_passant == "e3"


def test_en_passant_capture_removes_passed_pawn() -> None:
    board = BoardState.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    engine = ChessEngine(board)

    result = engine.move("e5d6")

    assert result.ok
    assert engine.board.piece_at("d6") == "P"
    assert engine.board.piece_at("d5") == "."
    assert engine.board.en_passant == "-"


def test_en_passant_expires_after_one_move() -> None:
    board = BoardState.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    engine = ChessEngine(board)

    assert "e5d6" in engine.legal_moves()
    engine.move("e1d1")

    assert engine.board.en_passant == "-"
    assert "d5e4" not in engine.legal_moves()


def test_en_passant_cannot_expose_king() -> None:
    board = BoardState.from_fen("4k3/8/8/r2pPK2/8/8/8/8 w - d6 0 1")

    assert "e5d6" not in ChessEngine(board).legal_moves()


def test_en_passant_formats_and_parses_san() -> None:
    board = BoardState.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    backend = ToolBackend(ChessEngine(board))

    assert move_to_san(board, "e5d6") == "exd6"
    assert backend.execute("<tool>move san=exd6</tool>") == "success: exd6"
