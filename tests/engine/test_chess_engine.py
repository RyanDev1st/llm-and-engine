from engine.research import ChessEngine
from engine.research.board import BoardState


def test_start_position_has_20_legal_moves() -> None:
    engine = ChessEngine()

    moves = engine.legal_moves()

    assert len(moves) == 20
    assert "e2e4" in moves
    assert "g1f3" in moves


def test_move_and_undo_restore_position() -> None:
    engine = ChessEngine()
    start = engine.board.to_fen()

    result = engine.move("e2e4")
    undo = engine.undo()

    assert result.ok
    assert result.message == "played: e2e4"
    assert undo.ok
    assert engine.board.to_fen() == start


def test_illegal_move_keeps_position() -> None:
    engine = ChessEngine()
    start = engine.board.to_fen()

    result = engine.move("e2e5")

    assert not result.ok
    assert result.message == "illegal move: e2e5"
    assert engine.board.to_fen() == start


def test_list_pieces_reports_squares() -> None:
    engine = ChessEngine()

    pieces = engine.list_pieces()

    assert "K@e1" in pieces
    assert "k@e8" in pieces
    assert "P@e2" in pieces


def test_material_evaluation_is_white_minus_black() -> None:
    board = BoardState.from_fen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    engine = ChessEngine(board)

    assert engine.evaluate_material() == 900


def test_load_fen_resets_history() -> None:
    engine = ChessEngine()
    engine.move("e2e4")

    engine.load_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    result = engine.undo()

    assert not result.ok
    assert result.message == "error: no move to undo"


def test_moves_cannot_capture_opponent_king() -> None:
    board = BoardState.from_fen("5k2/8/8/8/8/8/8/4KQ2 w - - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "f1f8" not in moves
    assert "f1a6" in moves


def test_king_cannot_capture_adjacent_king() -> None:
    board = BoardState.from_fen("8/8/8/8/8/8/4k3/4K3 w - - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e1e2" not in moves


def test_search_finds_mate_in_one() -> None:
    board = BoardState.from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    engine = ChessEngine(board)

    assert engine.move("f7g7").ok


def test_search_sees_stalemate_after_move() -> None:
    board = BoardState.from_fen("7k/5Q2/7K/8/8/8/8/8 w - - 0 1")
    engine = ChessEngine(board)

    assert engine.move("f7g7").ok
    assert engine.legal_moves() == []


def test_king_cannot_move_into_check() -> None:
    board = BoardState.from_fen("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e1e2" not in moves
    assert "e1d1" in moves


def test_pinned_piece_cannot_expose_king() -> None:
    board = BoardState.from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e2d2" not in moves
    assert "e2e8" in moves


def test_castling_blocked_while_in_check() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/4r3/R3K2R w KQk - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e1g1" not in moves
    assert "e1c1" not in moves


def test_castling_blocked_through_attacked_square() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/2b5/8/8/R3K2R w KQk - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e1g1" not in moves
    assert "e1c1" in moves


def test_black_castling_blocked_through_attacked_square() -> None:
    board = BoardState.from_fen("r3k2r/7B/8/8/8/8/8/R3K3 b kq - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e8g8" not in moves
    assert "e8c8" in moves


def test_castling_moves_are_generated_when_rights_exist() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")

    moves = ChessEngine(board).legal_moves()

    assert "e1g1" in moves
    assert "e1c1" in moves


def test_castling_move_repositions_rook_and_king() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")
    engine = ChessEngine(board)

    result = engine.move("e1g1")

    assert result.ok
    assert engine.board.piece_at("g1") == "K"
    assert engine.board.piece_at("f1") == "R"
    assert engine.board.piece_at("e1") == "."
    assert engine.board.piece_at("h1") == "."
    assert engine.board.castling == "k"


def test_rook_move_updates_castling_rights() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")
    engine = ChessEngine(board)

    result = engine.move("h1h2")

    assert result.ok
    assert engine.board.castling == "Qk"
