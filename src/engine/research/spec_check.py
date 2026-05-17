from __future__ import annotations

from engine.research import ChessEngine, ToolBackend
from engine.research.board import BoardState


def main() -> None:
    checks = [
        start_position_count,
        accepts_legal_move,
        rejects_illegal_move,
        undo_restores_position,
        lists_piece_locations,
        evaluates_material,
        fen_load_resets_history,
        tool_search_mate_shape,
        tool_capture_preference_shape,
        tool_move_shape,
        tool_eval_shape,
        tool_utility_shape,
        tool_review_shape,
        tool_piece_san_shape,
        tool_capture_san_shape,
        tool_quoted_query_shape,
        tool_disambiguated_piece_san_shape,
        tool_castling_san_shape,
        tool_castling_list_shape,
        filters_self_check,
        blocks_castle_through_check,
    ]
    passed = sum(1 for check in checks if check())
    print(passed)


def start_position_count() -> bool:
    return len(ChessEngine().legal_moves()) == 20


def accepts_legal_move() -> bool:
    result = ChessEngine().move("e2e4")
    return result.ok and result.fen.split()[1] == "b"


def rejects_illegal_move() -> bool:
    engine = ChessEngine()
    before = engine.board.to_fen()
    result = engine.move("e2e5")
    return not result.ok and engine.board.to_fen() == before


def undo_restores_position() -> bool:
    engine = ChessEngine()
    before = engine.board.to_fen()
    engine.move("g1f3")
    return engine.undo().ok and engine.board.to_fen() == before


def lists_piece_locations() -> bool:
    pieces = ChessEngine().list_pieces()
    return "K@e1" in pieces and "k@e8" in pieces


def evaluates_material() -> bool:
    board = BoardState.from_fen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    return ChessEngine(board).evaluate_material() == 900


def fen_load_resets_history() -> bool:
    engine = ChessEngine()
    engine.move("e2e4")
    engine.load_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    return not engine.undo().ok


def tool_search_mate_shape() -> bool:
    board = BoardState.from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    backend = ToolBackend(ChessEngine(board))
    return backend.execute("<tool>eval depth=15</tool>") == "score: mate in 1 for white, depth=15"


def tool_capture_preference_shape() -> bool:
    board = BoardState.from_fen("6k1/8/8/8/3q4/8/3Q4/6K1 w - - 0 1")
    backend = ToolBackend(ChessEngine(board))
    return backend.execute("<tool>best_move depth=15</tool>") == "best: Qxd4"


def tool_move_shape() -> bool:
    backend = ToolBackend()
    return backend.execute("<tool>move san=e4</tool>") == "success: e4"


def tool_eval_shape() -> bool:
    backend = ToolBackend()
    return backend.execute("<tool>eval depth=15</tool>") == "score: +0.00 pawns from white POV, depth=15"


def tool_utility_shape() -> bool:
    backend = ToolBackend()
    legal = backend.execute("<tool>legal_moves square=e2</tool>")
    pieces = backend.execute("<tool>list_pieces color=white</tool>")
    return legal == "legal: [e3, e4]" and "K=e1" in pieces


def tool_review_shape() -> bool:
    backend = ToolBackend()
    backend.execute("<tool>move san=e4</tool>")
    return backend.execute("<tool>review_move</tool>").startswith("review: e4, label=good")


def tool_piece_san_shape() -> bool:
    backend = ToolBackend()
    return backend.execute("<tool>move san=Nf3</tool>") == "success: Nf3"


def tool_capture_san_shape() -> bool:
    backend = ToolBackend()
    backend.execute("<tool>move san=e4</tool>")
    backend.execute("<tool>move san=d5</tool>")
    return backend.execute("<tool>move san=exd5</tool>") == "success: exd5"


def tool_quoted_query_shape() -> bool:
    backend = ToolBackend()
    response = backend.execute('<tool>ask_chessbot query="Sicilian defense ideas"</tool>')
    return "Sicilian Defense" in response


def tool_disambiguated_piece_san_shape() -> bool:
    board = BoardState.from_fen("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1")
    backend = ToolBackend(ChessEngine(board))
    return backend.execute("<tool>move san=Nbd2</tool>") == "success: Nbd2"


def tool_castling_san_shape() -> bool:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")
    backend = ToolBackend(ChessEngine(board))
    return backend.execute("<tool>move san=O-O</tool>") == "success: O-O"


def tool_castling_list_shape() -> bool:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")
    backend = ToolBackend(ChessEngine(board))
    return "O-O" in backend.execute("<tool>legal_moves square=e1</tool>")


def filters_self_check() -> bool:
    board = BoardState.from_fen("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")
    return "e2d2" not in ChessEngine(board).legal_moves()


def blocks_castle_through_check() -> bool:
    board = BoardState.from_fen("4k2r/8/8/8/2b5/8/8/R3K2R w KQk - 0 1")
    return "e1g1" not in ChessEngine(board).legal_moves()


if __name__ == "__main__":
    main()
