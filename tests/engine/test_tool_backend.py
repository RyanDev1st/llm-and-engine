from engine.research import ToolBackend, parse_tool_call
from engine.research.board import BoardState
from engine.research.engine import ChessEngine


def test_parse_tool_call_args() -> None:
    assert parse_tool_call("<tool>best_move depth=15 series=3</tool>") == (
        "best_move",
        {"depth": "15", "series": "3"},
    )


def test_parse_tool_call_quoted_args() -> None:
    assert parse_tool_call('<tool>ask_chessbot query="Sicilian defense ideas"</tool>') == (
        "ask_chessbot",
        {"query": "Sicilian defense ideas"},
    )


def test_backend_move_and_undo_shapes() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>move san=e4</tool>") == "success: e4"
    assert backend.execute("<tool>undo</tool>") == "success: undid e4"


def test_backend_accepts_piece_san() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>move san=Nf3</tool>") == "success: Nf3"


def test_backend_accepts_castling_san() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")
    backend = ToolBackend(ChessEngine(board))

    assert backend.execute("<tool>move san=O-O</tool>") == "success: O-O"


def test_backend_formats_castling_in_move_lists() -> None:
    board = BoardState.from_fen("4k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1")
    backend = ToolBackend(ChessEngine(board))

    legal = backend.execute("<tool>legal_moves square=e1</tool>")

    assert "O-O" in legal
    assert "O-O-O" in legal


def test_backend_accepts_disambiguated_piece_san() -> None:
    board = BoardState.from_fen("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1")
    backend = ToolBackend(ChessEngine(board))

    assert backend.execute("<tool>move san=Nbd2</tool>") == "success: Nbd2"


def test_backend_accepts_capture_san() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>move san=e4</tool>") == "success: e4"
    assert backend.execute("<tool>move san=d5</tool>") == "success: d5"
    assert backend.execute("<tool>move san=exd5</tool>") == "success: exd5"


def test_backend_rejects_illegal_move() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>move san=e5</tool>") == "error: illegal, reason=illegal move"


def test_backend_eval_reports_mate() -> None:
    board = BoardState.from_fen("6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    backend = ToolBackend(ChessEngine(board))

    assert backend.execute("<tool>eval depth=15</tool>") == "score: mate for white, requested_depth=15, searched_plies=3"
    assert backend.execute("<tool>best_move depth=15</tool>").startswith("best: Q")


def test_backend_best_move_prefers_big_capture() -> None:
    board = BoardState.from_fen("6k1/8/8/8/3q4/8/3Q4/6K1 w - - 0 1")
    backend = ToolBackend(ChessEngine(board))

    assert backend.execute("<tool>best_move depth=15</tool>") == "best: Qxd4, requested_depth=15, searched_plies=3"


def test_backend_eval_and_best_move_shapes() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>eval depth=15</tool>") == "score: +0.00 pawns from white POV, requested_depth=15, searched_plies=3"
    assert backend.execute("<tool>best_move depth=15</tool>").startswith("best: ")
    assert backend.execute("<tool>best_move depth=15 series=3</tool>").startswith("best_line: ")


def test_backend_utility_shapes() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>legal_moves square=e2</tool>") == "legal: [e3, e4]"
    assert "K=e1" in backend.execute("<tool>list_pieces color=white</tool>")


def test_backend_review_threats_and_chessbot() -> None:
    backend = ToolBackend()

    assert backend.execute("<tool>review_move</tool>") == "error: no moves to review"
    backend.execute("<tool>move san=e4</tool>")
    assert backend.execute("<tool>review_move</tool>").startswith("review: e4, label=good")
    assert backend.execute("<tool>threats depth=12</tool>").startswith("threats: opponent's best is ")
    assert "Sicilian Defense" in backend.execute("<tool>ask_chessbot query=Sicilian</tool>")
    assert "Sicilian Defense" in backend.execute('<tool>ask_chessbot query="Sicilian defense ideas"</tool>')
