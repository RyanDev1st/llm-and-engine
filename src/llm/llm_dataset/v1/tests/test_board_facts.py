import chess

from llm_dataset.v1.board_facts import (
    board_state_line,
    choose_move,
    legal_sans,
    move_echo,
)

WHITE_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# black to move (e4 is illegal for black here)
BLACK_TO_MOVE = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 2 2"


def test_board_state_reports_real_turn():
    assert "turn=white" in board_state_line(WHITE_START)
    assert "turn=black" in board_state_line(BLACK_TO_MOVE)


def test_board_state_basic_omits_fen_and_uses_backend_fields():
    line = board_state_line(WHITE_START)
    assert "fen=" not in line                       # real backend 'basic' omits fen
    for field in ("turn=", "last_move=", "check=", "legal_count="):
        assert field in line


def test_legal_sans_nonempty_and_correct():
    sans = legal_sans(WHITE_START)
    assert "e4" in sans and "Nf3" in sans


def test_choose_move_is_always_legal():
    for fen in (WHITE_START, BLACK_TO_MOVE):
        san = choose_move(fen, seed=7)
        chess.Board(fen).parse_san(san)  # raises if illegal


def test_choose_move_honors_legal_request():
    assert choose_move(WHITE_START, seed=1, requested="e4") == "e4"


def test_choose_move_rejects_illegal_request_and_falls_back_legal():
    out = choose_move(BLACK_TO_MOVE, seed=1, requested="e4")  # e4 illegal for black
    assert out != "e4"
    chess.Board(BLACK_TO_MOVE).parse_san(out)


def test_move_echo_matches_backend_success_string():
    assert move_echo(WHITE_START, "e4") == "success: e4"


def test_move_echo_returns_illegal_error_for_illegal_move():
    echo = move_echo(BLACK_TO_MOVE, "e4")
    assert echo.startswith("error: illegal")
