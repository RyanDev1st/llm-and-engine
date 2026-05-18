from engine.research import ChessEngine
from engine.research.board import BoardState
from engine.research.evaluation import bishop_mobility, bishop_pair, king_safety, knight_mobility, pawn_structure, piece_activity, rook_mobility, static_evaluation


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


def test_king_safety_rewards_active_bare_king() -> None:
    edge = BoardState.from_fen("4k3/8/8/8/8/8/8/K7 w - - 0 1")
    active = BoardState.from_fen("8/8/8/3k4/3K4/8/8/8 w - - 0 1")

    assert king_safety(active) > king_safety(edge)


def test_rook_mobility_rewards_open_files() -> None:
    blocked = BoardState.from_fen("4k3/8/8/8/8/8/R7/P3K3 w - - 0 1")
    open_file = BoardState.from_fen("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")

    assert rook_mobility(open_file) > rook_mobility(blocked)


def test_bishop_mobility_rewards_open_diagonals() -> None:
    blocked = BoardState.from_fen("4k3/8/8/2P1P3/3B4/2P1P3/8/4K3 w - - 0 1")
    open_diagonals = BoardState.from_fen("4k3/8/8/8/3B4/8/8/4K3 w - - 0 1")

    assert bishop_mobility(open_diagonals) > bishop_mobility(blocked)


def test_bishop_pair_rewards_two_bishops() -> None:
    one_bishop = BoardState.from_fen("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
    two_bishops = BoardState.from_fen("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")

    assert bishop_pair(two_bishops) > bishop_pair(one_bishop)


def test_knight_mobility_rewards_open_jumps() -> None:
    rim = BoardState.from_fen("4k3/8/8/8/N7/8/8/4K3 w - - 0 1")
    center = BoardState.from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")

    assert knight_mobility(center) > knight_mobility(rim)


def test_static_evaluation_includes_knight_mobility() -> None:
    rim = ChessEngine(BoardState.from_fen("4k3/8/8/8/N7/8/8/4K3 w - - 0 1"))
    center = ChessEngine(BoardState.from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"))

    assert static_evaluation(center) > static_evaluation(rim)
