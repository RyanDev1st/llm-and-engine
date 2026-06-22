import hashlib
from collections.abc import Iterable

import chess
import torch


WHITE_PAWN = 0
WHITE_KNIGHT = 1
WHITE_BISHOP = 2
WHITE_ROOK = 3
WHITE_QUEEN = 4
WHITE_KING = 5
BLACK_PAWN = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_ROOK = 9
BLACK_QUEEN = 10
BLACK_KING = 11
SIDE_TO_MOVE = 12
CASTLING_RIGHTS = 13
EN_PASSANT = 14
INPUT_PLANES = 15

_PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): WHITE_PAWN,
    (chess.WHITE, chess.KNIGHT): WHITE_KNIGHT,
    (chess.WHITE, chess.BISHOP): WHITE_BISHOP,
    (chess.WHITE, chess.ROOK): WHITE_ROOK,
    (chess.WHITE, chess.QUEEN): WHITE_QUEEN,
    (chess.WHITE, chess.KING): WHITE_KING,
    (chess.BLACK, chess.PAWN): BLACK_PAWN,
    (chess.BLACK, chess.KNIGHT): BLACK_KNIGHT,
    (chess.BLACK, chess.BISHOP): BLACK_BISHOP,
    (chess.BLACK, chess.ROOK): BLACK_ROOK,
    (chess.BLACK, chess.QUEEN): BLACK_QUEEN,
    (chess.BLACK, chess.KING): BLACK_KING,
}


def hash_chess_fen(fen: str) -> str:
    """Hash board identity while ignoring halfmove and fullmove counters."""
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"Expected full FEN with at least 4 fields: {fen!r}")
    identity = " ".join(fields[:4])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def board_to_tensor(
    board: chess.Board,
    *,
    flip_if_black: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    flip_vertical = flip_if_black and board.turn == chess.BLACK
    tensor = torch.zeros((INPUT_PLANES, 8, 8), dtype=dtype)

    for square, piece in board.piece_map().items():
        row, col = _square_to_row_col(square, flip_vertical)
        tensor[_PIECE_PLANES[(piece.color, piece.piece_type)], row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[SIDE_TO_MOVE].fill_(1.0)

    if board.castling_rights:
        tensor[CASTLING_RIGHTS].fill_(1.0)

    if board.ep_square is not None:
        row, col = _square_to_row_col(board.ep_square, flip_vertical)
        tensor[EN_PASSANT, row, col] = 1.0

    return tensor


def boards_to_tensor(
    boards: Iterable[chess.Board],
    *,
    flip_if_black: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensors = [board_to_tensor(board, flip_if_black=flip_if_black, dtype=dtype) for board in boards]
    return torch.stack(tensors)


def normalize_centipawn_label(
    board: chess.Board,
    white_cp: int | float,
    *,
    clip: int = 1500,
) -> int:
    side_to_move_cp = white_cp if board.turn == chess.WHITE else -white_cp
    return int(max(-clip, min(clip, side_to_move_cp)))


def _square_to_row_col(square: chess.Square, flip_vertical: bool) -> tuple[int, int]:
    rank = chess.square_rank(square)
    row = rank if flip_vertical else 7 - rank
    return row, chess.square_file(square)
