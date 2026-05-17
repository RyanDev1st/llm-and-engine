from __future__ import annotations

from .attack import is_attacked, king_square
from .board import BoardState
from .engine import ChessEngine


def move_to_san(board: BoardState, uci: str) -> str:
    if uci in {"e1g1", "e8g8"}:
        return "O-O" + _check_suffix(board, uci)
    if uci in {"e1c1", "e8c8"}:
        return "O-O-O" + _check_suffix(board, uci)
    source, target = uci[:2], uci[2:4]
    piece = board.piece_at(source)
    capture = board.piece_at(target) != "."
    suffix = _check_suffix(board, uci)
    promote = f"={uci[4].upper()}" if len(uci) > 4 else ""
    if piece.upper() == "P":
        return (f"{source[0]}x{target}" if capture else target) + promote + suffix
    prefix = piece.upper()
    return (f"{prefix}x{target}" if capture else f"{prefix}{target}") + suffix


def san_line(engine: ChessEngine, moves: list[str]) -> list[str]:
    labels: list[str] = []
    clone = ChessEngine(engine.board)
    for move in moves:
        labels.append(move_to_san(clone.board, move))
        clone.move(move)
    return labels


def _check_suffix(board: BoardState, uci: str) -> str:
    next_board = ChessEngine(board).next_board(uci)
    king = king_square(next_board, next_board.turn)
    if not king or not is_attacked(next_board, king, _other(next_board.turn)):
        return ""
    return "#" if not ChessEngine(next_board).legal_moves() else "+"


def _other(turn: str) -> str:
    return "b" if turn == "w" else "w"
