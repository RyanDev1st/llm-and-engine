from __future__ import annotations

from .board import BoardState
from .castle import CASTLES


def move_to_san(board: BoardState, uci: str) -> str:
    if uci in {"e1g1", "e8g8"}:
        return "O-O"
    if uci in {"e1c1", "e8c8"}:
        return "O-O-O"
    source, target = uci[:2], uci[2:4]
    piece = board.piece_at(source)
    capture = board.piece_at(target) != "."
    if piece.upper() == "P":
        return f"{source[0]}x{target}" if capture else target
    prefix = piece.upper()
    return f"{prefix}x{target}" if capture else f"{prefix}{target}"
