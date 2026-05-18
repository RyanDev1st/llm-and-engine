from __future__ import annotations

from .attack import is_attacked, king_square
from .engine import ChessEngine


def game_over(engine: ChessEngine) -> str:
    if insufficient_material(engine) or engine.board.halfmove >= 100 or repeated_position(engine):
        return ", game_over=draw"
    if engine.legal_moves():
        return ""
    king = king_square(engine.board, engine.board.turn)
    if king and is_attacked(engine.board, king, "b" if engine.board.turn == "w" else "w"):
        return ", game_over=checkmate"
    return ", game_over=stalemate"


def insufficient_material(engine: ChessEngine) -> bool:
    pieces = engine.list_pieces()
    kinds = [item.split("@")[0].upper() for item in pieces]
    if any(piece not in {"K", "B", "N"} for piece in kinds):
        return False
    minors = [(piece, square) for piece, square in (item.split("@") for item in pieces) if piece.upper() in {"B", "N"}]
    if len(minors) <= 1:
        return True
    return len(minors) == 2 and all(piece.upper() == "B" for piece, _ in minors) and same_color(minors[0][1], minors[1][1])


def same_color(a: str, b: str) -> bool:
    return (ord(a[0]) + int(a[1])) % 2 == (ord(b[0]) + int(b[1])) % 2


def repeated_position(engine: ChessEngine) -> bool:
    current = position_key(engine.board)
    count = 1
    for board, uci in reversed(engine._history):
        if irreversible(board, uci):
            break
        if position_key(board) == current:
            count += 1
    return count >= 3


def irreversible(board: object, uci: str) -> bool:
    piece = board.piece_at(uci[:2])
    return piece.upper() == "P" or board.piece_at(uci[2:4]) != "."


def position_key(board: object) -> str:
    return " ".join(board.to_fen().split()[:4])
