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
    pieces = [item.split("@")[0] for item in engine.list_pieces()]
    minor = [piece for piece in pieces if piece.upper() in {"B", "N"}]
    return all(piece.upper() in {"K", "B", "N"} for piece in pieces) and len(minor) <= 1


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
