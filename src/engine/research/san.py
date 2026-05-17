from __future__ import annotations

from .engine import ChessEngine


def san_to_uci(san: str, engine: ChessEngine) -> str:
    cleaned = san.replace("+", "").replace("#", "")
    if cleaned == "O-O":
        return f"e{rank(engine)}g{rank(engine)}"
    if cleaned == "O-O-O":
        return f"e{rank(engine)}c{rank(engine)}"
    promotion = ""
    if "=" in cleaned:
        cleaned, promote = cleaned.split("=", 1)
        promotion = promote.lower()
    capture_from = cleaned[0] if "x" in cleaned and cleaned[0] in "abcdefgh" else None
    cleaned = cleaned.replace("x", "")
    if len(cleaned) == 4 and cleaned[0] in "abcdefgh":
        return cleaned + promotion
    if len(cleaned) == 2 and cleaned[0] in "abcdefgh":
        return resolve_san_target(cleaned, "P", engine, capture_from, promotion)
    if len(cleaned) == 3 and cleaned[0] in "abcdefgh" and cleaned[1] in "abcdefgh":
        return resolve_san_target(cleaned[-2:], "P", engine, cleaned[0], promotion)
    if cleaned[0] in "NBRQK" and len(cleaned) >= 3:
        return resolve_san_target(cleaned[-2:], cleaned[0], engine, cleaned[1:-2] or None, promotion)
    raise ValueError("unsupported san")


def rank(engine: ChessEngine) -> str:
    return "1" if engine.board.turn == "w" else "8"


def resolve_san_target(target: str, piece: str, engine: ChessEngine, source_hint: str | None, promotion: str = "") -> str:
    candidates = []
    for move in engine.legal_moves():
        if move[2:4] != target or not matches_source_hint(move[:2], source_hint) or move[4:] != promotion:
            continue
        board_piece = engine.board.piece_at(move[:2])
        if board_piece != "." and board_piece.upper() == piece:
            candidates.append(move)
    if len(candidates) != 1:
        raise ValueError("ambiguous san")
    return candidates[0]


def matches_source_hint(source: str, source_hint: str | None) -> bool:
    return source_hint is None or source.startswith(source_hint) or source.endswith(source_hint)
