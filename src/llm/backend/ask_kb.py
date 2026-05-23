"""Canned knowledge base for ask_chessbot (MVP, per spec section 9 - RAG is v2).

Keyword-matched one-paragraph answers with a friendly generic fallback."""
from __future__ import annotations

_KB = [
    (("knight", "worth"), "A knight is worth about 3 points, the same as a bishop in most positions."),
    (("bishop", "worth"), "A bishop is worth about 3 points and grows stronger in open positions."),
    (("rook", "worth"), "A rook is worth about 5 points, a major piece behind only the queen."),
    (("queen", "strong"), "The queen is the strongest piece, worth roughly 9 points - it moves like a rook and bishop combined."),
    (("queen", "worth"), "The queen is worth about 9 points, the most valuable piece on the board."),
    (("pawn", "worth"), "A pawn is the baseline unit, worth about 1 point."),
    (("fork",), "A fork is a tactic where one piece attacks two or more enemy pieces at once."),
    (("pin",), "A pin freezes a piece because moving it would expose a more valuable piece behind it."),
    (("skewer",), "A skewer attacks a valuable piece, forcing it to move and exposing a lesser piece behind it."),
    (("castl",), "Castling tucks your king to safety and activates a rook in a single move."),
    (("sicilian",), "The Sicilian Defence (1.e4 c5) is Black's most popular and combative answer to 1.e4."),
    (("french",), "The French Defence (1.e4 e6) is a solid, slightly cramped but resilient setup for Black."),
    (("center",), "Controlling the center gives your pieces maximum mobility and flexible attacking options."),
    (("passed pawn",), "A passed pawn has no enemy pawns able to stop it from promoting - a powerful endgame asset."),
    (("zugzwang",), "Zugzwang is a position where any move a player makes worsens their own position."),
    (("opening",), "Good opening play means developing pieces quickly, controlling the center, and getting your king safe."),
    (("capablanca",), "Jose Raul Capablanca was the third World Champion, famed for his clear, effortless endgame technique."),
    (("en passant",), "En passant lets a pawn capture an enemy pawn that just advanced two squares, as if it had moved one."),
]
_FALLBACK = "That's a great chess question! In general, focus on piece activity, king safety, and controlling the center."


def answer(query: str) -> str:
    q = query.lower()
    for keys, text in _KB:
        if all(k in q for k in keys):
            return text
    return _FALLBACK
