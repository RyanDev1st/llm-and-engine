from __future__ import annotations

from .engine import ChessEngine
from .notation import san_line
from .search import MATE, search_position

MATE_CUTOFF = 99900


def review_last_move(engine: ChessEngine, san: str, uci: str, depth: int) -> str:
    if not engine.undo().ok:
        return "error: no moves to review"
    try:
        best = search_position(engine, depth)
        best_san = san_line(engine, best.pv[:1])[0] if best.pv else san
    finally:
        engine.move(uci)
    actual = search_position(engine, depth)
    best_score = best.score
    actual_score = -actual.score
    swing = actual_score - best_score
    return f"review: {san}, label={label(swing)}, delta={delta_text(swing, best_score, actual_score)}, best_was={best_san}"


def label(swing: int) -> str:
    if swing <= -150:
        return "blunder"
    if swing <= -50:
        return "mistake"
    if swing < -10:
        return "inaccuracy"
    return "good"


def delta_text(swing: int, best: int, actual: int) -> str:
    if abs(best) >= MATE_CUTOFF or abs(actual) >= MATE_CUTOFF:
        return mate_delta(best, actual) if swing else "+0.00 pawns"
    return f"{swing / 100:+.2f} pawns"


def mate_delta(best: int, actual: int) -> str:
    if abs(best) < MATE_CUTOFF or abs(actual) < MATE_CUTOFF:
        return "mate swing"
    return f"mate in {MATE - abs(actual)} vs mate in {MATE - abs(best)}"
