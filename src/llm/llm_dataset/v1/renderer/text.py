"""Text formatting helpers for score and eval language.

Two registers for the user-facing reply:
- `eval_magnitude` / `pawns_abs` qualitative-by-default standing (no raw number)
- `score_phrase` / `score_pawns` exact pawn score, only when the user asks.
Tool-result strings (`score_text`) stay numeric — they live in the thinking
turns, not the chat reply."""
from __future__ import annotations

import random

from ..annotator import AnnotatedPosition

# Magnitude buckets keyed off |centipawns|; a few phrasings each so the model
# learns the score->standing mapping, not one memorized sentence. {s} = side.
_MAG = {
    "equal": ("It's dead level — neither side has anything real yet.",
              "Roughly balanced here; no one's pulled ahead.",
              "About equal — this is anyone's game."),
    "slight": ("{s} is nudging slightly ahead — nothing decisive.",
               "{s} is a touch better, but it's close.",
               "{s} holds a small edge, not much in it."),
    "clear": ("{s} is clearly better here and should keep pressing.",
              "{s} is comfortably on top.",
              "{s} has the better of it by a clear margin."),
    "big": ("{s} is holding a big advantage — close to winning.",
            "{s} is well on top here, a commanding position.",
            "{s} has a large, near-decisive edge."),
    "winning": ("{s} is completely winning — this is all but decided.",
                "{s} is crushing here.",
                "{s} has a winning position, barring a slip."),
}


def _bucket(cp: int) -> str:
    a = abs(cp)
    if a < 40:
        return "equal"
    if a < 90:
        return "slight"
    if a < 200:
        return "clear"
    if a < 500:
        return "big"
    return "winning"


def eval_magnitude(annotated: AnnotatedPosition, seed: int = 0) -> str:
    """Qualitative standing — the default chat reply. Grounded by construction:
    a pure function of the real score, so it can't drift from the eval."""
    if annotated.score_kind == "mate":
        side = "White" if annotated.score_cp > 0 else "Black"
        return f"{side} has a forced mate in {abs(annotated.score_cp)}."
    cp = annotated.score_cp
    side = "White" if cp > 0 else "Black"
    return random.Random(seed * 13 + 7).choice(_MAG[_bucket(cp)]).replace("{s}", side)


def pawns_abs(annotated: AnnotatedPosition) -> str:
    """Two-decimal magnitude, e.g. '4.47 pawns' — keeps the fact regex groundable."""
    return f"{abs(annotated.score_cp) / 100:.2f} pawns"


def score_phrase(annotated: AnnotatedPosition) -> str:
    """Exact eval as a sentence — used only when the user asks for a number."""
    if annotated.score_kind == "mate":
        side = "White" if annotated.score_cp > 0 else "Black"
        return f"It's a forced mate in {abs(annotated.score_cp)} for {side}."
    side = "White" if annotated.score_cp > 0 else "Black"
    return f"{side} is ahead by {pawns_abs(annotated)}."


def score_text(annotated: AnnotatedPosition) -> str:
    if annotated.score_kind == "mate":
        side = "white" if annotated.score_cp > 0 else "black"
        return f"score: mate in {abs(annotated.score_cp)} for {side}, depth={annotated.depth}"
    return f"score: {score_pawns(annotated)}, depth={annotated.depth}"


def score_pawns(annotated: AnnotatedPosition) -> str:
    return f"{annotated.score_cp / 100:+.2f} pawns from white POV"


def best_move_score(annotated: AnnotatedPosition) -> str:
    """Score field for a best_move/threats tool RESULT — mirrors the live tool's
    `fmt_white_score(...).removeprefix('score: ').split(', depth')[0]` (backend/toolfmt.py),
    so a mate position reads 'mate in N for side' instead of a bogus pawn number
    (score_cp is the mate distance, not centipawns). Identical to score_pawns for
    non-mate, so it only changes mate rows — and keeps the 'mate in N' final grounded
    AND byte-matched to what the served engine returns."""
    if annotated.score_kind == "mate":
        side = "white" if annotated.score_cp > 0 else "black"
        return f"mate in {abs(annotated.score_cp)} for {side}"
    return score_pawns(annotated)


def eval_language(annotated: AnnotatedPosition) -> str:
    if annotated.score_kind == "mate":
        side = "white" if annotated.score_cp > 0 else "black"
        return f"{side.title()} has a forced mate in {abs(annotated.score_cp)}."
    cp = annotated.score_cp
    if abs(cp) < 50:
        return "Roughly equal — neither side has a meaningful edge yet."
    if abs(cp) < 150:
        return f"{'White' if cp > 0 else 'Black'} stands a little better, nothing decisive."
    return f"{'White' if cp > 0 else 'Black'} is clearly better and should keep pressing."
