"""Text formatting helpers for score and eval language."""
from __future__ import annotations

from ..annotator import AnnotatedPosition


def score_text(annotated: AnnotatedPosition) -> str:
    if annotated.score_kind == "mate":
        side = "white" if annotated.score_cp > 0 else "black"
        return f"score: mate in {abs(annotated.score_cp)} for {side}, depth={annotated.depth}"
    return f"score: {score_pawns(annotated)}, depth={annotated.depth}"


def score_pawns(annotated: AnnotatedPosition) -> str:
    return f"{annotated.score_cp / 100:+.2f} pawns from white POV"


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
