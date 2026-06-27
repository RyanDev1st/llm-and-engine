"""Grounded move review (slice F): turn the move just played + the engine's read into
an HONEST verdict — was it good or bad, by how much, and WHY (its own merit, or the
better move's point).

Replaces the old slice-F final, which hardcoded `label=good, delta=+0.05 pawns` for
EVERY move choose_move picked — a rubber-stamp AND a fabricated number. That taught
the model that "review" = always say good: exactly the "never learned to concretely
answer" failure. Here the label and the centipawn loss are MEASURED from two real
annotations (before the move, after it), and the praise/correction reuses the verified
move_facts, so a good move's praise and a bad move's fix both name only true points.

Pure functions (compute_review / label_for / why_review / delta_str) are engine-free
and unit-tested on hand-made positions; review_for_played does the one extra engine
call + terminal handling and is what the renderer calls."""
from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from ..annotator import AnnotatedPosition
from ..move_facts import move_facts
from .grounded import move_reason


def _mate_aware_cp(kind: str, cp: int) -> int:
    """One signed magnitude for comparison: a mate is a huge edge, sign by side (white POV)."""
    if kind != "mate":
        return cp
    return 10000 if cp > 0 else -10000


@dataclass(frozen=True)
class ReviewFacts:
    played: str
    best: str
    loss_cp: int            # mover-POV centipawns the played move conceded vs best (>=0)
    label: str              # best | good | inaccuracy | mistake | blunder
    mover_best_cp: int      # mover-POV eval if best had been played
    mover_after_cp: int     # mover-POV eval after the played move
    note: str = ""          # mate-swing note ("allows mate" / "missed a forced mate") or ""


def label_for(loss_cp: int, is_best: bool) -> str:
    """Centipawn-loss bins (mover POV). Thresholds are deliberately conservative so the
    label never overstates: a blunder is a clear >=2.5-pawn drop, not a rounding wobble."""
    if is_best or loss_cp <= 10:
        return "best" if is_best else "good"
    if loss_cp <= 40:
        return "good"
    if loss_cp <= 99:
        return "inaccuracy"
    if loss_cp <= 249:
        return "mistake"
    return "blunder"


def compute_review(before: AnnotatedPosition, after: AnnotatedPosition, played: str) -> ReviewFacts:
    """Measure `played` against the engine's best, both from the MOVER's POV. `before` is
    the position the move was played in; `after` is the resulting position (opponent to
    move). A mate that flips is flagged as a note — a centipawn 'loss' isn't pawns then."""
    white = chess.Board(before.fen).turn == chess.WHITE
    best_w = _mate_aware_cp(before.score_kind, before.score_cp)
    after_w = _mate_aware_cp(after.score_kind, after.score_cp)
    mover_best = best_w if white else -best_w
    mover_after = after_w if white else -after_w
    loss = max(0, mover_best - mover_after)
    is_best = played == before.best_san
    already_mated = before.score_kind == "mate" and mover_best < 0   # was already lost to mate
    allows_mate = after.score_kind == "mate" and mover_after < 0 and not already_mated
    missed_mate = (before.score_kind == "mate" and mover_best > 0
                   and not (after.score_kind == "mate" and mover_after > 0))
    note = "allows mate" if allows_mate else ("missed a forced mate" if missed_mate else "")
    label = "blunder" if note else label_for(loss, is_best)
    return ReviewFacts(played, before.best_san, loss, label, mover_best, mover_after, note)


def review_for_played(annotator, before: AnnotatedPosition, played: str, depth: int = 12) -> ReviewFacts:
    """compute_review wrapper that supplies `after` — one extra engine call, except on a
    terminal position (analysing a mated/stalemated board errors), where the result is
    synthesized: checkmate by the mover is a decisive win, stalemate/draw is level."""
    board = chess.Board(before.fen)
    mover_white = board.turn == chess.WHITE
    board.push(board.parse_san(played))
    if board.is_game_over():
        if board.is_checkmate():
            kind, cp = "mate", (1 if mover_white else -1)         # mover delivered mate
        else:
            kind, cp = "cp", 0                                    # stalemate / draw -> level
        after = AnnotatedPosition(board.fen(), depth, cp, kind, "", (), None, ())
    else:
        after = annotator.annotate(board.fen(), depth=depth)
    return compute_review(before, after, played)


def delta_str(rf: ReviewFacts) -> str:
    """The eval swing as it appears in the review tool result. A mate-swing has no honest
    pawn value, so it shows the note instead of a number."""
    if rf.note:
        return rf.note
    return f"{-rf.loss_cp / 100:+.2f} pawns"


_PRAISE_BEST = ("{m} was the best move here", "{m} is exactly right", "Nothing better than {m}")
_PRAISE_GOOD = ("{m} was a good move", "{m} holds up well", "{m} is a sound choice")


def why_review(rf: ReviewFacts, before: AnnotatedPosition, seed: int, ask_number: bool) -> str:
    """The user-facing verdict: praise a good move for its real point, or call a weak move
    what it is and name the stronger move's point. Reasons come from move_facts, so they
    can't invent a tactic; the cost number (when the user asked) equals delta_str's."""
    r = random.Random(seed * 167 + 5)
    if rf.label in ("best", "good"):
        mf = move_facts(before.fen, rf.played)
        reason = move_reason(mf, rf.mover_after_cp, seed) if mf else "it keeps the position sound"
        pool = _PRAISE_BEST if rf.label == "best" else _PRAISE_GOOD
        return f"{r.choice(pool).replace('{m}', rf.played)} — {reason}."
    if rf.note == "allows mate":
        return f"{rf.played} walks into mate — {rf.best} was essential."
    if rf.note == "missed a forced mate":
        return f"{rf.played} let a forced mate slip — {rf.best} finished it."
    bmf = move_facts(before.fen, rf.best)
    breason = move_reason(bmf, rf.mover_best_cp, seed) if bmf else "it holds more"
    verdict = {"inaccuracy": "a slight inaccuracy", "mistake": "a mistake", "blunder": "a blunder"}[rf.label]
    cost = f" It cost about {rf.loss_cp / 100:.2f} pawns." if ask_number else ""
    return r.choice([
        f"{rf.played} was {verdict} — {rf.best} was stronger, since {breason}.{cost}",
        f"{rf.played} isn't best; {rf.best} was the move — {breason}.{cost}",
        f"That's {verdict}. {rf.best} keeps more in hand: {breason}.{cost}",
    ])
