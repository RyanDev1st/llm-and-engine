"""Grounded 'why' composer: turn a recommended move's verified facts (move_facts) +
the engine's eval/line into a concrete reason the move is good — the piece v1-v4
never generated. Their best-move final was 'Bb5 is the move; the line runs ...':
move + line, no WHY (the 80%-no-why gap that fed serve-time confabulation).

Every tactical claim here comes from a TRUE fact (move_facts), so the final can't
invent a threat; when the move has no flashy point, the reason stays honestly
positional (not every best move forks something). Phrasing varies by seed to dodge
the final-repetition trap. The SAN/number the final cites are the move, the line,
and the score — all present in the row's tool result, so numeric/SAN grounding holds."""
from __future__ import annotations

import random

import chess

from ..annotator import AnnotatedPosition
from ..move_facts import MoveFacts, move_facts
from .text import eval_magnitude, score_phrase

# Honest defensive reasons — used when the mover is losing even after best play, so we
# never dress a damage-limiting move up as a winning idea.
_DEFENCE = ("it's the most resilient defence here", "it puts up the stiffest resistance",
            "it's the best practical try to hang on", "it keeps the game alive")


def move_reason(mf: MoveFacts, mover_cp: int, seed: int) -> str:
    """A grounded reason clause (no move name / number) for why the move is good, from
    the MOVER's POV centipawns. Material/fork claims are gated on the eval actually
    showing that edge — move_facts' 1-ply heuristic can't see a deep recapture, so a
    'wins the knight' that the engine values at +0.5 would be a lie. Check/pin/develop
    are structural truths (kept ungated); when the mover is lost, the reason is honestly
    defensive, not a fabricated point."""
    r = random.Random(seed * 131 + 7)
    if mf.is_mate:
        return r.choice(["it forces mate", "it's checkmate", "it finishes the game on the spot"])
    if mover_cp <= -150 and not mf.gives_check:        # lost: it's defence, not a winning idea
        return r.choice(_DEFENCE)
    if mf.forks and mover_cp >= 80:
        n1, n2 = mf.fork_names[0], mf.fork_names[1]
        return r.choice([f"it hits both the {n1} and the {n2} at once",
                         f"it forks the {n1} and the {n2}",
                         f"it attacks the {n1} and the {n2} in a single stroke"])
    if mf.wins_material and mf.captured and mover_cp >= 120:
        return r.choice([f"it wins the {mf.captured}", f"it picks off the {mf.captured}",
                         f"it nets the {mf.captured}"])
    if mf.pin_to_king and mf.pin_name:
        return r.choice([f"it pins the {mf.pin_name} to the king",
                         f"it pins the {mf.pin_name}, freezing it in place",
                         f"it nails the {mf.pin_name} against the king"])
    if mf.attacks_queen and mover_cp >= 40:
        return r.choice(["it hits the queen and gains a tempo",
                         "it harasses the queen, winning time"])
    if mf.gives_check:
        return r.choice(["it comes with check and seizes the initiative",
                         "it checks the king and forces the reply"])
    if mf.is_castling:
        return r.choice(["it tucks the king to safety and connects the rooks",
                         "it gets the king safe and the rook into the game"])
    if mf.develops_minor and mover_cp >= -40:
        return r.choice([f"it develops the {mf.piece} toward the centre",
                         f"it brings the {mf.piece} into the game with tempo",
                         f"it develops and fights for the centre"])
    # No concrete tactic — keep the reason honest and positional, scaled to the eval.
    if mover_cp >= 150:
        return r.choice(["it keeps you firmly in control", "it presses the advantage cleanly",
                         "it keeps the position on your terms"])
    if mover_cp <= -60:
        return r.choice(_DEFENCE)
    return r.choice(["it's the soundest continuation here", "it keeps the position healthiest",
                     "it's the most accurate try, keeping the balance"])


def threat_reason(fen: str, threat_san: str, seed: int) -> str | None:
    """A concrete, THREAT-FRAMED reason naming what the opponent's threat would do, from
    move_facts on the position after a null move (opponent on the move). The annotator's
    threats_san is the engine's top threat, so describing its real effect is grounded;
    the conditional framing ('would', 'threatens') keeps it honest even if the threat is
    defendable. Returns None when the threat has no flashy point — the caller falls back
    to a generic 'serious initiative' line rather than inventing one."""
    try:
        b = chess.Board(fen)
        b.push(chess.Move.null())                 # hand the move to the opponent
    except (ValueError, AssertionError):
        return None
    mf = move_facts(b.fen(), threat_san)
    if not mf:
        return None
    r = random.Random(seed * 191 + 13)
    if mf.is_mate:
        return r.choice(["it threatens mate", "mate is the threat"])
    if mf.forks:
        n1, n2 = mf.fork_names[0], mf.fork_names[1]
        return r.choice([f"it would hit both your {n1} and {n2}", f"it forks your {n1} and {n2}"])
    if mf.wins_material and mf.captured:
        return r.choice([f"it would win your {mf.captured}", f"it picks off your {mf.captured}"])
    if mf.pin_to_king and mf.pin_name:
        return f"it pins your {mf.pin_name} to the king"
    if mf.gives_check:
        return r.choice(["it comes in with check", "it hits you with check"])
    if mf.attacks_queen:
        return "it goes after your queen"
    return None


def _short_line(annotated: AnnotatedPosition, n: int = 4) -> str:
    return " ".join(annotated.best_line_sans[:n])


def why_best_move(annotated: AnnotatedPosition, ask_number: bool, seed: int, top_form: bool = False) -> str:
    """A full best-move recommendation final: the move + a grounded reason + the
    evidence (line, or the alternatives in top=N form) + the standing. top_form must
    NOT cite the deep line — that row's tool result is the best_moves list, not the PV,
    so citing the line would be ungrounded SAN."""
    move = annotated.best_san
    mf = move_facts(annotated.fen, move)
    # mover-POV centipawns (the reason's eval gates are from the side to move). A mate
    # score is already a forcing fact, so map it to a large signed magnitude.
    white_to_move = chess.Board(annotated.fen).turn == chess.WHITE
    raw = 10000 if annotated.score_kind == "mate" and annotated.score_cp > 0 else (
        -10000 if annotated.score_kind == "mate" else annotated.score_cp)
    mover_cp = raw if white_to_move else -raw
    reason = move_reason(mf, mover_cp, seed) if mf else "it's the engine's top choice"
    r = random.Random(seed * 149 + 3)
    cap = reason[0].upper() + reason[1:]
    if top_form and len(annotated.top_moves) >= 2:
        backups = ", ".join(san for san, _ in annotated.top_moves[1:3]) or "the alternatives"
        tail = f" {backups} are the other tries." if not ask_number else (
            f" Backups: {backups}.")
        evid = f"{move} looks best — {reason}.{tail}"
    else:
        line = _short_line(annotated)
        evid = r.choice([
            f"{move} is the move — {reason}. The line runs {line}.",
            f"I'd play {move}: {reason}. Play continues {line}.",
            f"{move} here — {cap}. The line runs {line}.",
        ])
    standing = score_phrase(annotated) if ask_number else eval_magnitude(annotated, seed)
    return f"{evid} {standing}"
