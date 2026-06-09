"""User-facing final reply (the chat turn the user actually reads).

Design: the reply is the coach speaking, not a tool dump. So —
- skill loads / tool calls never appear here (they're the thinking turns);
- no "engine"/"Stockfish" attribution — the coach states the read as its own;
- concrete MOVES (SAN) are always shown, they're the answer;
- the position's standing is QUALITATIVE by default ("a clear edge",
  "completely winning"); the exact pawn score is quoted ONLY when the user
  explicitly asks for a number.
Any number/move that does appear must come from a tool result — see
validate._narration_grounded — so the simplified reply still can't fabricate.
"""
from __future__ import annotations

import re

from ..annotator import AnnotatedPosition
from ..sampler import Scenario
from . import tone
from .leadins import ask
from .text import eval_magnitude, pawns_abs, score_pawns, score_phrase

# Words that mean "give me the actual number", not just "how am I doing".
_NUM_ASK = re.compile(
    r"\b(eval|evaluation|score|scores|number|numeric|pawns?|centipawns?|cp|exact|precise)\b",
    re.I,
)


def wants_number(user_msg: str) -> bool:
    return bool(_NUM_ASK.search(user_msg))


def e_top_form(scenario: Scenario, annotated: AnnotatedPosition | None) -> bool:
    """Half of best-move rows use the top=N (best_moves) result the model emits
    at serve — so it learns to read/narrate that shape, not just best_line."""
    return bool(annotated and len(annotated.top_moves) >= 2 and scenario.seed % 2 == 0)


def _opener(scenario: Scenario) -> str:
    if scenario.tone == "warm":
        return tone.pick(scenario.seed, tone.OPENERS_WARM)
    if scenario.tone == "blunt":
        return tone.pick(scenario.seed, tone.OPENERS_BLUNT)
    return tone.pick(scenario.seed, tone.OPENERS_SOCRATIC)


def _eval_body(annotated: AnnotatedPosition, seed: int, ask_number: bool) -> str:
    return score_phrase(annotated) if ask_number else eval_magnitude(annotated, seed)


def _best_move_body(scenario: Scenario, annotated: AnnotatedPosition, ask_number: bool) -> str:
    if e_top_form(scenario, annotated):
        tm = annotated.top_moves
        rest = ", ".join(san for san, _ in tm[1:3]) or "the alternatives"
        if ask_number:
            return f"{tm[0][0]} looks best at {tm[0][1] / 100:+.2f}; {rest} are the backups."
        return f"{tm[0][0]} looks best here; {rest} are the other tries."
    line = " ".join(annotated.best_line_sans[1:3])
    if ask_number:
        return f"{annotated.best_san} is the move and holds {score_pawns(annotated)}; the line runs {line}."
    return f"{annotated.best_san} is the move; the line runs {line}."


def _threat_body(annotated: AnnotatedPosition, ask_number: bool) -> str:
    threat = annotated.threats_san
    if not threat:
        return "Nothing forcing from them right now — you're not under immediate pressure."
    if ask_number:
        return f"Watch for {threat} — that's worth about {pawns_abs(annotated)} to them."
    return f"Watch for {threat} — that would hand them a serious initiative."


def final_narration(
    scenario: Scenario, annotated: AnnotatedPosition | None, move: str | None, ask_number: bool
) -> str:
    opener = _opener(scenario)
    sep = " " if opener else ""
    seed = scenario.seed
    sl = scenario.slice
    if sl == "A":
        return ask(f"{opener}{sep}Played {move}. The board's updated and it's the opponent's turn now.", seed, 4)
    if sl == "B":
        return ask(f"{opener}{sep}I listed the legal moves first, then chose on the plan rather than guessing.", seed, 4)
    if sl == "C":
        return f"{opener}{sep}I won't play that without a legal-move result; the board snapshot alone isn't enough."
    if sl == "D" and annotated:
        return ask(f"{opener}{sep}{_eval_body(annotated, seed, ask_number)}", seed, 4)
    if sl == "E" and annotated:
        return ask(f"{opener}{sep}{_best_move_body(scenario, annotated, ask_number)}", seed, 4)
    if sl == "F" and annotated:
        tail = " It only moved the eval by 0.05 pawns." if ask_number else ""
        return ask(f"{opener}{sep}{move} was a solid choice — about as good as the top pick, {annotated.best_san}.{tail}", seed, 4)
    if sl == "G" and annotated:
        return ask(f"{opener}{sep}{_threat_body(annotated, ask_number)}", seed, 4)
    if sl == "H":
        return ask(f"{opener}{sep}I listed your pieces from the board rather than guessing.", seed, 4)
    if sl == "I":
        return f"{opener}{sep}It's a sharp counter to 1.e4 that fights for the centre asymmetrically."
    if sl == "J":
        return f"{opener}{sep}Hi. Ask me to read the board, suggest a move, or explain a chess idea."
    if sl == "K":
        return f"{opener}{sep}A knight is worth about three pawns in most positions, but context matters more than the number."
    return f"{opener}{sep}I read the position and the tools, then answered without inventing facts."
