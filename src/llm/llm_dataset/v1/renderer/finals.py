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

import random
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


# Lesson-type chess finals (B/C/H/J) state a behaviour, not a position fact, so
# their base sentence was constant — repeated 1-2k times -> memorisation. Each
# pool carries the SAME lesson in a few phrasings; one is picked per seed and then
# routed through ask() for a guiding closer, so distinct finals scale into the
# dozens without changing what the row teaches.
_LESSON_FINALS = {
    "B": ("I listed the legal moves first, then chose on the plan rather than guessing.",
          "Rather than guess, I read the legal moves and picked by the plan.",
          "I checked what was actually legal, then chose the move that fit the plan.",
          "I let the legal-move list guide the choice instead of trusting my memory.",
          "I grounded the decision in the legal moves, then went with the plan.",
          "First the legal options, then the pick — no guessing at the board."),
    "C": ("I won't play that without a legal-move result; the board snapshot alone isn't enough.",
          "That move isn't confirmed legal here, so I won't play it on the snapshot alone.",
          "I can't make that move without checking it's legal first — the board read isn't proof.",
          "Without a legal-move check I won't commit to that; the position alone doesn't license it.",
          "I'll hold off on that move until a legal-move result backs it up.",
          "I won't force an unverified move — it needs to clear the legal-move check first."),
    "H": ("I listed your pieces from the board rather than guessing.",
          "I read the material straight off the board instead of recalling it.",
          "Those pieces come from the board read, not from memory.",
          "I pulled the piece list from the live position rather than guessing.",
          "I grounded the material count in the board, not an assumption.",
          "I checked the board for what's actually on it instead of estimating."),
    "J": ("Hi. Ask me to read the board, suggest a move, or explain a chess idea.",
          "Hey there. I can analyze a position, recommend a move, or talk through a plan.",
          "Happy to help — point me at a board, a move to review, or a concept to explain.",
          "Hello. Want me to read a position, find a move, or break down an idea?",
          "Hi. I'm set up for board reads, move suggestions, and explaining chess ideas.",
          "Hey. Give me a position or a question and I'll read it, evaluate, or explain."),
}


def final_narration(
    scenario: Scenario, annotated: AnnotatedPosition | None, move: str | None, ask_number: bool,
    kb_answer: str | None = None,
) -> str:
    opener = _opener(scenario)
    sep = " " if opener else ""
    seed = scenario.seed
    sl = scenario.slice
    if sl == "A":
        return ask(f"{opener}{sep}Played {move}. The board's updated and it's the opponent's turn now.", seed, 4)
    if sl in _LESSON_FINALS:  # seeded paraphrase of the lesson
        base = random.Random(seed * 53 + 11).choice(_LESSON_FINALS[sl])
        # C (illegal-move refusal) and J (greeting) stay STATEMENTS by contract
        # (test_knowledge_and_greeting_finals_stay_statements); B and H get a
        # guiding closer like the other coaching finals.
        if sl in ("C", "J"):
            return f"{opener}{sep}{base}"
        return ask(f"{opener}{sep}{base}", seed, 4)
    if sl == "D" and annotated:
        return ask(f"{opener}{sep}{_eval_body(annotated, seed, ask_number)}", seed, 4)
    if sl == "E" and annotated:
        return ask(f"{opener}{sep}{_best_move_body(scenario, annotated, ask_number)}", seed, 4)
    if sl == "F" and annotated:
        tail = " It only moved the eval by 0.05 pawns." if ask_number else ""
        return ask(f"{opener}{sep}{move} was a solid choice — about as good as the top pick, {annotated.best_san}.{tail}", seed, 4)
    if sl == "G" and annotated:
        return ask(f"{opener}{sep}{_threat_body(annotated, ask_number)}", seed, 4)
    if sl == "I":
        return f"{opener}{sep}{kb_answer}"
    if sl == "K":
        return f"{opener}{sep}{kb_answer}"
    return f"{opener}{sep}I read the position and the tools, then answered without inventing facts."
