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
from ..board_facts import king_moves, piece_summary
from ..sampler import Scenario
from . import tone
from .grounded import threat_reason, why_best_move
from .leadins import ask
from .review import ReviewFacts, why_review
from .text import eval_magnitude, pawns_abs, score_phrase

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
    # The grounded composer: move + a REASON drawn from the move's true facts +
    # evidence (line, or top-N alternatives) + standing. Replaces the old
    # move-and-line-only body that never said WHY (the 80%-no-why gap).
    return why_best_move(annotated, ask_number, scenario.seed, top_form=e_top_form(scenario, annotated))


# No-threat phrasings — the common case (most positions have no forcing threat),
# so one fixed string repeated ~150x. Varied by seed; all say the same thing.
_NO_THREAT = (
    "Nothing forcing from them right now — you're not under immediate pressure.",
    "No real threat at the moment — nothing of theirs is forcing.",
    "They've got nothing forcing here; you're not in any immediate danger.",
    "No immediate threat to deal with — nothing of theirs hits hard right now.",
    "Nothing pressing from them this move; you're not under fire yet.",
    "There's no forcing threat on the board for them right now.",
    "They aren't threatening anything concrete at the moment.",
    "No direct threat to answer — nothing of theirs is forcing just now.",
)


def _threat_body(annotated: AnnotatedPosition, ask_number: bool, seed: int = 0) -> str:
    # threats_san is the opponent's best free move; it's only a REAL threat if it does
    # something concrete (capture / fork / check / mate / pin). threat_reason returns the
    # grounded, threat-framed point, or None for a quiet move -> nothing forcing to warn of.
    threat = annotated.threats_san
    reason = threat_reason(annotated.fen, threat, seed) if threat else None
    if not reason:
        return random.Random(seed * 37 + 7).choice(_NO_THREAT)
    if ask_number:
        return f"Watch for {threat} — {reason}; that's worth about {pawns_abs(annotated)} to them."
    return f"Watch for {threat} — {reason}."


# Greeting finals (J). B/C/H NO LONGER live here: they were process-narration ("I checked the
# legal moves, then chose by the plan") with zero position content, which the model learned and
# reproduced live. They now render GROUNDED content from the position (see final_narration). J is
# a real greeting (not narration); one phrasing is picked per seed.
_LESSON_FINALS = {
    "J": ("Hi. Ask me to read the board, suggest a move, or explain a chess idea.",
          "Hey there. I can analyze a position, recommend a move, or talk through a plan.",
          "Happy to help — point me at a board, a move to review, or a concept to explain.",
          "Hello. I can read a position, find a move, or break down an idea.",
          "Hi. I'm set up for board reads, move suggestions, and explaining chess ideas.",
          "Hey. Give me a position or a question and I'll read it, evaluate, or explain.",
          "Hey. I can read a position, suggest a move, or explain an idea.",
          "Hi there. Show me a board and I'll analyze it, find a move, or walk through a plan.",
          "Hello. I'm here to read positions, recommend moves, and explain chess ideas.",
          "Hey. Point me at a position or a question and I'll dig in.",
          "Hi. I handle board reads, move suggestions, and explaining what's going on.",
          "Hello there. I read boards, suggest moves, and break down chess ideas.",
          "Hi. Give me a position or a chess question and I'll take it from there.",
          "Hey. I'm ready to analyze a board, recommend a move, or explain a concept.",
          "Hi. I can look at a position, find the best move, or explain the idea behind it.",
          "Hello. Set up a board or ask a chess question and I'll help."),
}


def final_narration(
    scenario: Scenario, annotated: AnnotatedPosition | None, move: str | None, ask_number: bool,
    kb_answer: str | None = None, review: ReviewFacts | None = None,
) -> str:
    opener = _opener(scenario)
    sep = " " if opener else ""
    seed = scenario.seed
    sl = scenario.slice
    if sl == "A":
        return ask(f"{opener}{sep}Played {move}. The board's updated and it's the opponent's turn now.", seed, 4)
    if sl == "B" and annotated:
        # B surveys the legal moves THEN asks the engine -> a grounded recommendation (the same
        # composer slice E uses), not the old "I checked the legal moves, then chose by the plan".
        return ask(f"{opener}{sep}{_best_move_body(scenario, annotated, ask_number)}", seed, 4)
    if sl == "H" and annotated:
        # H answers "what's left?" with the ACTUAL material read off the board (the list_pieces result).
        return f"{opener}{sep}You've still got {piece_summary(annotated.fen)}."
    if sl == "C" and annotated:
        # C answers the legality question from the king's real legal squares — a STATEMENT (per
        # test_knowledge_and_greeting_finals_stay_statements), not a refusal to guess.
        ksq, ksans = king_moves(annotated.fen)
        body = (f"Your king on {ksq} can move to {', '.join(ksans)} — those are its legal squares." if ksans
                else f"Your king on {ksq} has no legal squares right now; it's boxed in.")
        return f"{opener}{sep}{body}"
    if sl in _LESSON_FINALS:  # J greeting (seeded paraphrase; stays a STATEMENT)
        return f"{opener}{sep}{random.Random(seed * 53 + 11).choice(_LESSON_FINALS[sl])}"
    if sl == "D" and annotated:
        return ask(f"{opener}{sep}{_eval_body(annotated, seed, ask_number)}", seed, 4)
    if sl == "E" and annotated:
        return ask(f"{opener}{sep}{_best_move_body(scenario, annotated, ask_number)}", seed, 4)
    if sl == "F" and annotated and review:
        # Grounded verdict: praise a good move for its real point, or name the better move.
        return ask(f"{opener}{sep}{why_review(review, annotated, ask_number=ask_number, seed=seed)}", seed, 4)
    if sl == "F" and annotated:                              # defensive: no review computed
        return ask(f"{opener}{sep}{move} is a playable move here.", seed, 4)
    if sl == "G" and annotated:
        return ask(f"{opener}{sep}{_threat_body(annotated, ask_number, seed)}", seed, 4)
    if sl == "I":
        return f"{opener}{sep}{kb_answer}"
    if sl == "K":
        return f"{opener}{sep}{kb_answer}"
    return f"{opener}{sep}I read the position and the tools, then answered without inventing facts."
