from __future__ import annotations

import random

OPENERS_WARM = (
    "Sure thing —",
    "Happy to help.",
    "Let's look at it together.",
    "Quick read for you:",
    "Here's what I'm seeing:",
    "Glad you asked —",
)

OPENERS_BLUNT = (
    "Here's what I see:",
    "Straight read:",
    "No fluff —",
    "Direct answer:",
    "Cutting to it:",
    "Plain take:",
)

OPENERS_SOCRATIC = (
    "Before I answer, what do you notice first?",
    "What pieces feel underused right now?",
    "If you had to guess, where's the pressure?",
    "Where is your king's safety weakest?",
    "Which piece would you trade away?",
    "What's the most active line you can find?",
)

EVAL_PHRASES = (
    "Roughly equal — neither side has a meaningful edge yet.",
    "Slim edge that needs careful play to convert.",
    "Pressure is real but not winning.",
    "Engine sees a clear advantage building.",
    "Decisive in concrete lines.",
    "Material is balanced; activity decides.",
)

MATE_PHRASES = (
    "Mate is on the board — no quiet moves left.",
    "Forced sequence ends the game.",
    "The king is hunted; only the mating moves matter now.",
    "Material is gone; mate runs the whole evaluation.",
    "All escape squares fall in the forced line.",
    "The check cascade resolves to mate.",
)

APOLOGY_PHRASES = (
    "Engine is unreachable right now — try once more.",
    "I lost the engine briefly. Mind retrying that?",
    "The analysis tool didn't answer in time.",
    "Couldn't reach the engine, sorry. One more try?",
    "Stockfish hiccuped. Want me to retry?",
    "Engine timeout — please ask again.",
)


def pick(seed: int, bank: tuple[str, ...]) -> str:
    return random.Random(seed).choice(bank)
