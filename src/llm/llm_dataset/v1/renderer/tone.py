from __future__ import annotations

import random

# Persona openers removed: SFT trains tool use, not tone. Finals are plain,
# grounded sentences. Pools kept (as empty connectors) so renderer call sites
# and `scenario.tone` selection keep working without change.
OPENERS_WARM = ("",)
OPENERS_BLUNT = ("",)
OPENERS_SOCRATIC = ("",)

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
