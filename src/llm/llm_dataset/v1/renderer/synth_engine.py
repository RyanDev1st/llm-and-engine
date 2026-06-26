"""Seeded synthetic engine scene for the universality chess-tool slices.

The chess slices A-K narrate REAL Stockfish output (varied per position). The
universality slices teach the HARNESS protocol against a board the model can't
see, so their engine results were canned constants: every V1_G row showed
`+0.15` / `Nf3 (+0.20)` and the final said "eval near equal, Nf3 tops" regardless.
That trains the narration to be INDEPENDENT of the tool output — the opposite of
grounding, which is the product's whole point.

This makes each row's engine numbers vary by seed AND lets the final be DERIVED
from those exact numbers, so the model learns to copy whatever the tool returned
(the anti-fabrication lesson) instead of memorizing one line. Every format string
mirrors the live backend (tools._best_move / eval / threats / board_state) exactly,
so train == serve shape. No facts are invented at narration time: the final quotes
only values that appear in this scene's tool results.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .tags import tool_call

# Plausible candidate moves (piece moves are grounding "facts"; the regex in
# validate._FACT matches them, so any move named in the final must come from the
# best_moves result here). Mix of developing moves so top/backup vary per seed.
_MOVES = ("Nf3", "Nc3", "Bb5", "Bc4", "Be2", "Qd2", "Rd1", "Nd5", "Bg5", "Nf5",
          "Re1", "Qe2", "Bd3", "Nbd2", "h3", "a4", "c4", "d4", "e4", "g3")
# Eval buckets sampled from (white POV centipawns); spans equal -> winning, both
# colors. Jitter makes the 2-decimal value distinct row to row.
_CP_BASE = (0, 9, 16, 24, 33, 48, 62, 85, 115, 150, 195, 255, 330,
            -11, -19, -28, -42, -58, -80, -120, -175, -240)


@dataclass(frozen=True)
class Scene:
    cp: int                      # eval, white POV centipawns
    turn: str                    # "white" | "black"
    legal_count: int
    top: tuple[str, int]         # (san, cp white POV)
    backups: tuple[tuple[str, int], ...]
    threat: str | None           # opponent's best SAN, or None
    threat_cp: int               # score for them (white POV), when threat present


def engine_scene(seed: int) -> Scene:
    r = random.Random(seed * 131 + 17)
    cp = r.choice(_CP_BASE) + r.randint(-6, 6)
    turn = "white" if r.random() < 0.5 else "black"
    legal_count = r.randint(16, 40)
    moves = r.sample(_MOVES, 3)
    # Top move realizes the eval; backups are a touch worse for the side to move.
    d1, d2 = r.randint(6, 22), r.randint(24, 55)
    sign = 1 if cp >= 0 else -1
    top = (moves[0], cp)
    backups = ((moves[1], cp - sign * d1), (moves[2], cp - sign * d2))
    has_threat = r.random() < 0.5
    threat = r.choice(_MOVES) if has_threat else None
    threat_cp = -sign * r.randint(20, 90) if has_threat else 0
    return Scene(cp, turn, legal_count, top, backups, threat, threat_cp)


def _pawns(cp: int) -> str:
    return f"{cp / 100:+.2f} pawns from white POV"


def board_state_result(s: Scene) -> str:
    return f"board_state: turn={s.turn}, check=no, legal_count={s.legal_count}"


def eval_result(s: Scene) -> str:
    return f"score: {_pawns(s.cp)}, depth=15"


def threats_result(s: Scene) -> str:
    if not s.threat:
        return "threats: none significant"
    return f"threats: opponent's best is {s.threat}, score for them: {_pawns(s.threat_cp)}"


def best_move_result(s: Scene) -> str:
    moves = (s.top, *s.backups)
    return "best_moves: " + "; ".join(f"{i}. {san} ({cp / 100:+.2f})" for i, (san, cp) in enumerate(moves, 1))


def chain(s: Scene) -> list[tuple[str, str]]:
    """The V1_G multi-tool budget chain: (call, result) pairs in the live format.
    First call is the decision step; the rest are rote execution (silent <think>).
    Three calls (board -> eval -> best_move): still multi-tool, and keeps the
    longest slice safely under the train seq ceiling even on the largest manifest."""
    return [
        (tool_call("board_state", "fields=basic"), board_state_result(s)),
        (tool_call("eval", "depth=15"), eval_result(s)),
        (tool_call("best_move", "depth=15 top=3"), best_move_result(s)),
    ]


def _standing(cp: int) -> str:
    side = "White" if cp > 0 else "Black"
    a = abs(cp)
    if a < 35:
        return "it's about level"
    if a < 90:
        return f"{side} is a touch better"
    if a < 200:
        return f"{side} is clearly better"
    if a < 450:
        return f"{side} is close to winning"
    return f"{side} is winning"


def budget_verdict(s: Scene) -> str:
    """Final reply for V1_G — derived from this scene's tool results, so every
    fact it states (eval number, top move) is grounded. Varies per seed. Kept
    tight (eval + top move only) so the longest slice stays under the seq ceiling."""
    return (f"Eval is {s.cp / 100:+.2f}, so {_standing(s.cp)}. "
            f"{s.top[0]} is the engine's top move within the tool budget.")


def equal_eval_scene_cp(seed: int) -> int:
    """A varied NEAR-ZERO eval for V1_I (the 'starting position is equal' lesson):
    keeps |cp| small so 'basically equal' stays true, but the exact number differs
    row to row so the model copies it instead of memorizing +0.12."""
    r = random.Random(seed * 149 + 5)
    return r.choice((-18, -12, -8, -5, 0, 4, 7, 11, 16, 21, 25))


def recovery_eval_cp(seed: int) -> int:
    """Varied recovered eval for V1_H (error-recovery retry result)."""
    r = random.Random(seed * 151 + 9)
    return r.choice(_CP_BASE) + r.randint(-5, 5)
