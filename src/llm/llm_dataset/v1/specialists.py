"""Chess specialist 'domains' for the v5 plan-mode slices (compound V1_S, audited V1_T)
and specialist routing (V1_U). Each is a REAL served specialist skill (descriptions
verbatim from backend/plugins/{analysis,openings,puzzles}.py) with a terse body, its
primary tool, and grounded (call, tool_result, finding) scenes — so a multi-skill chess
plan reads like the real product. The tool_result/finding in each scene are a consistent
triple (the finding only states what the result shows), so a final built from findings
stays grounded even though plan rows don't run a live engine."""
from __future__ import annotations

import random
from dataclasses import dataclass

Scene = tuple[str, str, str]   # (tool call args, tool result, grounded finding)


@dataclass(frozen=True)
class Specialist:
    skill: str
    description: str
    tool: str
    tool_args: dict
    applies_when: str
    scenes: tuple[Scene, ...]
    prompts: tuple[str, ...]

    def body(self) -> str:
        return f"# {self.skill}\nUse {self.tool} for this, read the result, then report the finding."


GAME_REVIEWER = Specialist(
    skill="game-reviewer",
    description="Use when the user asks how they played overall, their accuracy, or to find blunders across the game.",
    tool="accuracy_report", tool_args={"depth": "required"}, applies_when="has_history",
    scenes=(
        ("depth=14", "accuracy: white=86%, black=79%, moves=34",
         "you played at 86% accuracy to their 79% over 34 moves — solid, with a couple of slips to tidy"),
        ("depth=14", "accuracy: white=72%, black=81%, moves=41",
         "you came in at 72% to their 81% across 41 moves — a few costly inaccuracies dragged it down"),
        ("depth=12", "accuracy: white=91%, black=88%, moves=28",
         "this was a clean game — 91% to their 88% over 28 moves, very few real mistakes"),
    ),
    prompts=("how did I play?", "what was my accuracy this game?", "review my game", "how accurate was I?"),
)

OPENING_ADVISOR = Specialist(
    skill="opening-advisor",
    description="Use when the user asks what opening this is, or for opening plans, theory, or a repertoire.",
    tool="name_opening", tool_args={}, applies_when="has_history",
    scenes=(
        ("", "opening: Sicilian Defence",
         "you're in a Sicilian — an unbalanced fight where you play the open c-file and queenside"),
        ("", "opening: Italian Game",
         "this is the Italian — develop fast, eye f7, and aim for the c3-d4 central break"),
        ("", "opening: French Defence",
         "it's a French — solid but cramped; freeing the c8 bishop is your long-term job"),
    ),
    prompts=("what opening is this?", "what should I study from here?", "name this opening",
             "what's the plan in this opening?"),
)

TACTICAL_PUZZLES = Specialist(
    skill="tactical-puzzles",
    description="Use when the user wants a tactical puzzle, to practice or hone tactics, or to be coached through a combination.",
    tool="random_position", tool_args={"kind": ["puzzle", "scramble", "open"]}, applies_when="always",
    scenes=(
        ("kind=puzzle", "position set: white to move, motif=fork",
         "set you a fork puzzle — White to move; hunt for the move that hits two pieces at once"),
        ("kind=puzzle", "position set: black to move, motif=pin",
         "here's a pin puzzle — Black to move; look for the move that nails a piece to the king"),
        ("kind=puzzle", "position set: white to move, motif=back-rank",
         "a back-rank puzzle for you — White to move; the king's trapped on the back rank"),
    ),
    prompts=("give me a puzzle", "set me a tactic", "I want to practice tactics", "puzzle me"),
)

SPECIALISTS = (GAME_REVIEWER, OPENING_ADVISOR, TACTICAL_PUZZLES)
BY_SKILL = {s.skill: s for s in SPECIALISTS}


def pick_two(seed: int) -> tuple[Specialist, Specialist]:
    """Two distinct specialists for a compound chess plan (e.g. review + opening study)."""
    return tuple(random.Random(seed * 2654435761 % (2 ** 32)).sample(SPECIALISTS, 2))


def scene(spec: Specialist, seed: int) -> Scene:
    return random.Random(seed * 17 + 5).choice(spec.scenes)
