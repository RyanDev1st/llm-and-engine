"""PLAN-mode emission for Stage 1 (goal-driven completion) and Stage 2 (audited plan).

Purpose is NOT implementation planning. `<goal>` is an ANTI-EARLY-STOP / turn-looping
mechanism: a small model's failure is doing ONE tool call then giving a half-answer when
the request needed several. Committing the objective as held state makes it KEEP LOOPING
through every necessary tool/skill until the goal is met. The checklist (`<plan>`) is just
the set of necessary steps so it knows when it is actually done -- inspiration from the
superpowers plan/execute loop, not a copy. Complements think mode (think = reason per
step; goal = hold the objective across the loop). Deterministic by design: the structure
forces a weak model to finish the multi-step sequence instead of abandoning it.

Flow: COMMIT goal -> LIST needed steps -> DO EVERY box in order -> SYNTHESIZE. Honest-
partial if a box truly can't be done (loop-cap).

Serve routing: `<goal>` and `<plan>` go to a separate plan PANEL (like Claude Code's todo
list), NOT the chat. The row emits the plan ONCE (seq economy); the deterministic serve
gate ticks each box when its bound action's result lands — so we don't re-emit the whole
checklist every step. Box binding = the trailing `(name)`, which the gate maps to the
executed `<skill>name</skill>` / `<tool>name …</tool>`.

Box format (the gate parses this exactly):  `- [ ] <action> (<binding>)`
"""
from __future__ import annotations

import random

# Seeded framings for the goal restatement — vary the lead so it's not one rote token
# pattern, but keep the <goal> wrapper constant (like <think>). Content varies per row.
_GOAL_LEAD = ("", "", "Goal: ", "I need to ", "The ask: ")


def goal_block(seed: int, goal: str) -> str:
    g = (goal or "").strip().rstrip("?.!")
    lead = random.Random(seed * 13 + 2).choice(_GOAL_LEAD)
    inner = (lead + g) if not lead or lead.endswith(" ") else f"{lead}{g}"
    return f"<goal>{inner}</goal>"


def plan_block(boxes: list[tuple[str, str]], done: int = 0) -> str:
    """`<plan>` with checkboxes. boxes = [(action_text, binding_name), ...]; the first
    `done` boxes render checked. binding_name is the skill/tool the box maps to (the
    serve gate ticks the box when that action's result lands)."""
    lines = []
    for i, (action, binding) in enumerate(boxes):
        mark = "x" if i < done else " "
        lines.append(f"- [{mark}] {action} ({binding})")
    return "<plan>\n" + "\n".join(lines) + "\n</plan>"


def partial_report(boxes: list[tuple[str, str]], done: int, blocker: str) -> str:
    """Honest-partial final when the loop caps before all boxes clear: state what's done
    and what's blocked, never fake the rest. boxes/done as in plan_block."""
    n = len(boxes)
    cleared = ", ".join(a for a, _ in boxes[:done]) or "nothing yet"
    blocked = boxes[done][0] if done < n else "the remaining step"
    return (f"I cleared {done} of {n}: {cleared}. I couldn't finish \"{blocked}\" — "
            f"{blocker}. Tell me how you'd like to proceed and I'll continue.")
