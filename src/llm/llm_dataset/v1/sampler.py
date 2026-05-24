from __future__ import annotations

import random
from dataclasses import dataclass

from .catalog import (
    OFFICIAL_SKILL,
    OFFICIAL_TOOLS,
    alt_skills,
    alt_tools,
    synthetic_skill_name,
    synthetic_tool_name,
)
from .positions import Position, load_default_bank, sample_position

CHESS_SLICES = set("ABCDEFGHIJK")
UNIVERSALITY_SLICES = {
    "V1_A_skill_index_selection",
    "V1_B_skill_conflict_and_absence",
    "V1_C_dynamic_tool_schema",
    "V1_D_tool_unavailable_and_readonly",
    "V1_E_board_grounding",
    "V1_F_special_chess_rules",
    "V1_G_multi_tool_budget",
    "V1_H_error_recovery",
    "V1_I_eval_language",
    "V1_J_no_tool_and_mixed_intent",
    "V1_K_adversarial_injection",
    "V1_L_rejects_and_audit_fixtures",
}

CATEGORY_FOR_SLICE = {
    "A": "opening",
    "B": "middlegame",
    "C": "middlegame",
    "D": "middlegame",
    "E": "middlegame",
    "F": "middlegame",
    "G": "middlegame",
    "H": "middlegame",
    "I": "opening",
    "J": "opening",
    "K": "opening",
    "V1_F_special_chess_rules": "terminal",
}


@dataclass(frozen=True)
class Scenario:
    slice: str
    intent: str
    position: Position | None
    skills_index: tuple
    tool_manifest: tuple
    plugin_context: dict
    name_family: str
    tone: str
    length: str
    seed: int


def plan_scenarios(plan: dict[str, int], seed: int) -> list[Scenario]:
    rng = random.Random(seed)
    bank = load_default_bank()
    scenarios: list[Scenario] = []
    for slice_name, count in plan.items():
        for n in range(count):
            scenarios.append(_one(slice_name, bank, rng, n))
    return scenarios


def _one(slice_name: str, bank, rng: random.Random, n: int) -> Scenario:
    name_family = "synthetic" if rng.random() < 0.30 else "real"
    tone = rng.choice(["warm", "blunt", "socratic"])
    length = rng.choice(["short", "medium", "long"])
    category = CATEGORY_FOR_SLICE.get(slice_name, "middlegame")
    needs_position = slice_name in CHESS_SLICES or slice_name == "V1_F_special_chess_rules"
    position = (
        sample_position(bank, category, seed=rng.randint(1, 10**9))
        if needs_position
        else None
    )
    skills_index = _skills(rng, name_family)
    tool_manifest = _tools(rng, name_family, slice_name)
    plugin_context = {
        "installed": ["chess-official", "user-skills"],
        "enabled": ["chess-official", "user-skills"],
    }
    intent = f"{slice_name.lower()}_{n:04d}"
    return Scenario(
        slice_name,
        intent,
        position,
        skills_index,
        tool_manifest,
        plugin_context,
        name_family,
        tone,
        length,
        rng.randint(1, 10**9),
    )


def _skills(rng: random.Random, name_family: str) -> tuple:
    base = [OFFICIAL_SKILL] + rng.sample(alt_skills(), 4)
    if name_family == "synthetic":
        base += [
            {
                "name": synthetic_skill_name(rng.randint(1, 10**6)),
                "description": "Domain-neutral skill for the harness universality test.",
            }
        ]
    rng.shuffle(base)
    return tuple(base)


def _tools(rng: random.Random, name_family: str, slice_name: str) -> tuple:
    base = list(OFFICIAL_TOOLS) + rng.sample(alt_tools(), 3)
    if name_family == "synthetic" or slice_name == "V1_C_dynamic_tool_schema":
        base += [
            {
                "name": synthetic_tool_name(rng.randint(1, 10**6)),
                "description": "Harness universality test tool. Args defined inline.",
                "args": {"input": "required"},
                "applies_when": "always",
            }
        ]
    rng.shuffle(base)
    return tuple(base)
