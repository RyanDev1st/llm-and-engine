from __future__ import annotations

import random
from dataclasses import dataclass

from .catalog import (
    HUMAN_CHAT_SKILL,
    OFFICIAL_SKILL,
    USER_SKILL_TOOLS,
    alt_skills,
    alt_tools,
    compute_tools,
    official_tools,
    synthetic_skill_name,
    synthetic_tool_name,
    with_plugin,
)
from .positions import Position, load_default_bank, sample_position

CHESS_SLICES = set("ABCDEFGHIJK")
MULTITURN_SLICE = "V1_P_multiturn_followup"  # chess-grounded but its own renderer
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
    "V1_M_marketplace_navigation",
    "V1_N_human_chat_skill_bridge",
    "V1_Q_no_skill_direct",
}
# Compute-grounding slice: its own renderer, domain-neutral, calls the calc tool.
COMPUTE_SLICES = {"V1_R_compute_grounding"}

PROMPT_STYLES = ("formal", "casual", "slang", "typo", "anxious", "beginner")

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
    prompt_style: str
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
    needs_position = (slice_name in CHESS_SLICES or slice_name == "V1_F_special_chess_rules"
                      or slice_name == MULTITURN_SLICE)
    position = (
        sample_position(bank, category, seed=rng.randint(1, 10**9))
        if needs_position
        else None
    )
    skills_index = _skills(rng, name_family)
    tool_manifest = _tools(rng, name_family, slice_name)
    prompt_style = PROMPT_STYLES[n % len(PROMPT_STYLES)]
    plugin_context = {
        "installed": ["chess-official", "user-skills", "market-tactics", "synthetic-pack"],
        "enabled": ["chess-official", "user-skills", "synthetic-pack"],
        "marketplace": ["market-openings", "market-endgames"],
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
        prompt_style,
        rng.randint(1, 10**9),
    )


def _skills(rng: random.Random, name_family: str) -> tuple:
    user = with_plugin(rng.sample(alt_skills(), 2), "user-skills", "user_skill")
    market = with_plugin(rng.sample(alt_skills(), 2), "market-tactics", "marketplace_plugin")
    base = [OFFICIAL_SKILL, HUMAN_CHAT_SKILL] + user + market
    if name_family == "synthetic":
        base += [
            {
                "name": synthetic_skill_name(rng.randint(1, 10**6)),
                "description": "Domain-neutral skill for the harness universality test.",
                "plugin": "synthetic-pack",
                "source": "synthetic_plugin",
                "enabled": True,
            }
        ]
    rng.shuffle(base)
    return tuple(base)


def _tools(rng: random.Random, name_family: str, slice_name: str) -> tuple:
    base = official_tools() + with_plugin(USER_SKILL_TOOLS, "user-skills", "user_skill")
    base += with_plugin(rng.sample(alt_tools(), 2), "user-skills", "user_skill")
    base += with_plugin(rng.sample(alt_tools(), 1), "market-tactics", "marketplace_plugin", enabled=False)
    if name_family == "synthetic" or slice_name == "V1_C_dynamic_tool_schema":
        base += [
            {
                "name": synthetic_tool_name(rng.randint(1, 10**6)),
                "description": "Harness universality test tool. Args defined inline.",
                "args": {"input": "required"},
                "applies_when": "always",
                "plugin": "synthetic-pack",
                "source": "synthetic_plugin",
                "enabled": True,
            }
        ]
    if slice_name in COMPUTE_SLICES:
        base += compute_tools()       # calc must be listed for the model to call it
    rng.shuffle(base)
    return tuple(base)
