from __future__ import annotations

import random
from dataclasses import dataclass

from .catalog import chess_skills, chess_tools
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
# Stage 1 compound-plan slice: its own renderer builds the multi-skill index/manifest.
COMPOUND_SLICES = {"V1_S_compound_plan"}
# Stage 2 audited-plan slice: its own renderer (audit skill + python-verified boxes).
AUDIT_SLICES = {"V1_T_audited_plan"}

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
    plugin_context: dict = {}   # v5 flat catalog: serve aggregates plugins flat, no gating
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
    """The flat pure-chess skill set, order shuffled so the model routes by description,
    not by position. Same set every row (no distractors) — v5 is a chess-only catalog."""
    skills = chess_skills()
    rng.shuffle(skills)
    return tuple(skills)


def _tools(rng: random.Random, name_family: str, slice_name: str) -> tuple:
    """The flat pure-chess tool manifest (incl. python + the opening/analysis specialist
    tools), order shuffled. No plugin gating, no cross-domain distractors."""
    tools = chess_tools()
    rng.shuffle(tools)
    return tuple(tools)
