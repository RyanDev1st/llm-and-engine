"""Stage 2 — V1_T_audited_plan: a self-authored checklist whose CHECKABLE boxes are
verified by RUNNING the python tool and reading stdout, never by asserting.

The Stage-0 verify-then-claim shape (renderer/compute.py) scaled to a multi-box plan, in
the product's domain: the boxes are CHESS claims (material points, accuracy, score swing —
the chess compute families), audited via python. The model commits the goal(s), authors a
<plan>, loads the `plan-audit` skill, then for each tool-checkable box runs python and
grounds the verdict in the real output. It also teaches split determinism: a SEMANTIC box
("is my position actually better?") is left SOFT — a judgment, not a faked tool check.

v5 pure-chess + flat catalog. Honest-partial (disabled-skill) is dropped — it has no
trigger in a flat catalog; the audit-via-tool + split-determinism lessons remain."""
from __future__ import annotations

import random
from typing import Any

from ..catalog import chess_skills, chess_tools
from .compute import _VERIFY, _exec
from .planning import goal_block, plan_block
from .tags import skill_call_msg, tool_call_msg, tool_calls_of, tool_result_msg

_AUDIT_BODY = (
    "# plan-audit\n"
    "When to use: a goal with checkable claims you must not just assert.\n"
    "Steps:\n"
    "1. For each checkable box, run the python tool and read its output.\n"
    "2. Mark the box from that output — never state a number you didn't run.\n"
    "3. Leave judgment/positional boxes soft; say so, don't fake a tool check.\n"
    "Constraint: if a box can't be verified, report it and stop — don't spin."
)

# Semantic (NOT tool-checkable) chess box prompts + the soft, non-audited read.
_SEMANTIC = (
    ("is my position actually better", "and whether my position is genuinely better"),
    ("does my attack look sound", "and whether my attack looks sound"),
    ("is my pawn structure okay", "and whether my pawn structure holds up"),
    ("does my middlegame plan make sense", "and whether my plan makes sense"),
)
_SOFT_VERDICT = (
    "that's a judgment call, not a tool check — to my read it mostly holds, but I won't fake a measured verdict on it",
    "I can't run that one — it's positional, so I'll give my honest read rather than a fake audit number",
)
def _box(seed: int, fam, idx: int) -> tuple[str, str, str]:
    """One tool-checkable box: (user-fragment, python script, grounded verdict). Reuses a
    chess compute verify family — build() turns the real executor output into the verdict,
    so the audit value is grounded, never asserted."""
    r = random.Random(seed * 101 + idx * 7)
    prompt, code, build = fam(r)
    val = _exec(code).split("output: ", 1)[1]      # exact executor stdout -> grounded
    return prompt, code, build(val)


def _run_step(code: str) -> list[dict]:
    """Assistant python-audit step + its real tool result."""
    return [tool_call_msg("python", {"code": code}), tool_result_msg("python", _exec(code))]


def render_audited_plan_row(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    mode = "plan"
    semantic = (seed % 4 == 1)                      # ~25% split-determinism (one soft box)

    fam_a = rng.choice(_VERIFY)
    p_a, code_a, verdict_a = _box(seed, fam_a, 0)
    goals = ["verify " + p_a.rstrip("?.")]
    boxes = [("verify the first claim", "python")]

    if semantic:
        sem_ask, sem_frag = rng.choice(_SEMANTIC)
        prompt = f"{p_a}, {sem_frag}?"
        goals.append("judge " + sem_ask)
        boxes.append(("judge the second box — soft, not tool-checkable", "none"))
    else:
        fam_b = rng.choice([f for f in _VERIFY if f is not fam_a])
        p_b, code_b, verdict_b = _box(seed, fam_b, 1)
        prompt = f"{p_a}, and also {p_b.rstrip('?')}?"
        goals.append("verify " + p_b.rstrip("?."))
        boxes.append(("verify the second claim", "python"))
    boxes.append(("synthesize one combined answer", "none"))

    # The plan (goal + checklist) rides the FIRST action's native reasoning channel — here
    # the load_skill(plan-audit) call — so it's committed before any audit runs.
    plan_text = goal_block(seed, goals) + "\n" + plan_block(boxes)
    load = skill_call_msg("plan-audit")
    load["reasoning"] = plan_text

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    messages.append(load)
    messages.append(tool_result_msg("load_skill", _AUDIT_BODY))

    messages += _run_step(code_a)
    if semantic:
        final = f"On the first: {verdict_a} On the second: {random.Random(seed * 31 + 5).choice(_SOFT_VERDICT)}."
    else:
        messages += _run_step(code_b)
        final = f"On the first: {verdict_a} On the second: {verdict_b}"
    messages.append({"role": "assistant", "content": final})
    return _envelope(seed, messages, ["plan-audit"], mode)


def _envelope(seed: int, messages: list[dict], selected: list[str], mode: str) -> dict[str, Any]:
    expected = [tc["name"] for msg in messages if msg["role"] == "assistant"
                for tc in tool_calls_of(msg) if tc["name"] != "load_skill"]
    return {
        "id": f"v1_t_audited_{seed:09d}",
        "slice": "V1_T_audited_plan",
        "kind": "audited_plan",
        "reasoning_mode": mode,
        "intent": f"v1_t_{seed:06d}",
        "plugin_context": {},
        "skills_index": chess_skills(),
        "selected_skills": selected,
        "tool_manifest": chess_tools(),
        "expected_tool_calls": expected,
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "selected_skill_exists", "skill_body_strict", "narration_grounded",
                             "goal_before_plan", "plan_boxes_bound", "audit_boxes_grounded"],
        "position_fen": None,
        "stockfish_truth": None,
    }
