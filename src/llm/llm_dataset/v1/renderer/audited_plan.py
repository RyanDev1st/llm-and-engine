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
from .thinking import gated_think

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
_LEAD_LOAD = ("Loading the audit procedure.", "Pulling up how to audit this.", "First, the audit skill.")
_LEAD_RUN = ("Running the check.", "Verifying that box now.", "Let me run the numbers, not guess them.")


def _pick(seed: int, step: int, pool: tuple[str, ...]) -> str:
    return random.Random(seed * 31 + step).choice(pool)


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def _box(seed: int, fam, idx: int) -> tuple[str, str, str]:
    """One tool-checkable box: (user-fragment, python script, grounded verdict). Reuses a
    chess compute verify family — build() turns the real executor output into the verdict,
    so the audit value is grounded, never asserted."""
    r = random.Random(seed * 101 + idx * 7)
    prompt, code, build = fam(r)
    val = _exec(code).split("output: ", 1)[1]      # exact executor stdout -> grounded
    return prompt, code, build(val)


def _run_step(seed: int, code: str, step: int, goal: str, mode: str) -> list[dict]:
    """Assistant python-audit step + its real tool result."""
    think = gated_think(seed, "python", step, mode=mode, kind="execute", goal=goal, have="skill")
    call = _join(think, _pick(seed, step, _LEAD_RUN), f"<tool>python code={code}</tool>")
    return [{"role": "assistant", "content": call}, {"role": "tool", "content": _exec(code)}]


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
    goal_text = "; ".join(goals)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    messages.append({"role": "assistant", "content": goal_block(seed, goals) + "\n" + plan_block(boxes)})
    messages.append({"role": "assistant", "content": _join(
        gated_think(seed, "load_skill", 1, mode=mode, kind="select", goal=goal_text),
        _pick(seed, 1, _LEAD_LOAD), "<skill>plan-audit</skill>")})
    messages.append({"role": "tool", "content": _AUDIT_BODY})

    messages += _run_step(seed, code_a, 2, goal_text, mode)
    if semantic:
        final = f"On the first: {verdict_a} On the second: {_pick(seed, 5, _SOFT_VERDICT)}."
    else:
        messages += _run_step(seed, code_b, 4, goal_text, mode)
        final = f"On the first: {verdict_a} On the second: {verdict_b}"
    messages.append({"role": "assistant", "content": final})
    return _envelope(seed, messages, ["plan-audit"], mode)


import re as _re
_TOOL = _re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")


def _envelope(seed: int, messages: list[dict], selected: list[str], mode: str) -> dict[str, Any]:
    expected = [m for c in (msg["content"] for msg in messages if msg["role"] == "assistant")
                for m in _TOOL.findall(c)]
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
