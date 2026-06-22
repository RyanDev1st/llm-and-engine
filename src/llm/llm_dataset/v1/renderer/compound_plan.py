"""Stage 1 — V1_S_compound_plan: goal-driven completion across MULTIPLE skills.

Teaches the anti-early-stop loop: a request that needs two different-domain skills to
fully answer. The model commits the <goal>, lists the needed steps as a <plan>, then
DOES EVERY box (load skill -> call its tool -> read result) before synthesizing across
both findings. Closes the 3+ distinct-skill composition gap (handoff §2: currently 0%).

Seq is tight (plan-mode floor + two skill+tool chains), so this slice is deliberately
terse: 2 domains, ONE-LINE skill bodies (V1_O already teaches full bodies; here the
lesson is composition + not-stopping-early, not re-teaching bodies). ~12% of rows show
the honest-partial abort: a needed skill is disabled, so the model clears what it can and
reports the blocker instead of faking the box (loop-cap discipline).
"""
from __future__ import annotations

import random
from typing import Any

from ..domains import REAL_DOMAINS, Domain
from .planning import goal_block, partial_report, plan_block
from .thinking import gated_think, pick_mode

_LEAD_LOAD = ("Loading the skill for this part.", "Next skill for the next box.",
              "Pulling the skill this box needs.")
_LEAD_TOOL = ("Now its data.", "Running its tool.", "Getting the specifics.")


def _terse_body(d: Domain) -> str:
    """One-line skill body for seq economy — names the tool the box must call."""
    return f"# {d.skill}\nUse {d.tool} for this, then report the finding."


def _two_domains(seed: int) -> tuple[Domain, Domain]:
    r = random.Random(seed * 2654435761 % (2 ** 32))
    a, b = r.sample(REAL_DOMAINS, 2)
    return a, b


def _compound_prompt(r: random.Random, a: Domain, b: Domain) -> str:
    pa, pb = r.choice(a.prompts), r.choice(b.prompts)
    return r.choice((f"{pa}, and also {pb}", f"two things: {pa}; then {pb}",
                     f"{pa} — and {pb} while you're at it"))


def _index(a: Domain, b: Domain, rng: random.Random, partial: bool) -> list[dict]:
    """Both needed skills + 2 distractors. If partial, skill b is disabled (the blocker)."""
    entries = [
        {"name": a.skill, "description": a.description, "plugin": a.plugin,
         "source": a.source, "enabled": True},
        {"name": b.skill, "description": b.description, "plugin": b.plugin,
         "source": b.source, "enabled": not partial},
    ]
    others = [d for d in REAL_DOMAINS if d.skill not in (a.skill, b.skill)]
    for d in rng.sample(others, min(2, len(others))):
        entries.append({"name": d.skill, "description": d.description,
                        "plugin": "market-tactics", "source": "marketplace_plugin", "enabled": True})
    rng.shuffle(entries)
    return entries


def _manifest(a: Domain, b: Domain, rng: random.Random) -> list[dict]:
    tools = []
    for d in (a, b):
        tools.append({"name": d.tool, "description": f"Domain tool for {d.skill}.",
                      "args": d.tool_args, "applies_when": "always",
                      "plugin": "user-skills", "source": "user_skill", "enabled": True})
    rng.shuffle(tools)
    return tools


def _box_steps(seed: int, d: Domain, step0: int, goal: str, mode: str) -> tuple[list[dict], str]:
    """One box = load skill -> call its tool. Returns (messages, finding)."""
    call, tool_result, finding = random.Random(seed * 17 + step0).choice(d.scenes)
    msgs = [
        {"role": "assistant", "content": _join(
            gated_think(seed, "load_skill", step0, mode=mode, kind="select", goal=goal),
            _pick(seed, step0, _LEAD_LOAD), f"<skill>{d.skill}</skill>")},
        {"role": "tool", "content": _terse_body(d)},
        {"role": "assistant", "content": _join(
            gated_think(seed, d.tool, step0 + 1, mode=mode, kind="execute", goal=goal, have="skill"),
            _pick(seed, step0 + 1, _LEAD_TOOL), f"<tool>{d.tool} {call}</tool>")},
        {"role": "tool", "content": tool_result},
    ]
    return msgs, finding


def _pick(seed: int, step: int, pool: tuple[str, ...]) -> str:
    return random.Random(seed * 31 + step).choice(pool)


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def render_compound_plan_row(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    a, b = _two_domains(seed)
    partial = (seed % 8 == 0)                 # ~12% honest-partial (skill b disabled)
    mode = "plan"
    # BOTH goals are committed explicitly (compound request = two distinct asks), then
    # the plan covers each — "a planning mode that gets both goals and writes the plans".
    goal_a = f"the {a.skill.replace('-', ' ')} ask"
    goal_b = f"the {b.skill.replace('-', ' ')} ask"
    goal_text = f"{goal_a} and {goal_b}"     # for gated_think goal= (held intent)
    boxes = [(f"handle the {a.skill.replace('-', ' ')} part", a.skill),
             (f"handle the {b.skill.replace('-', ' ')} part", b.skill),
             ("synthesize one combined answer", "none")]

    messages: list[dict[str, str]] = [{"role": "user", "content": _compound_prompt(rng, a, b)}]
    # Plan panel: commit BOTH goals + author the checklist (one assistant turn -> panel).
    messages.append({"role": "assistant", "content": goal_block(seed, [goal_a, goal_b]) + "\n" + plan_block(boxes)})

    box_a, finding_a = _box_steps(seed, a, 1, goal_text, mode)
    messages += box_a
    if partial:
        # box b's skill is disabled -> can't complete; honest-partial, don't fake it.
        final = partial_report(boxes, 1, f"the {b.skill} skill is disabled in this manifest")
        messages.append({"role": "assistant", "content": final})
        selected = [a.skill]
    else:
        box_b, finding_b = _box_steps(seed, b, 3, goal_text, mode)
        messages += box_b
        final = f"On the first: {finding_a} On the second: {finding_b}"
        messages.append({"role": "assistant", "content": final})
        selected = [a.skill, b.skill]

    return _envelope(seed, messages, _index(a, b, rng, partial), _manifest(a, b, rng), selected, mode)


import re as _re
_TOOL = _re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")


def _envelope(seed: int, messages: list[dict], skills_index: list[dict],
              tool_manifest: list[dict], selected: list[str], mode: str) -> dict[str, Any]:
    expected = [m for c in (msg["content"] for msg in messages if msg["role"] == "assistant")
                for m in _TOOL.findall(c)]
    return {
        "id": f"v1_s_compound_{seed:09d}",
        "slice": "V1_S_compound_plan",
        "kind": "compound_plan",
        "reasoning_mode": mode,
        "intent": f"v1_s_{seed:06d}",
        "plugin_context": {"installed": ["user-skills", "market-tactics"],
                           "enabled": ["user-skills"], "marketplace": []},
        "skills_index": skills_index,
        "selected_skills": selected,
        "tool_manifest": tool_manifest,
        "expected_tool_calls": expected,
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "selected_skill_exists", "skill_body_strict", "plugin_only_tools",
                             "applies_when_respected", "goal_before_plan", "plan_boxes_bound"],
        "position_fen": None,
        "stockfish_truth": None,
    }
