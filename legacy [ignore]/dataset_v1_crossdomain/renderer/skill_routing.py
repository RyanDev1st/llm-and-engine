"""Render one cross-domain skill-routing row: the user asks something in some
domain; the agent loads the skill whose DESCRIPTION fits (any domain, not
chess), reads its real body, calls the one tool the body names, and answers
with a guiding question. One tool per inference step; a short lead-in precedes
each call. `normalize=True` first loads hood-human-chat to clean messy chat —
two skills loaded across separate steps."""
from __future__ import annotations

import random
import re
from typing import Any

from ..catalog import HUMAN_CHAT_SKILL, OFFICIAL_SKILL
from ..domains import CLOSERS, REAL_DOMAINS, Domain
from .thinking import gated_answer, gated_think, pick_mode, prepend_open_goal

_TOOL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)", re.DOTALL)

_LEAD_LOAD = ("Let me pull up the right skill for this.", "First, the skill that fits this.",
              "I'll load the matching skill before acting.")
_LEAD_NORMALIZE = ("This wording is messy — let me clean it up first.",
                   "Let me normalize what you said before routing.")
_LEAD_HOOD = ("Loading the chat-cleanup skill first.", "Let me grab the message-normalizer.")
_LEAD_TOOL = ("Now the data I need.", "Let me check the specifics.", "Pulling the details now.")

_HOOD_BODY = ("# hood-human-chat\nWhen to use: the message is slang, shorthand, or vague.\n"
              "Steps:\n1. Restate the message as explicit intent.\n2. Keep the domain signal.\n"
              "Constraint: ask when slang stays ambiguous.")
# Varied normalize results — reflects the messy->intent translation per domain, no
# facts. One fixed string repeated thousands of times was a memorization smell.
_NORMALIZED_POOL = (
    "normalized: shorthand resolved to an explicit {d} request.",
    "normalized: slang cleaned up — the user wants {d} help.",
    "normalized: vague wording mapped to a clear {d} ask.",
    "normalized: typo-filled message restated as a {d} request.",
    "normalized: messy phrasing resolved; routing to {d}.",
    "normalized: intent is clear now — a {d} task.",
)


def _normalized(domain: Domain, rng: random.Random) -> str:
    return rng.choice(_NORMALIZED_POOL).format(d=domain.skill.replace("-", " "))

_RULES = ["final_no_xml", "known_tool_only", "args_match_schema", "selected_skill_exists",
          "skill_index_only_before_load", "skill_body_strict", "plugin_only_tools",
          "applies_when_respected"]


def _style(base: str, style: str) -> str:
    if style == "formal":
        return f"Please {base}."
    if style == "slang":
        return f"yo {base}"
    if style == "typo":
        return f"{base} plz"
    if style == "anxious":
        return f"not sure about this — {base}"
    if style == "beginner":
        return f"i'm new to this; {base}"
    return base


def _index(domain: Domain, rng: random.Random) -> list[dict]:
    entries = [
        {"name": domain.skill, "description": domain.description,
         "plugin": domain.plugin, "source": domain.source, "enabled": True},
        {"name": OFFICIAL_SKILL["name"], "description": OFFICIAL_SKILL["description"],
         "plugin": "chess-official", "source": "official_plugin", "enabled": True},
        {"name": HUMAN_CHAT_SKILL["name"], "description": HUMAN_CHAT_SKILL["description"],
         "plugin": "user-skills", "source": "user_skill", "enabled": True},
    ]
    others = [d for d in REAL_DOMAINS if d.skill != domain.skill]
    for d in rng.sample(others, min(3, len(others))):
        enabled = d.skill not in {domain.skill}
        entries.append({"name": d.skill, "description": d.description,
                        "plugin": "market-tactics", "source": "marketplace_plugin", "enabled": enabled})
    rng.shuffle(entries)
    return entries


def _manifest(domain: Domain, rng: random.Random, normalize: bool) -> list[dict]:
    tools = [
        {"name": domain.tool, "description": f"Domain tool for {domain.skill}.",
         "args": domain.tool_args, "applies_when": "always",
         "plugin": "user-skills", "source": "user_skill", "enabled": True},
        {"name": "board_state", "description": "Snapshot the live board.",
         "args": {"fields": ["basic", "all", "fen"]}, "applies_when": "always",
         "plugin": "chess-official", "source": "official_plugin", "enabled": True},
    ]
    if normalize:
        tools.append({"name": "normalize_human_chat", "description": "Translate messy chat to intent.",
                      "args": {"text": "required"}, "applies_when": "always",
                      "plugin": "user-skills", "source": "user_skill", "enabled": True})
    rng.shuffle(tools)
    return tools


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def render_skill_routing_row(domain: Domain, seed: int, style: str, normalize: bool) -> dict[str, Any]:
    rng = random.Random(seed)
    prompt = _style(rng.choice(domain.prompts), style)
    # Pick one of the domain's scenes (varied call/result/finding) and a guiding
    # closer, so the same skill yields many distinct grounded finals, not one canned
    # line repeated hundreds of times. The finding states the result; the closer
    # offers a next step (no facts -> stays grounded).
    call, tool_result, finding = rng.choice(domain.scenes)
    answer = f"{finding} {rng.choice(CLOSERS)}"
    goal = f"route this {domain.skill.replace('-', ' ')} request"
    mode = pick_mode(seed)
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    selected = [domain.skill]
    if normalize:
        selected = ["hood-human-chat", domain.skill]
        messages += [
            {"role": "assistant", "content": _join(gated_think(seed, "load_skill", 0, mode=mode, kind="select", goal=goal), rng.choice(_LEAD_HOOD), "<skill>hood-human-chat</skill>")},
            {"role": "tool", "content": _HOOD_BODY},
            {"role": "assistant", "content": _join(gated_think(seed, "normalize_human_chat", 1, mode=mode, kind="decide", goal=goal, have="skill"), rng.choice(_LEAD_NORMALIZE), "<tool>normalize_human_chat text=messy_user_chat</tool>")},
            {"role": "tool", "content": _normalized(domain, rng)},
        ]
    messages += [
        {"role": "assistant", "content": _join(gated_think(seed, "load_skill", 2, mode=mode, kind="select", goal=goal), rng.choice(_LEAD_LOAD), f"<skill>{domain.skill}</skill>")},
        {"role": "tool", "content": domain.body},
        {"role": "assistant", "content": _join(gated_think(seed, domain.tool, 3, mode=mode, kind="routine", goal=goal, have="skill"), rng.choice(_LEAD_TOOL), f"<tool>{domain.tool} {call}</tool>")},
        {"role": "tool", "content": tool_result},
        {"role": "assistant", "content": _join(gated_answer(seed, goal, mode=mode), answer)},
    ]
    prepend_open_goal(messages, seed, mode, goal)   # lead with <goal> in thinking modes
    return _envelope(domain, seed, style, selected, messages, _index(domain, rng), _manifest(domain, rng, normalize), mode)


def _envelope(domain: Domain, seed: int, style: str, selected: list[str],
              messages: list[dict[str, str]], skills_index: list[dict],
              tool_manifest: list[dict], mode: str = "think") -> dict[str, Any]:
    expected = [m for content in (msg["content"] for msg in messages if msg["role"] == "assistant")
                for m in _TOOL.findall(content)]
    return {
        "id": f"v1_o_{domain.skill}_{seed:09d}",
        "slice": "V1_O_cross_domain_skill_routing",
        "kind": "skill_routing",
        "reasoning_mode": mode,
        "intent": f"v1_o_{seed:06d}",
        "plugin_context": {
            "installed": ["chess-official", "user-skills", "market-tactics", "synthetic-pack"],
            "enabled": ["chess-official", "user-skills", "synthetic-pack"],
            "marketplace": ["market-openings", "market-endgames"],
        },
        "skills_index": skills_index,
        "selected_skills": selected,
        "tool_manifest": tool_manifest,
        "expected_tool_calls": expected,
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": list(_RULES),
        "position_fen": None,
        "stockfish_truth": None,
    }
