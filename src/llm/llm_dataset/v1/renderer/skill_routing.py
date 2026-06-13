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
from ..domains import REAL_DOMAINS, Domain
from .thinking import think, think_answer

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
_NORMALIZED = "normalized: vague phrasing resolved to an explicit request; routing to the fitting skill."

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
        {"name": "load_skill", "description": "Load a listed skill's body.",
         "args": {"name": "required"}, "applies_when": "always",
         "plugin": "chess-official", "source": "official_plugin", "enabled": True},
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


def render_skill_routing_row(domain: Domain, seed: int, style: str, normalize: bool) -> dict[str, Any]:
    rng = random.Random(seed)
    prompt = _style(rng.choice(domain.prompts), style)
    goal = f"route this {domain.skill.replace('-', ' ')} request"
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    selected = [domain.skill]
    if normalize:
        selected = ["hood-human-chat", domain.skill]
        messages += [
            {"role": "assistant", "content": f"{think(seed, 'load_skill', 0, goal=goal, have='')}\n{rng.choice(_LEAD_HOOD)}\n<tool>load_skill name=hood-human-chat</tool>"},
            {"role": "tool", "content": _HOOD_BODY},
            {"role": "assistant", "content": f"{think(seed, 'normalize_human_chat', 1, goal=goal, have='skill')}\n{rng.choice(_LEAD_NORMALIZE)}\n<tool>normalize_human_chat text=messy_user_chat</tool>"},
            {"role": "tool", "content": _NORMALIZED},
        ]
    messages += [
        {"role": "assistant", "content": f"{think(seed, 'load_skill', 2, goal=goal, have='')}\n{rng.choice(_LEAD_LOAD)}\n<tool>load_skill name={domain.skill}</tool>"},
        {"role": "tool", "content": domain.body},
        {"role": "assistant", "content": f"{think(seed, domain.tool, 3, goal=goal, have='skill')}\n{rng.choice(_LEAD_TOOL)}\n<tool>{domain.tool} {domain.call}</tool>"},
        {"role": "tool", "content": domain.tool_result},
        {"role": "assistant", "content": f"{think_answer(seed, goal)}\n{domain.answer}"},
    ]
    return _envelope(domain, seed, style, selected, messages, _index(domain, rng), _manifest(domain, rng, normalize))


def _envelope(domain: Domain, seed: int, style: str, selected: list[str],
              messages: list[dict[str, str]], skills_index: list[dict],
              tool_manifest: list[dict]) -> dict[str, Any]:
    expected = [m for content in (msg["content"] for msg in messages if msg["role"] == "assistant")
                for m in _TOOL.findall(content)]
    return {
        "id": f"v1_o_{domain.skill}_{seed:09d}",
        "slice": "V1_O_cross_domain_skill_routing",
        "kind": "skill_routing",
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
