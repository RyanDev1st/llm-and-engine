"""Plugin registry: bundles of TOOLS + SKILLS (+ optional prompt-start hooks).

Each plugin module exposes:
  NAME           : str
  TOOLS          : list[dict]              manifest entries (name/description/args/applies_when)
  SKILLS         : list[dict]              {name, description, body} skill specs
  handle(name, args, executor) -> str|None dispatch its own tools (None = not mine)
  prompt_start(context) -> str             optional additional-context injection (Anthropic-style)

The registry aggregates the ENABLED plugins so the served tool manifest + skill catalog
grow with whatever bundles are turned on — which is exactly what stresses the model's
routing: it must read names + descriptions across bundles and pick the right one. chess-
official is always on; the rest toggle via plugin_context["enabled"].
"""
from __future__ import annotations

from . import chess_official, openings, analysis, puzzles

# All installed plugins. chess-official first (its core tools lead the manifest).
REGISTRY = [chess_official, openings, analysis, puzzles]
ALWAYS_ON = {"chess-official"}


def _enabled(plugin_context: dict | None) -> set[str]:
    names = set((plugin_context or {}).get("enabled", [])) | ALWAYS_ON
    return names


def active(plugin_context: dict | None) -> list:
    en = _enabled(plugin_context)
    return [p for p in REGISTRY if getattr(p, "NAME", "") in en]


def plugin_tools(plugin_context: dict | None) -> list[dict]:
    """Tool manifest contributed by enabled plugins, EXCLUDING chess-official's (those
    come from the official catalog already). Tagged with their plugin for provenance."""
    out: list[dict] = []
    for p in active(plugin_context):
        if p.NAME == "chess-official":
            continue
        for t in getattr(p, "TOOLS", []):
            out.append({**t, "plugin": p.NAME, "source": "plugin", "enabled": True})
    return out


def plugin_skills(plugin_context: dict | None) -> list[dict]:
    """Skill catalog entries (name + description) contributed by enabled plugins."""
    out: list[dict] = []
    for p in active(plugin_context):
        for s in getattr(p, "SKILLS", []):
            out.append({"name": s["name"], "description": s["description"],
                        "plugin": p.NAME, "source": "plugin", "enabled": True})
    return out


def skill_body(name: str, plugin_context: dict | None) -> str | None:
    """The SKILL.md body of a plugin-provided skill, for load_skill. None if not found."""
    for p in active(plugin_context):
        for s in getattr(p, "SKILLS", []):
            if s["name"] == name:
                return s["body"]
    return None


def dispatch(name: str, args: dict, executor, plugin_context: dict | None) -> str | None:
    """Route a tool call to the enabled plugin that owns it. None if no plugin handles it."""
    for p in active(plugin_context):
        handler = getattr(p, "handle", None)
        if handler is None:
            continue
        res = handler(name, args, executor)
        if res is not None:
            return res
    return None


def prompt_start(context: dict, plugin_context: dict | None = None) -> str:
    """Run every enabled plugin's prompt-start hook and join the injections."""
    parts: list[str] = []
    for p in active(plugin_context):
        hook = getattr(p, "prompt_start", None)
        if hook is None:
            continue
        text = hook(context)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def catalog() -> list[dict]:
    """All installed plugins (for the Plugins panel): name + the skills/tools they bundle."""
    return [{"name": p.NAME, "always_on": p.NAME in ALWAYS_ON,
             "tools": [t["name"] for t in getattr(p, "TOOLS", [])],
             "skills": [s["name"] for s in getattr(p, "SKILLS", [])]}
            for p in REGISTRY]
