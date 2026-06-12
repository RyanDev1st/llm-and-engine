"""Plugin registry + prompt-start hooks (Anthropic-style additional-context injection).

A plugin bundles tools + skills and may register a `prompt_start(context)` hook that
returns text injected into the system prompt at the start of every turn — so a plugin's
always-on context (its default skill body, live state) is present WITHOUT the model
spending tool calls to fetch it. Versatile: any future plugin registers its own hook and
its injection is appended the same way.
"""
from __future__ import annotations

from . import chess_official

# Registered plugins, in injection order. Add a module here to register it.
REGISTRY = [chess_official]


def prompt_start(context: dict) -> str:
    """Run every registered plugin's prompt-start hook and join the injections.
    `context` carries per-turn state (e.g. {"game": <Game>}). Empty string when no
    plugin injects anything."""
    parts: list[str] = []
    for plugin in REGISTRY:
        hook = getattr(plugin, "prompt_start", None)
        if hook is None:
            continue
        text = hook(context)
        if text:
            parts.append(text)
    return "\n\n".join(parts)
