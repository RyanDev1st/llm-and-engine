"""Single emission point for the harness ACTION tags, so the on-the-wire format
lives in ONE place instead of ~40 literal strings scattered across the renderers.

Phase 1 reproduces the CURRENT custom-XML format byte-for-byte (verified by a
fixed-seed regen diff). The native-Gemma swap (Phase 2) changes ONLY the bodies of
these two functions — every call site stays the same. All renderer modules call
these instead of writing `<tool>`/`<skill>` literals.
"""
from __future__ import annotations


def tool_call(name: str, args: str = "") -> str:
    """A tool call. `args` is the already-formatted `k=v k=v` string (or '')."""
    return f"<tool>{name} {args}</tool>" if args else f"<tool>{name}</tool>"


def skill_load(name: str) -> str:
    """A skill load (progressive disclosure)."""
    return f"<skill>{name}</skill>"
