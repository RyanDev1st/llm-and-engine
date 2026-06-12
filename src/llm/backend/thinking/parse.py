"""Parse a Controller turn into an action: a tool call or DONE.

Reuses inference.extract_call so the Controller benefits from the same recovery as
the single loop (<tool_code> normalize, malformed wrapper, hint-echo, stop-trim)."""
from __future__ import annotations

from ..inference import extract_call


def parse_controller(raw: str) -> tuple[str, str | None]:
    s = (raw or "").strip()
    if not s:
        return ("done", None)
    call = extract_call(s)                 # canonical <tool>…</tool> (recovered) or None
    if call is not None and "<tool>" in call:
        return ("tool", call)
    if s.upper().startswith("DONE"):
        return ("done", None)
    return ("done", None)                  # neither -> fail toward narrating
