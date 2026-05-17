from __future__ import annotations

import re
from dataclasses import dataclass

TOOL_CALL_RE = re.compile(r"^<tool>([a-z_]+)(?:\s+[a-z_]+=[^\s<>]+)*</tool>$")


@dataclass(frozen=True)
class ToolCallParse:
    raw: str
    tool_name: str


def is_exact_tool_call(content: str) -> bool:
    return bool(TOOL_CALL_RE.match(content.strip()))


def parse_tool_name(content: str) -> str | None:
    stripped = content.strip()
    match = TOOL_CALL_RE.match(stripped)
    if not match:
        return None
    return match.group(1)
