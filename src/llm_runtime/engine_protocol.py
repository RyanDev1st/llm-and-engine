from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ToolContext:
    conversation_id: str = "default"
    state: dict = field(default_factory=dict)


class ToolBackend(Protocol):
    def execute(self, tool: str, args: dict, context: ToolContext) -> dict:
        ...
