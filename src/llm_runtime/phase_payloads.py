from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PayloadViolation:
    error_id: str
    reason: str


@dataclass(frozen=True)
class Message:
    role: str
    content: Any


@dataclass(frozen=True)
class RouterPayload:
    history: list[Message]
    summary: str
    user_message: str


@dataclass(frozen=True)
class NarratorPayload:
    history: list[Message]
    summary: str
    latest_tool_result: dict[str, Any]


def validate_router_payload(payload: RouterPayload) -> list[PayloadViolation]:
    violations: list[PayloadViolation] = []
    if not isinstance(payload.summary, str) or not payload.summary.strip():
        violations.append(PayloadViolation("INV_HISTORY_SUMMARY_REQUIRED", "summary required"))
    if not payload.history:
        violations.append(PayloadViolation("ROUTER_HISTORY_REQUIRED", "history required"))
    if not isinstance(payload.user_message, str) or not payload.user_message.strip():
        violations.append(PayloadViolation("ROUTER_USER_MESSAGE_REQUIRED", "user_message required"))
    return violations


def validate_narrator_payload(payload: NarratorPayload) -> list[PayloadViolation]:
    violations: list[PayloadViolation] = []
    if not isinstance(payload.summary, str) or not payload.summary.strip():
        violations.append(PayloadViolation("INV_HISTORY_SUMMARY_REQUIRED", "summary required"))
    if not payload.history:
        violations.append(PayloadViolation("NARRATOR_HISTORY_REQUIRED", "history required"))
    elif payload.history[-1].role != "tool":
        violations.append(PayloadViolation("INV_NARRATOR_REQUIRES_TOOL", "latest role must be tool"))
    if not isinstance(payload.latest_tool_result, dict) or not payload.latest_tool_result:
        violations.append(PayloadViolation("NARRATOR_TOOL_RESULT_REQUIRED", "latest_tool_result required"))
    return violations
