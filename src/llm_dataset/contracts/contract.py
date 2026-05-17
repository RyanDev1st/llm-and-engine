from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.schemas import validate_record_shape
from ..contracts.tool_grammar import is_exact_tool_call

TOOL_CALL_PREFIX = "<tool>"
TOOL_CALL_SUFFIX = "</tool>"


@dataclass(frozen=True)
class ContractViolation:
    rule_id: str
    turn_index: int
    reason: str


class TurnContract:
    """Canonical turn contract from spec v3 for dataset generation/validation."""

    @staticmethod
    def is_tool_call(content: str) -> bool:
        return is_exact_tool_call(content)

    @staticmethod
    def check_required_fields(record: dict[str, Any]) -> list[ContractViolation]:
        return [
            ContractViolation(rule_id=v.rule_id, turn_index=-1, reason=v.reason)
            for v in validate_record_shape(record)
        ]

    @staticmethod
    def check_role_order(messages: list[dict[str, str]]) -> list[ContractViolation]:
        violations: list[ContractViolation] = []
        if not messages:
            return [ContractViolation("V1_EMPTY_MESSAGES", -1, "messages is empty")]
        if messages[0].get("role") != "system":
            violations.append(
                ContractViolation("V1_SYSTEM_FIRST", 0, "first role must be system")
            )
        for i in range(1, len(messages)):
            prev_role = messages[i - 1].get("role")
            role = messages[i].get("role")
            if role == "assistant" and prev_role == "assistant":
                violations.append(
                    ContractViolation("V1_DOUBLE_ASSISTANT", i, "assistant follows assistant")
                )
            if role == "tool" and prev_role != "assistant":
                violations.append(
                    ContractViolation("V1_TOOL_SEQUENCE", i, "tool must follow assistant")
                )
        return violations

    @staticmethod
    def check_mode_discipline(messages: list[dict[str, str]]) -> list[ContractViolation]:
        violations: list[ContractViolation] = []
        for i in range(1, len(messages)):
            prev_role = messages[i - 1].get("role")
            role = messages[i].get("role")
            content = messages[i].get("content", "")
            if role != "assistant":
                continue
            has_tool_call = TurnContract.is_tool_call(content) or TOOL_CALL_PREFIX in content
            if prev_role == "tool" and has_tool_call:
                violations.append(
                    ContractViolation(
                        "V2_AFTER_TOOL_NO_TOOLCALL",
                        i,
                        "assistant after tool contains tool call",
                    )
                )
            if prev_role == "user" and TOOL_CALL_PREFIX in content and not TurnContract.is_tool_call(content):
                violations.append(
                    ContractViolation(
                        "V2_TOOLCALL_FORMAT",
                        i,
                        "tool call must be exact single payload",
                    )
                )
        return violations

    @staticmethod
    def validate_record(record: dict[str, Any]) -> list[ContractViolation]:
        violations = TurnContract.check_required_fields(record)
        messages = record.get("messages", [])
        if not isinstance(messages, list):
            return violations + [
                ContractViolation("V1_MESSAGES_TYPE", -1, "messages must be list")
            ]
        violations.extend(TurnContract.check_role_order(messages))
        violations.extend(TurnContract.check_mode_discipline(messages))
        return violations
