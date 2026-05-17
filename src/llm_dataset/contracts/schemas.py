from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VALID_SLICES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}
VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class SchemaViolation:
    rule_id: str
    reason: str


def validate_record_shape(record: dict[str, Any]) -> list[SchemaViolation]:
    violations: list[SchemaViolation] = []
    required = ("id", "slice", "messages", "validated", "notes")
    for field in required:
        if field not in record:
            violations.append(SchemaViolation("V1_REQUIRED_FIELD", f"missing field: {field}"))
    if violations:
        return violations

    if not isinstance(record["id"], str) or not record["id"].strip():
        violations.append(SchemaViolation("V1_ID_TYPE", "id must be non-empty string"))
    if record["slice"] not in VALID_SLICES:
        violations.append(SchemaViolation("V1_SLICE_VALUE", "slice must be A..K"))
    if not isinstance(record["validated"], bool):
        violations.append(SchemaViolation("V1_VALIDATED_TYPE", "validated must be boolean"))
    if not isinstance(record["notes"], str):
        violations.append(SchemaViolation("V1_NOTES_TYPE", "notes must be string"))

    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        violations.append(SchemaViolation("V1_MESSAGES_TYPE", "messages must be non-empty list"))
        return violations

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            violations.append(SchemaViolation("V1_MESSAGE_OBJECT", f"message[{idx}] must be object"))
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES:
            violations.append(SchemaViolation("V1_ROLE_VALUE", f"message[{idx}].role invalid"))
        if not isinstance(content, str):
            violations.append(SchemaViolation("V1_CONTENT_TYPE", f"message[{idx}].content must be string"))
    return violations
