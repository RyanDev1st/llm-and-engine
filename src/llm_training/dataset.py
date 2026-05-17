from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetViolation:
    error_id: str
    reason: str


REQUIRED_FIELDS = {"id", "phase", "summary", "history", "input", "target", "metadata"}
PHASES = {"router", "narrator"}


def validate_record_shape(record: dict[str, Any]) -> list[DatasetViolation]:
    violations: list[DatasetViolation] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        violations.append(DatasetViolation("DATASET_REQUIRED_FIELDS", f"missing fields: {sorted(missing)}"))
        return violations
    if record.get("phase") not in PHASES:
        violations.append(DatasetViolation("DATASET_PHASE_INVALID", "phase must be router or narrator"))
    if not isinstance(record.get("summary"), str) or not record["summary"].strip():
        violations.append(DatasetViolation("DATASET_SUMMARY_REQUIRED", "summary required"))
    if not isinstance(record.get("history"), list) or not record["history"]:
        violations.append(DatasetViolation("DATASET_HISTORY_REQUIRED", "history required"))
    if not isinstance(record.get("input"), dict):
        violations.append(DatasetViolation("DATASET_INPUT_INVALID", "input must be object"))
    if not isinstance(record.get("target"), dict):
        violations.append(DatasetViolation("DATASET_TARGET_INVALID", "target must be object"))
    if not isinstance(record.get("metadata"), dict):
        violations.append(DatasetViolation("DATASET_METADATA_INVALID", "metadata must be object"))
    return violations
