from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.contract import TurnContract
from ..validation.replay import ReplayFailure, ToolBackend, replay_validate
from ..validation.routing_sanity import RoutingSanityFailure, check_routing_sanity


@dataclass(frozen=True)
class RecordDecision:
    record_id: str
    accepted: bool
    reasons: list[str]


@dataclass(frozen=True)
class AdmissionReport:
    accepted: list[dict[str, Any]]
    rejected: list[RecordDecision]


def evaluate_record(record: dict[str, Any], backend: ToolBackend) -> RecordDecision:
    reasons: list[str] = []

    contract_violations = TurnContract.validate_record(record)
    if contract_violations:
        reasons.extend([f"{v.rule_id}: {v.reason}" for v in contract_violations])

    messages = record.get("messages", [])
    if isinstance(messages, list):
        replay_failures = replay_validate(messages, backend)
        reasons.extend([_fmt_replay_failure(item) for item in replay_failures])

        routing_failures = check_routing_sanity(str(record.get("slice", "")), messages)
        reasons.extend([_fmt_routing_failure(item) for item in routing_failures])

    return RecordDecision(
        record_id=str(record.get("id", "unknown")),
        accepted=len(reasons) == 0,
        reasons=reasons,
    )


def admit_records(records: list[dict[str, Any]], backend: ToolBackend) -> AdmissionReport:
    accepted: list[dict[str, Any]] = []
    rejected: list[RecordDecision] = []
    for record in records:
        decision = evaluate_record(record, backend)
        if decision.accepted:
            accepted.append(record)
        else:
            rejected.append(decision)
    return AdmissionReport(accepted=accepted, rejected=rejected)


def _fmt_replay_failure(item: ReplayFailure) -> str:
    return f"V3_REPLAY turn={item.turn_index} tool={item.tool_name}: {item.reason}"


def _fmt_routing_failure(item: RoutingSanityFailure) -> str:
    return f"{item.rule_id}: {item.reason}"
