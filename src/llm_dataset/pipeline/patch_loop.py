from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..validation.redteam import RedTeamFinding


@dataclass(frozen=True)
class PatchRequest:
    category: str
    record_ids: list[str]
    reason: str


@dataclass(frozen=True)
class PatchResult:
    category: str
    replaced_ids: list[str]
    new_records: list[dict]


Regenerator = Callable[[PatchRequest], PatchResult]


def build_patch_requests(failures_by_category: dict[str, list[RedTeamFinding]]) -> list[PatchRequest]:
    requests: list[PatchRequest] = []
    for category, findings in failures_by_category.items():
        ids = [f.probe_id for f in findings]
        reason = findings[0].reasons[0] if findings and findings[0].reasons else "red-team failure"
        requests.append(PatchRequest(category=category, record_ids=ids, reason=reason))
    return requests


def apply_patch_requests(existing_records: list[dict], requests: list[PatchRequest], regenerator: Regenerator) -> tuple[list[dict], list[PatchResult]]:
    records = {str(item.get("id", "")): item for item in existing_records}
    results: list[PatchResult] = []

    for request in requests:
        outcome = regenerator(request)
        results.append(outcome)
        for rid in outcome.replaced_ids:
            if rid in records:
                del records[rid]
        for row in outcome.new_records:
            records[str(row.get("id", ""))] = row

    return list(records.values()), results
