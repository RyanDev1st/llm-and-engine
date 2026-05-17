from __future__ import annotations

from dataclasses import dataclass

from ..validation.admission import AdmissionReport
from ..validation.hygiene import find_near_duplicates
from ..validation.redteam import RedTeamReport, category_stats


@dataclass(frozen=True)
class FreezeDecision:
    approved: bool
    reasons: list[str]


@dataclass(frozen=True)
class AuditSummary:
    total_records: int
    slice_counts: dict[str, int]
    hard_fail_count: int
    replay_fail_count: int
    duplicate_count: int
    redteam_stats: dict[str, dict[str, int | float]]
    freeze: FreezeDecision


def count_slices(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        name = str(record.get("slice", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def build_audit_summary(
    records: list[dict],
    admission: AdmissionReport,
    redteam: RedTeamReport,
    replay_fail_count: int,
    duplicate_threshold: float = 0.85,
) -> AuditSummary:
    duplicates = find_near_duplicates(records, threshold=duplicate_threshold)
    red_stats = category_stats(redteam)

    reasons: list[str] = []
    hard_fail_count = len(admission.rejected)
    if hard_fail_count != 0:
        reasons.append(f"hard_fail_count={hard_fail_count}")
    if replay_fail_count != 0:
        reasons.append(f"replay_fail_count={replay_fail_count}")

    for category, bucket in red_stats.items():
        if float(bucket["pass_rate"]) < 0.95:
            reasons.append(f"redteam_{category}_pass_rate={bucket['pass_rate']:.3f}")

    approved = len(reasons) == 0
    return AuditSummary(
        total_records=len(records),
        slice_counts=count_slices(records),
        hard_fail_count=hard_fail_count,
        replay_fail_count=replay_fail_count,
        duplicate_count=len(duplicates),
        redteam_stats=red_stats,
        freeze=FreezeDecision(approved=approved, reasons=reasons),
    )
