from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..validation.admission import evaluate_record
from ..validation.replay import ToolBackend

CATEGORIES = (
    "mode_violation",
    "ambiguity_handling",
    "illegal_invalid",
    "tool_failure",
    "adversarial_routing",
    "injection_style",
    "tone_quality",
)


@dataclass(frozen=True)
class RedTeamProbe:
    probe_id: str
    category: str
    record: dict


@dataclass(frozen=True)
class RedTeamFinding:
    probe_id: str
    category: str
    passed: bool
    reasons: list[str]


@dataclass(frozen=True)
class RedTeamReport:
    findings: list[RedTeamFinding]


def run_redteam(probes: list[RedTeamProbe], backend: ToolBackend) -> RedTeamReport:
    findings: list[RedTeamFinding] = []
    for probe in probes:
        decision = evaluate_record(probe.record, backend)
        findings.append(
            RedTeamFinding(
                probe_id=probe.probe_id,
                category=probe.category,
                passed=decision.accepted,
                reasons=decision.reasons,
            )
        )
    return RedTeamReport(findings=findings)


def category_stats(report: RedTeamReport) -> dict[str, dict[str, int | float]]:
    stats: dict[str, dict[str, int | float]] = {
        category: {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        for category in CATEGORIES
    }
    for finding in report.findings:
        bucket = stats.setdefault(
            finding.category,
            {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
        )
        bucket["total"] += 1
        if finding.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    for bucket in stats.values():
        total = int(bucket["total"])
        bucket["pass_rate"] = (int(bucket["passed"]) / total) if total else 0.0
    return stats


def find_holes(report: RedTeamReport, threshold: float = 0.95) -> dict[str, list[RedTeamFinding]]:
    holes: dict[str, list[RedTeamFinding]] = {}
    grouped: dict[str, list[RedTeamFinding]] = {}
    for finding in report.findings:
        grouped.setdefault(finding.category, []).append(finding)
    for category, findings in grouped.items():
        passed = sum(1 for f in findings if f.passed)
        rate = passed / len(findings) if findings else 0.0
        if rate < threshold:
            holes[category] = [f for f in findings if not f.passed]
    return holes
