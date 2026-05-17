from __future__ import annotations

from pathlib import Path

from ..reports.audit import AuditSummary


def write_audit_report(path: Path, summary: AuditSummary) -> Path:
    lines = [
        "# Final Dataset Audit Report",
        "",
        f"total_records: {summary.total_records}",
        f"hard_fail_count: {summary.hard_fail_count}",
        f"replay_fail_count: {summary.replay_fail_count}",
        f"duplicate_count: {summary.duplicate_count}",
        "",
        "## Slice Counts",
    ]
    for name, count in sorted(summary.slice_counts.items()):
        lines.append(f"- {name}: {count}")

    lines.append("")
    lines.append("## Red-Team Stats")
    for category, bucket in summary.redteam_stats.items():
        lines.append(
            f"- {category}: total={bucket['total']} passed={bucket['passed']} failed={bucket['failed']} pass_rate={bucket['pass_rate']:.3f}"
        )

    lines.append("")
    lines.append("## Freeze Decision")
    lines.append(f"- approved: {summary.freeze.approved}")
    if summary.freeze.reasons:
        lines.append("- reasons:")
        for reason in summary.freeze.reasons:
            lines.append(f"  - {reason}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
