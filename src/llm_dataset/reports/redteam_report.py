from __future__ import annotations

from pathlib import Path

from ..validation.redteam import RedTeamReport, category_stats, find_holes


def write_redteam_report(path: Path, report: RedTeamReport, threshold: float = 0.95) -> Path:
    stats = category_stats(report)
    holes = find_holes(report, threshold=threshold)

    lines = [
        "# Dataset Red-Team Report",
        "",
        f"threshold: {threshold}",
        "",
        "## Category Stats",
    ]
    for category, bucket in stats.items():
        lines.append(
            f"- {category}: total={bucket['total']} passed={bucket['passed']} failed={bucket['failed']} pass_rate={bucket['pass_rate']:.3f}"
        )

    lines.append("")
    lines.append("## Holes")
    if not holes:
        lines.append("- none")
    for category, findings in holes.items():
        lines.append(f"- {category}")
        for finding in findings:
            lines.append(f"  - {finding.probe_id}")
            for reason in finding.reasons:
                lines.append(f"    - {reason}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
