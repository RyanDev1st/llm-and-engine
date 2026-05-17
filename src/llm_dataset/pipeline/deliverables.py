from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..validation.admission import AdmissionReport


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def split_train_val(records: list[dict[str, Any]], val_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []
    val_count = max(1, int(len(records) * val_ratio))
    train_count = max(0, len(records) - val_count)
    train = records[:train_count]
    val = records[train_count:]
    return train, val


def write_admission_summary(path: Path, report: AdmissionReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validation Summary",
        "",
        f"accepted: {len(report.accepted)}",
        f"rejected: {len(report.rejected)}",
        "",
        "## Rejections",
    ]
    for item in report.rejected:
        lines.append(f"- {item.record_id}")
        for reason in item.reasons:
            lines.append(f"  - {reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
