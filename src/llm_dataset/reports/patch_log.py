from __future__ import annotations

from pathlib import Path

from ..pipeline.patch_loop import PatchResult


def write_patch_log(path: Path, results: list[PatchResult]) -> Path:
    lines = ["# Dataset Patch Log", ""]
    if not results:
        lines.append("- no patches applied")
    for result in results:
        lines.append(f"## {result.category}")
        lines.append(f"- replaced: {len(result.replaced_ids)}")
        lines.append(f"- added: {len(result.new_records)}")
        if result.replaced_ids:
            lines.append("- replaced_ids:")
            for item in result.replaced_ids:
                lines.append(f"  - {item}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
