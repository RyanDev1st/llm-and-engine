from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .annotator import DEFAULT_SF, StockfishAnnotator
from .dedup import drop_near_duplicates
from .paths import OUT
from .renderer.chess import render_chess_row
from .renderer.universality import render_universality_row
from .sampler import CHESS_SLICES, UNIVERSALITY_SLICES, plan_scenarios
from .validate import validate_row

DEFAULT_PLAN: dict[str, int] = {
    "A": 612, "B": 374, "C": 272, "D": 306, "E": 340, "F": 306,
    "G": 136, "H": 204, "I": 408, "J": 272, "K": 170,
    "V1_A_skill_index_selection": 60,
    "V1_B_skill_conflict_and_absence": 60,
    "V1_C_dynamic_tool_schema": 60,
    "V1_D_tool_unavailable_and_readonly": 60,
    "V1_E_board_grounding": 60,
    "V1_F_special_chess_rules": 60,
    "V1_G_multi_tool_budget": 60,
    "V1_H_error_recovery": 60,
    "V1_I_eval_language": 60,
    "V1_J_no_tool_and_mixed_intent": 60,
    "V1_K_adversarial_injection": 60,
    "V1_L_rejects_and_audit_fixtures": 60,
}


def run(plan: dict[str, int], seed: int, out: Path = OUT) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    scenarios = plan_scenarios(plan, seed=seed)
    annotator: StockfishAnnotator | None = (
        StockfishAnnotator() if os.path.exists(DEFAULT_SF) else None
    )
    accepted: list[dict] = []
    rejected: list[dict] = []
    for scenario in scenarios:
        if scenario.slice in CHESS_SLICES and annotator is not None:
            row = render_chess_row(scenario, annotator)
        elif scenario.slice in UNIVERSALITY_SLICES:
            row = render_universality_row(scenario)
        else:
            continue
        errs = validate_row(row)
        if errs:
            rejected.append({**row, "reject_reason": f"validator: {errs[0].rule}"})
            continue
        accepted.append(row)
    accepted = drop_near_duplicates(accepted)
    _write(out / "accepted.jsonl", accepted)
    _write(out / "rejected.jsonl", rejected)
    return len(accepted), len(rejected)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()
    ok, bad = run(DEFAULT_PLAN, seed=args.seed)
    print(f"wrote accepted={ok} rejected={bad}")


if __name__ == "__main__":
    main()
