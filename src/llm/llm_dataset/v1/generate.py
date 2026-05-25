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
    "A": 632, "B": 394, "C": 292, "D": 326, "E": 360, "F": 326,
    "G": 156, "H": 224, "I": 428, "J": 292, "K": 190,
    "V1_A_skill_index_selection": 70,
    "V1_B_skill_conflict_and_absence": 70,
    "V1_C_dynamic_tool_schema": 70,
    "V1_D_tool_unavailable_and_readonly": 70,
    "V1_E_board_grounding": 70,
    "V1_F_special_chess_rules": 70,
    "V1_G_multi_tool_budget": 70,
    "V1_H_error_recovery": 70,
    "V1_I_eval_language": 70,
    "V1_J_no_tool_and_mixed_intent": 70,
    "V1_K_adversarial_injection": 70,
    "V1_L_rejects_and_audit_fixtures": 70,
}


def run(plan: dict[str, int], seed: int, out: Path = OUT) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    scenarios = plan_scenarios(plan, seed=seed)
    annotator: StockfishAnnotator | None = (
        StockfishAnnotator() if os.path.exists(DEFAULT_SF) else None
    )
    accepted: list[dict] = []
    rejected: list[dict] = []
    try:
        for scenario in scenarios:
            if scenario.slice in CHESS_SLICES and annotator is not None:
                try:
                    row = render_chess_row(scenario, annotator)
                except Exception as exc:
                    rejected.append({
                        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
                        "slice": scenario.slice,
                        "kind": "harness_chess",
                        "intent": scenario.intent,
                        "reject_reason": f"annotator_error: {type(exc).__name__}: {exc}",
                    })
                    continue
            elif scenario.slice in UNIVERSALITY_SLICES:
                row = render_universality_row(scenario)
            else:
                continue
            errs = validate_row(row)
            if errs:
                rejected.append({**row, "reject_reason": f"validator: {errs[0].rule}"})
                continue
            accepted.append(row)
    finally:
        if annotator is not None:
            annotator.quit()
    accepted = drop_near_duplicates(accepted)
    rejected.extend(_audit_rejects(accepted, 800 - len(rejected)))
    _write(out / "accepted.jsonl", accepted)
    _write(out / "rejected.jsonl", rejected)
    return len(accepted), len(rejected)


def _audit_rejects(rows: list[dict], needed: int) -> list[dict]:
    if needed <= 0:
        return []
    rejects: list[dict] = []
    for idx, row in enumerate(rows):
        bad = {**row, "id": f"reject_{idx:05d}_{row['id']}", "reject_reason": "audit_fixture: undeclared_tool"}
        bad["messages"] = row["messages"][:-1] + [
            {"role": "assistant", "content": "<tool>undeclared_probe input=x</tool>"},
            row["messages"][-1],
        ]
        rejects.append(bad)
        if len(rejects) == needed:
            break
    return rejects


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
