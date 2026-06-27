from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from .annotator import DEFAULT_SF, StockfishAnnotator
from .dedup import drop_near_duplicates
from .jsonl_io import write_rows
from .paths import OUT
from .profiles import DatasetProfile, profile
from .renderer.chess import render_chess_row
from .renderer.audited_plan import render_audited_plan_row
from .renderer.compound_plan import render_compound_plan_row
from .renderer.compute import render_compute_row
from .renderer.multiturn import render_multiturn_row
from .renderer.specialist_routing import render_specialist_routing_row
from .sampler import (
    AUDIT_SLICES, CHESS_SLICES, COMPOUND_SLICES, COMPUTE_SLICES, MULTITURN_SLICE,
    plan_scenarios,
)
from .validate import validate_row

# PURE-CHESS mix (v5): the product is a chess coach. The grounded-answer slices —
# D eval, E best-move, F review, G threats — are UP-WEIGHTED because concretely
# ANSWERING (the grounded "why") is the product, not just routing; v1-v4 over-fit
# routing and under-fit answering, which fed serve-time confabulation. The keystones
# (python-verify / compound-plan / audited-plan, refocused to chess) and a moderate
# specialist-routing slice are added back in later steps. Proportions are preserved
# through plan_for_profile scaling.
DEFAULT_PLAN: dict[str, int] = {
    "A": 120,   # play a named move
    "B": 110,   # decide between options
    "C": 90,    # refuse an illegal move
    "D": 150,   # evaluate the position
    "E": 200,   # best move (grounded "why")
    "F": 200,   # review the move played (grounded verdict)
    "G": 150,   # opponent threats (grounded)
    "H": 80,    # list pieces
    "I": 130,   # chess knowledge (ask_chessbot)
    "J": 90,    # greeting / capabilities
    "K": 90,    # general chess Q&A
    "V1_P_multiturn_followup": 200,   # multi-turn dialogue state (chess)
    # keystones, refocused to chess: verify-by-running-python, multi-skill plan, audited plan.
    "V1_R_compute_grounding": 80,     # verify a chess number via the python tool, don't assert
    "V1_S_compound_plan": 70,         # two-specialist chess plan (review + opening), anti-early-stop
    "V1_T_audited_plan": 70,          # chess plan with python-audited boxes + split-determinism
    "V1_U_specialist_routing": 90,    # pick the right specialist by intent (moderate hardening)
}


def plan_for_profile(dataset_profile: DatasetProfile, tiny: bool = False) -> dict[str, int]:
    if tiny:
        return {key: 2 for key in DEFAULT_PLAN}
    if dataset_profile.name == "v1.1":
        return DEFAULT_PLAN
    scale = dataset_profile.accepted_target / sum(DEFAULT_PLAN.values())
    plan = {key: max(60, round(value * scale)) for key, value in DEFAULT_PLAN.items()}
    # Per-slice rounding can land just under the target (e.g. 49,990 < 50,000).
    # Top up the largest slice so accepted always clears accepted_target, plus a
    # small buffer in case a rare annotator error drops a row. Routing/universality
    # slices have only a lower bound in audit, so the headroom is harmless.
    shortfall = max(0, dataset_profile.accepted_target - sum(plan.values()))
    biggest = max(plan, key=plan.get)
    plan[biggest] += shortfall + 60
    return plan


def run(
    plan: dict[str, int],
    seed: int,
    out: Path = OUT,
    rejected_target: int = 800,
    progress: Callable[[int, int, int, int], None] | None = None,
    stage_progress: Callable[[str], None] | None = None,
    near_dedup_limit: int = 10_000,
) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    scenarios = plan_scenarios(plan, seed=seed)
    total = len(scenarios)
    annotator: StockfishAnnotator | None = (
        StockfishAnnotator() if os.path.exists(DEFAULT_SF) else None
    )
    accepted: list[dict] = []
    rejected: list[dict] = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
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
                    if progress and _should_report(index, total):
                        progress(index, total, len(accepted), len(rejected))
                    continue
            elif scenario.slice == MULTITURN_SLICE and annotator is not None:
                try:
                    row = render_multiturn_row(scenario, annotator)
                except Exception as exc:
                    rejected.append({
                        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
                        "slice": scenario.slice,
                        "kind": "harness_chess",
                        "intent": scenario.intent,
                        "reject_reason": f"annotator_error: {type(exc).__name__}: {exc}",
                    })
                    if progress and _should_report(index, total):
                        progress(index, total, len(accepted), len(rejected))
                    continue
            elif scenario.slice == "V1_U_specialist_routing":
                row = render_specialist_routing_row(scenario.seed)
            elif scenario.slice in COMPUTE_SLICES:
                row = render_compute_row(scenario)
            elif scenario.slice in COMPOUND_SLICES:
                row = render_compound_plan_row(scenario.seed)
            elif scenario.slice in AUDIT_SLICES:
                row = render_audited_plan_row(scenario.seed)
            else:
                if progress and _should_report(index, total):
                    progress(index, total, len(accepted), len(rejected))
                continue
            errs = validate_row(row)
            if errs:
                rejected.append({**row, "reject_reason": f"validator: {errs[0].rule}"})
            else:
                accepted.append(row)
            if progress and _should_report(index, total):
                progress(index, total, len(accepted), len(rejected))
    finally:
        if annotator is not None:
            annotator.quit()
    if len(accepted) <= near_dedup_limit:
        if stage_progress:
            stage_progress(f"dedup start accepted={len(accepted)}")
        accepted = drop_near_duplicates(accepted)
        if stage_progress:
            stage_progress(f"dedup done accepted={len(accepted)}")
    elif stage_progress:
        stage_progress(f"dedup skipped accepted={len(accepted)} limit={near_dedup_limit}")
    needed_rejects = rejected_target - len(rejected)
    if stage_progress:
        stage_progress(f"audit rejects start needed={needed_rejects}")
    rejected.extend(_audit_rejects(accepted, needed_rejects))
    if stage_progress:
        stage_progress(f"audit rejects done rejected={len(rejected)}")
    if stage_progress:
        stage_progress(f"write start out={out}")
    write_rows(out / "accepted.jsonl", accepted)
    write_rows(out / "rejected.jsonl", rejected)
    if stage_progress:
        stage_progress(f"write done accepted={len(accepted)} rejected={len(rejected)}")
    return len(accepted), len(rejected)


def _should_report(index: int, total: int) -> bool:
    return index == 1 or index == total or index % 1000 == 0


def _audit_rejects(rows: list[dict], needed: int) -> list[dict]:
    if needed <= 0:
        return []
    rejects: list[dict] = []
    fixtures = (
        ("audit_fixture: undeclared_tool", _bad_undeclared_tool),
        ("audit_fixture: final_xml", _bad_final_xml),
        ("audit_fixture: duplicate_tool", _bad_duplicate_tool),
        ("audit_fixture: invalid_arg", _bad_invalid_arg),
        ("audit_fixture: disabled_plugin_tool", _bad_disabled_plugin_tool),
        ("audit_fixture: uninstalled_market_tool", _bad_uninstalled_market_tool),
        ("audit_fixture: absent_skill", _bad_absent_skill),
        ("audit_fixture: false_install_claim", _bad_false_install_claim),
        ("audit_fixture: skipped_helper_skill", _bad_skipped_helper_skill),
        ("audit_fixture: helper_tool_before_skill", _bad_helper_tool_before_skill),
        ("audit_fixture: irrelevant_skill_selected", _bad_irrelevant_skill_selected),
    )
    for idx, row in enumerate(rows):
        reason, mutate = fixtures[idx % len(fixtures)]
        bad = {**row, "id": f"reject_{idx:05d}_{row['id']}", "reject_reason": reason}
        bad["messages"] = mutate(row["messages"])
        rejects.append(bad)
        if len(rejects) == needed:
            break
    return rejects


def _bad_undeclared_tool(messages: list[dict]) -> list[dict]:
    return messages[:-1] + [
        {"role": "assistant", "content": "<tool>undeclared_probe input=x</tool>"},
        messages[-1],
    ]


def _bad_final_xml(messages: list[dict]) -> list[dict]:
    return messages[:-1] + [{"role": "assistant", "content": "Final leaks <tool>eval depth=15</tool>."}]


def _bad_duplicate_tool(messages: list[dict]) -> list[dict]:
    return [
        {"role": "assistant", "content": "<tool>board_state fields=basic</tool>"},
        {"role": "assistant", "content": "<tool>board_state fields=basic</tool>"},
    ]


def _bad_invalid_arg(messages: list[dict]) -> list[dict]:
    return messages[:-1] + [
        {"role": "assistant", "content": "<tool>board_state fields=basic extra=bad</tool>"},
        messages[-1],
    ]


def _bad_disabled_plugin_tool(messages: list[dict]) -> list[dict]:
    return messages[:-1] + [
        {"role": "assistant", "content": "<tool>market_scan input=position</tool>"},
        messages[-1],
    ]


def _bad_uninstalled_market_tool(messages: list[dict]) -> list[dict]:
    return messages[:-1] + [
        {"role": "assistant", "content": "<tool>market_openings_search query=sicilian</tool>"},
        messages[-1],
    ]


def _bad_absent_skill(messages: list[dict]) -> list[dict]:
    return [
        {"role": "assistant", "content": "<skill>missing-market-skill</skill>"},
        messages[-1],
    ]


def _bad_false_install_claim(messages: list[dict]) -> list[dict]:
    return messages[:-1] + [
        {"role": "assistant", "content": "<tool>market_install plugin=market-tactics</tool>"},
        {"role": "assistant", "content": "Installed market-tactics and used it successfully."},
    ]


def _bad_skipped_helper_skill(messages: list[dict]) -> list[dict]:
    return messages[:1] + [m for m in messages[3:] if "hood-human-chat" not in m.get("content", "")]


def _bad_helper_tool_before_skill(messages: list[dict]) -> list[dict]:
    return [
        messages[0],
        {"role": "assistant", "content": "<tool>normalize_human_chat text=messy_user_chat extra=before_skill</tool>"},
        *messages[1:],
    ]


def _bad_irrelevant_skill_selected(messages: list[dict]) -> list[dict]:
    return [
        messages[0],
        {"role": "assistant", "content": "<skill>cooking-helper</skill>"},
        *messages[1:],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--profile", default="v1.2")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    dataset_profile = profile(args.profile)
    plan = plan_for_profile(dataset_profile, tiny=args.tiny)
    rejected_target = 8 if args.tiny else dataset_profile.rejected_target
    progress = print_progress if not args.tiny else None
    ok, bad = run(
        plan,
        seed=args.seed,
        out=dataset_profile.gold_dir,
        rejected_target=rejected_target,
        progress=progress,
        stage_progress=print_stage_progress if not args.tiny else None,
        near_dedup_limit=sum(plan.values()) if args.tiny else 10_000,
    )
    print(f"wrote accepted={ok} rejected={bad}")


def print_progress(index: int, total: int, accepted: int, rejected: int) -> None:
    print(
        f"progress {index}/{total} accepted={accepted} rejected={rejected}",
        flush=True,
    )


def print_stage_progress(message: str) -> None:
    print(f"stage {message}", flush=True)


if __name__ == "__main__":
    main()
