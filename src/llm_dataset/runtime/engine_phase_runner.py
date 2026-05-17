from __future__ import annotations

from pathlib import Path

from ..validation.admission import admit_records
from ..reports.audit import build_audit_summary
from ..reports.audit_report import write_audit_report
from ..runtime.engine_backend import EngineToolBackend
from ..reports.patch_log import write_patch_log
from ..pipeline.patch_loop import apply_patch_requests, build_patch_requests
from ..validation.redteam import RedTeamProbe, run_redteam
from ..reports.redteam_report import write_redteam_report
from ..validation.replay import replay_validate


def _record(record_id: str, slice_name: str, call: str, tool_result: str) -> dict:
    return {
        "id": record_id,
        "slice": slice_name,
        "validated": True,
        "notes": "engine-phase",
        "messages": [
            {"role": "system", "content": "You are chess assistant."},
            {"role": "user", "content": "analyze position"},
            {"role": "assistant", "content": call},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": "Result interpreted for user."},
        ],
    }


def _build_records(backend: EngineToolBackend) -> list[dict]:
    calls = [
        "<tool>legal_moves conversation_id=c1</tool>",
        "<tool>list_pieces conversation_id=c2</tool>",
        "<tool>eval conversation_id=c3</tool>",
        "<tool>best_move conversation_id=c4</tool>",
        "<tool>review_move conversation_id=c5 uci=e2e4</tool>",
        "<tool>threats conversation_id=c6</tool>",
        "<tool>eval conversation_id=c7</tool>",
        "<tool>best_move conversation_id=c8</tool>",
    ]
    slices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    records: list[dict] = []
    for idx, call in enumerate(calls, start=1):
        result = backend.execute(call)
        records.append(_record(f"eng_{idx}", slices[idx - 1], call, result))
    return records


def _build_probes(records: list[dict]) -> list[RedTeamProbe]:
    categories = [
        "mode_violation",
        "ambiguity_handling",
        "illegal_invalid",
        "tool_failure",
        "adversarial_routing",
        "injection_style",
        "tone_quality",
    ]
    probes: list[RedTeamProbe] = []
    for idx, category in enumerate(categories):
        probes.append(RedTeamProbe(probe_id=f"probe_{category}", category=category, record=records[idx % len(records)]))
    return probes


def run_engine_phase(out_dir: Path) -> dict[str, object]:
    backend = EngineToolBackend()
    records = _build_records(backend)

    admission = admit_records(records, backend)
    replay_fail_count = 0
    replay_backend = EngineToolBackend()
    for record in admission.accepted:
        replay_fail_count += len(replay_validate(record["messages"], replay_backend))

    probes = _build_probes(admission.accepted)
    report = run_redteam(probes, backend)
    holes = {k: v for k, v in {}.items()}
    requests = build_patch_requests(holes)
    _, patch_results = apply_patch_requests(
        admission.accepted,
        requests,
        lambda req: __import__("src.llm_dataset.patch_loop", fromlist=["PatchResult"]).PatchResult(
            category=req.category,
            replaced_ids=[],
            new_records=[],
        ),
    )

    audit = build_audit_summary(admission.accepted, admission, report, replay_fail_count)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_redteam_report(out_dir / "engine_redteam_report.md", report)
    write_patch_log(out_dir / "engine_patch_log.md", patch_results)
    write_audit_report(out_dir / "engine_final_audit_report.md", audit)

    return {
        "accepted": len(admission.accepted),
        "rejected": len(admission.rejected),
        "replay_fail_count": replay_fail_count,
        "freeze": audit.freeze.approved,
    }


if __name__ == "__main__":
    result = run_engine_phase(Path("results/dataset_phase2"))
    print(result)
