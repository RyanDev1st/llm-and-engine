from __future__ import annotations

from typing import Any

from .tags import tool_calls_of


def build_chess_envelope(scenario, messages: list[dict[str, Any]], annotated, mode: str) -> dict[str, Any]:
    expected = [
        tc["name"]
        for m in messages
        if m["role"] == "assistant"
        for tc in tool_calls_of(m)
        if tc["name"] != "load_skill"
    ]
    rules = [
        "final_no_xml", "known_tool_only", "args_match_schema",
        "selected_skill_exists", "skill_index_only_before_load", "skill_body_strict",
    ]
    if scenario.slice != "J":
        rules.append("engine_grounded")
    if scenario.slice in {"D", "E", "F", "G"}:
        rules.append("narration_grounded")
    return {
        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
        "slice": scenario.slice,
        "kind": "harness_chess",
        "reasoning_mode": mode,
        "intent": scenario.intent,
        "plugin_context": scenario.plugin_context,
        "skills_index": [dict(s) for s in scenario.skills_index],
        "selected_skills": ["chess-coach"] if any(
            tc["name"] == "load_skill" for m in messages if m["role"] == "assistant" for tc in tool_calls_of(m)
        ) else [],
        "tool_manifest": list(scenario.tool_manifest),
        "expected_tool_calls": expected,
        "grounding_sources": ["board_state"] if scenario.slice in {"A", "B", "C", "D", "E", "F", "G", "H"} else [],
        "messages": messages,
        "acceptance_rules": rules,
        "position_fen": scenario.position.fen if scenario.position else None,
        "stockfish_truth": (
            {"score_cp": annotated.score_cp, "best_san": annotated.best_san, "depth": annotated.depth}
            if annotated else None
        ),
    }
