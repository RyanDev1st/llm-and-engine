from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from ..contracts.tool_grammar import parse_tool_name

EXACT_MATCH_TOOLS = {"move", "undo", "legal_moves", "list_pieces"}
TOLERANCE_TOOLS = {"eval", "best_move", "review_move", "threats"}
SKIP_REPLAY_TOOLS = {"ask_chessbot"}


@dataclass(frozen=True)
class ReplayFailure:
    turn_index: int
    tool_name: str
    reason: str


class ToolBackend(Protocol):
    def execute(self, tool_call: str) -> str:
        ...


def replay_validate(messages: list[dict[str, str]], backend: ToolBackend) -> list[ReplayFailure]:
    failures: list[ReplayFailure] = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        call = message.get("content", "")
        tool_name = parse_tool_name(call)
        if not tool_name:
            continue
        if idx + 1 >= len(messages) or messages[idx + 1].get("role") != "tool":
            failures.append(ReplayFailure(idx, tool_name, "missing tool result after tool call"))
            continue
        recorded = messages[idx + 1].get("content", "")
        if tool_name in SKIP_REPLAY_TOOLS:
            continue
        observed = backend.execute(call)
        if tool_name in EXACT_MATCH_TOOLS and observed != recorded:
            failures.append(ReplayFailure(idx, tool_name, "exact mismatch"))
            continue
        if tool_name in TOLERANCE_TOOLS and not _tolerance_match(observed, recorded):
            failures.append(ReplayFailure(idx, tool_name, "tolerance mismatch"))
    return failures


def _tolerance_match(observed: str, recorded: str) -> bool:
    if observed == recorded:
        return True
    observed_json = _maybe_json(observed)
    recorded_json = _maybe_json(recorded)
    if observed_json is not None and recorded_json is not None:
        return _tolerance_match_json(observed_json, recorded_json)
    if "mate in" in observed and "mate in" in recorded:
        return observed == recorded
    return _bucket(observed) == _bucket(recorded)


def _tolerance_match_json(observed: dict, recorded: dict) -> bool:
    if observed.get("ok") is False or recorded.get("ok") is False:
        return observed.get("error_code") == recorded.get("error_code")
    if observed.get("tool") == "eval" and recorded.get("tool") == "eval":
        obs = int(observed.get("score_cp", 0))
        rec = int(recorded.get("score_cp", 0))
        return abs(obs - rec) <= 30
    if observed.get("tool") == "best_move" and recorded.get("tool") == "best_move":
        observed_top = observed.get("top_k", [])
        recorded_move = recorded.get("move")
        return isinstance(observed_top, list) and recorded_move in observed_top
    return observed.get("tool") == recorded.get("tool") and _bucket(str(observed)) == _bucket(str(recorded))


def _maybe_json(value: str) -> dict | None:
    try:
        data = json.loads(value)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _bucket(value: str) -> str:
    lowered = value.lower()
    if "+" in lowered:
        if any(token in lowered for token in ["+3", "+4", "+5", "+6", "+7", "+8", "+9"]):
            return "white_decisive"
        if any(token in lowered for token in ["+1", "+2"]):
            return "white_edge"
    if "-" in lowered:
        if any(token in lowered for token in ["-3", "-4", "-5", "-6", "-7", "-8", "-9"]):
            return "black_decisive"
        if any(token in lowered for token in ["-1", "-2"]):
            return "black_edge"
    return "even_or_unknown"
