from __future__ import annotations

from typing import Any

from .tags import LOAD_SKILL, tool_calls_of
from .thinking import gated_answer, gated_think


def attach_reasoning_traces(messages: list[dict[str, Any]], *, mode: str, seed: int,
                            goal: str = "") -> list[dict[str, Any]]:
    """Attach lightweight thinking traces to non-fast assistant turns.

    The training pipeline renders these through the native thought channel but masks
    them from loss. They keep the LoRA conditioned on actions-after-thought without
    forcing our stub wording over Gemma's native CoT.
    """
    if (mode or "").strip().lower() == "fast":
        return messages
    step = 0
    have = ""
    for msg in messages:
        if msg.get("role") != "assistant" or msg.get("train") is False:
            continue
        calls = tool_calls_of(msg)
        if calls:
            action = calls[0]["name"]
            kind = _kind_for(action)
            trace = gated_think(seed, action, step, mode=mode, kind=kind, goal=goal, have=have)
            if action == LOAD_SKILL:
                have = "skill"
            elif action == "board_state":
                have = "board"
            else:
                have = "results"
        else:
            trace = gated_answer(seed + step, goal, mode=mode)
        if trace:
            msg["reasoning"] = trace
        step += 1
    return messages


def _kind_for(action: str) -> str:
    if action == LOAD_SKILL:
        return "select"
    if action in {"board_state", "list_pieces"}:
        return "routine"
    return "decide"
