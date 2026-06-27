"""Single source for building the harness ACTION messages.

v5-native: renderers build STRUCTURED messages here instead of writing literal
`<tool>`/`<skill>` strings into content. The Gemma-4 chat template renders these
to native single-token `<|tool_call>call:NAME{args}<tool_call|>` /
`<|tool_response>response:NAME{...}<tool_response|>` blocks (see
docs/reference/native-gemma-format.md). Loading a skill is a native tool call
(`load_skill{name:NAME}`) — Gemma has no `<skill>` verb. Thinking is NOT emitted
here: it is omitted from training rows and invoked at serve via enable_thinking,
so the base model's native reasoning is never overwritten by stubs.
"""
from __future__ import annotations

from typing import Any

LOAD_SKILL = "load_skill"


def _coerce(v: Any) -> Any:
    """Tool-call arg values are structured (int/bool/str), so the template renders
    `depth:18` bare and `san:<|"|>e4<|"|>` quoted exactly as the base was trained."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lstrip("+-").isdigit():
        return int(s)
    return s


def tool_call_msg(name: str, args: dict[str, Any] | None = None, *, content: str = "") -> dict[str, Any]:
    """An assistant turn that calls ONE tool, as a structured message. `content` is
    normally '' (native style: a tool-call turn carries no prose; the grounded
    narration is the FINAL turn). Args are coerced so types match the native render."""
    arguments = {k: _coerce(v) for k, v in (args or {}).items()}
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def tool_result_msg(name: str, content: str) -> dict[str, Any]:
    """A tool result. Stays role='tool': the native template folds it into a
    `<|tool_response>` block after the preceding assistant's structured tool_calls
    (it survives now — no remap needed)."""
    return {"role": "tool", "name": name, "content": content}


def skill_call_msg(name: str) -> dict[str, Any]:
    """Load a listed skill — a native tool call `load_skill{name:NAME}`."""
    return tool_call_msg(LOAD_SKILL, {"name": name})


def scene_args(call: str) -> dict[str, Any]:
    """Parse a specialist scene's terse arg string ('depth=14' / 'kind=puzzle' / '')
    into a structured dict for tool_call_msg. Specialist tools take 0-1 args."""
    call = (call or "").strip()
    if not call:
        return {}
    key, _, val = call.partition("=")
    return {key.strip(): val.strip()}


def tool_calls_of(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """The structured calls on an assistant message (names+args), for envelopes/validators."""
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        out.append({"name": fn.get("name", ""), "arguments": fn.get("arguments", {}) or {}})
    return out
