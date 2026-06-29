"""Native Gemma-4 wire-format translation for the v5 serve (the single home).

The v5 corpus trains IN Gemma's own native format (single-token markers — see
docs/reference/native-gemma-format.md): assistant turns carry STRUCTURED `tool_calls`,
tool results are native `role="tool"` blocks, thinking is the native `<|channel>thought`,
and a tool call on the wire is `<|tool_call>call:NAME{args}<tool_call|>` (string values
wrapped in the quote token `<|"|>`, ints/bools bare, keys bare, sorted).

To reuse the entire battle-tested CoachLoop (coverage / grounding guards / dedup / plan
boxes) we translate at the two boundaries ONLY:
  - parse_native_call: model OUTPUT  -> the canonical `<tool>NAME k=v</tool>` the loop speaks.
  - to_native_messages: loop HISTORY -> structured `tool_calls` + named `role="tool"` so the
    template re-renders byte-identically to training (NO remap — role="tool" survives natively).
Everything between these two boundaries is unchanged from the v4 path.
"""
from __future__ import annotations

import re

from .toolfmt import parse_call

# A call on the wire: `<|tool_call>call:NAME{args}<tool_call|>` (close may be cut by the stop).
_NATIVE_CALL = re.compile(
    r"<\|tool_call>\s*call:\s*([a-z0-9_-]+)\s*(\{.*?\})?\s*(?:<tool_call\|>|$)", re.DOTALL)
# Fallback: the model dropped the wrapper but kept `call:NAME{...}` (seen as a recovery artifact).
_NATIVE_CALL_BARE = re.compile(r"\bcall:\s*([a-z0-9_-]+)\s*(\{.*?\})?", re.DOTALL)
_QUOTE = '<|"|>'                                   # the string-value quote token (id 52)
# Free-text args go LAST so parse_call's rest-of-line capture (it may contain spaces / '=') works.
_FREE_LAST = ("query", "fen", "code")
_NATIVE_CLOSERS = ("<tool_call|>",)               # generation stops here -> one action per step
_SKILL = re.compile(r"<skill>\s*([A-Za-z0-9_][\w-]*)\s*</skill>")


def _native_args(blob: str) -> list[tuple[str, str]]:
    """key:value pairs from a `{...}` arg blob. Values run to the next ', key:' or end, so a
    string value with spaces/commas survives. The quote token is dropped (display schema only)."""
    if not blob:
        return []
    inner = blob.strip()
    inner = inner[1:-1] if inner.startswith("{") and inner.endswith("}") else inner.lstrip("{").rstrip("}")
    inner = inner.replace(_QUOTE, "")
    pairs: list[tuple[str, str]] = []
    for m in re.finditer(r"([A-Za-z_][\w-]*)\s*:\s*(.*?)(?=,\s*[A-Za-z_][\w-]*\s*:|$)", inner, re.DOTALL):
        v = m.group(2).strip().strip(",").strip()
        pairs.append((m.group(1).strip().lower(), v))
    return pairs


def parse_native_call(text: str) -> str | None:
    """Translate the first native tool call in `text` to canonical `<tool>NAME k=v</tool>`,
    or None when `text` carries no call (a plain final answer). A preceding `<|channel>thought`
    block is ignored (it's lifted to the panel by inference._split_reasoning)."""
    s = (text or "").strip()
    m = _NATIVE_CALL.search(s) or _NATIVE_CALL_BARE.search(s)
    if not m:
        return None
    name = m.group(1)
    pairs = _native_args(m.group(2) or "")
    pairs.sort(key=lambda kv: kv[0] in _FREE_LAST)
    args = " ".join(f"{k}={v}" for k, v in pairs if v != "")
    return f"<tool>{name}{(' ' + args) if args else ''}</tool>"


def _coerce(v):
    """Re-type a string arg so the re-rendered history matches training (bare ints, not quoted)."""
    if not isinstance(v, str):
        return v
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _assistant_tool_calls(content: str) -> list[dict] | None:
    """Structured tool_calls for an assistant history turn, or None if it is a plain answer.
    `<skill>X</skill>` -> load_skill(name=X); `<tool>NAME args</tool>` -> NAME(args)."""
    c = (content or "").strip()
    sk = _SKILL.search(c)
    if sk:
        return [{"type": "function", "function": {"name": "load_skill", "arguments": {"name": sk.group(1)}}}]
    if "<tool>" in c:
        name, args = parse_call(c)
        if name:
            return [{"type": "function",
                     "function": {"name": name, "arguments": {k: _coerce(v) for k, v in args.items()}}}]
    return None


def to_native_messages(messages: list[dict]) -> list[dict]:
    """Rewrite loop history into the native shape the v5 template trained on: assistant
    `<tool>`/`<skill>` text -> structured `tool_calls` (content=""), and each `role="tool"`
    result carries the `name` of the call it answers (so the template renders a native
    <|tool_response> block instead of dropping it). NO remap-to-user."""
    out: list[dict] = []
    last_name = None
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({"role": "tool", "name": m.get("name") or last_name or "tool",
                        "content": m.get("content", "")})
            continue
        if role == "assistant":
            calls = _assistant_tool_calls(m.get("content", ""))
            if calls:
                last_name = calls[0]["function"]["name"]
                out.append({"role": "assistant", "content": "", "tool_calls": calls})
                continue
        out.append(m)
    return out


def native_stop_ids(tok) -> set[int]:
    """Token ids that END one action: the `<tool_call|>` close marker (single token). Added to
    the model's eos set so a generation halts right after a tool call (one action per step)."""
    ids: set[int] = set()
    unk = getattr(tok, "unk_token_id", None)
    for name in _NATIVE_CLOSERS:
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0 and i != unk:
            ids.add(i)
    return ids
