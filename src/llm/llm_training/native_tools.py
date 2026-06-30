from __future__ import annotations

import re
from typing import Any

_TOOLS_HEAD = "AVAILABLE TOOLS"
_TOOL_LINE = re.compile(r"^-\s+([A-Za-z_][\w-]*)(.*?)\s{2,}(.+?)(?:\s+\[[^\]]+\])?$")
_ARG = re.compile(r"([A-Za-z_][\w-]*)=<([^>]+)>")
_INT_ARGS = {"depth", "top", "series", "num", "count", "ply", "limit"}


def _schema_for_arg(name: str, rule: Any) -> dict[str, Any]:
    if isinstance(rule, list):
        vals = [_coerce_enum(v) for v in rule]
        typ = "integer" if vals and all(isinstance(v, int) for v in vals) else "string"
        return {"type": typ, "enum": vals}
    if name in _INT_ARGS:
        return {"type": "integer"}
    return {"type": "string"}


def _coerce_enum(value: Any) -> Any:
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def manifest_to_native_tools(tool_manifest: list[dict[str, Any]] | tuple[dict[str, Any], ...],
                             include_descriptions: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tool_manifest or []:
        args = tool.get("args") or {}
        props = {name: _schema_for_arg(name, rule) for name, rule in args.items()}
        required = [name for name, rule in args.items() if rule == "required"]
        params: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            params["required"] = required
        out.append({"type": "function", "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", "") if include_descriptions else "",
            "parameters": params,
        }})
    return out


def tools_from_system_prompt(system: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    in_tools = False
    for raw in (system or "").splitlines():
        line = raw.rstrip()
        if line.startswith(_TOOLS_HEAD):
            in_tools = True
            continue
        if in_tools and not line:
            break
        if not in_tools or not line.startswith("- "):
            continue
        m = _TOOL_LINE.match(line)
        if not m:
            continue
        name, arg_text, desc = m.groups()
        args = {a: r for a, r in _ARG.findall(arg_text)}
        tools.append({"name": name, "description": desc.strip(), "args": args})
    return manifest_to_native_tools(tools)


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        clean = {k: v for k, v in msg.items() if not k.startswith("_") and k != "reasoning"}
        if msg.get("reasoning"):
            clean["reasoning_content"] = msg["reasoning"]
        out.append(clean)
    return out


def tools_for_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[0].get("_native_tools"):
        return list(messages[0]["_native_tools"])
    if messages and messages[0].get("role") == "system":
        return tools_from_system_prompt(messages[0].get("content", ""))
    return []
