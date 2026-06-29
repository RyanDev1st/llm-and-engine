"""Domain-neutral view of the LIVE tool surface for a given plugin_context.

The deterministic harness helpers — malformed/tagless-call recovery, the tool-as-skill
corrective error, and final-answer result-grounding — were hardcoded to the chess tool set.
That made them blind to enabled plugin tools: a tagless plugin call wouldn't recover, a
plugin result wasn't grounding-enforced, and the corrective error couldn't show a plugin
tool's real args. This module exposes the same surface the system prompt advertises
(official catalog + the compute tool + enabled plugins' tools) so those helpers read the
live manifest instead of a baked-in chess list.

NO per-domain regex: generic_result_signal parses the shared `<name>: ... <token>` result
shape every tool already emits, so one heuristic grounds chess and plugin results alike."""
from __future__ import annotations

import re


def full_manifest(plugin_context: dict | None = None) -> list[dict]:
    """Official catalog tools + the compute tool + enabled plugins' tools — the same set
    serving_tool_manifest advertises in the prompt. Composed from the primitives here (not
    imported from inference) so this module stays import-cycle-free."""
    from llm_dataset.v1.catalog import compute_tools, harness_tools, official_tools
    from . import plugins
    return harness_tools() + official_tools() + compute_tools() + plugins.plugin_tools(plugin_context)


def live_tool_names(plugin_context: dict | None = None) -> set[str]:
    """Every callable tool name in the current surface (feeds recovery + coverage gates)."""
    return {t["name"] for t in full_manifest(plugin_context)}


def tool_schema(plugin_context: dict | None, name: str) -> dict | None:
    """The args dict of a tool in the live manifest (for a schema-rich corrective error),
    or None when the name isn't a live tool."""
    for t in full_manifest(plugin_context):
        if t.get("name") == name:
            return t.get("args") or {}
    return None


_RESULT_PREFIX = re.compile(r"^[a-z][a-z0-9_]*:\s*(.*)$", re.I | re.S)
_AFTER_EQ = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")            # the computed value after '='
# number GLUED to a unit (2x, 10s) — but the number must start its own token, so a square
# coordinate like e1 (digit glued to a LETTER) is not mistaken for a quantity.
_COMPACT_NUM_UNIT = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?[A-Za-z%]+")
_NUM = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")          # standalone number, not the 1 in e1


def generic_result_signal(result: str) -> str | None:
    """A short distinctive token a grounded reply should echo, extracted from ANY tool's
    `<name>: ...` result line — domain-neutral, mirroring chess `_result_signal`'s job. Prefers
    the value AFTER an '=' (a computed conversion: `convert: 5 mi = 8.047 km` -> '8.047'), else a
    compact number+unit (`by 2x` -> '2x', `10s set` -> '10s'), else the first bare number. Returns
    None for an error or a numberless result, so we never risk a spurious grounded append."""
    t = (result or "").strip()
    if not t or t.startswith("error"):
        return None
    m = _RESULT_PREFIX.match(t)
    body = m.group(1) if m else t
    eq = _AFTER_EQ.search(body)
    if eq:
        return eq.group(1)
    nu = _COMPACT_NUM_UNIT.search(body)
    if nu:
        return nu.group(0)
    n = _NUM.search(body)
    return n.group(0) if n else None
