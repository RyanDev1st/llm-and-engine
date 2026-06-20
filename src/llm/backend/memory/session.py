"""Session memory plane — a FEN-keyed cache of the facts already computed this session, so a
follow-up reuses them instead of re-calling the engine (the cross-turn tool-result reuse).

The in-turn tool scratchpad is discarded after each turn (web_app keeps only user+reply), so
without this a "why?" / "what's the plan?" follow-up re-runs eval/best_move from scratch. We
capture the grounded fact lines from a turn's tool results, tagged with the FEN they were
computed at, and re-inject them next turn ONLY IF the live board still matches that FEN —
a strict freshness guard, so a move/undo silently invalidates the cache (no stale facts).

Deterministic, no model call (matches the compaction philosophy). Bounded to a few lines."""
from __future__ import annotations

_MAX_FACTS = 5
# Grounded tool-result prefixes worth carrying forward (an analysis fact about THIS position).
# Excludes board_state (re-derivable, cheap) and load_skill bodies (not a fact).
_FACT_PREFIXES = ("score:", "best:", "best_line:", "best_moves:", "threats:", "review:")


def update(session: dict, fen: str, tool_results: list[str]) -> None:
    """Record this turn's grounded analysis facts under the FEN they pertain to. Replaces the
    cache when the position changed (a new FEN), else merges (same position, more facts)."""
    fresh = [r.strip() for r in (tool_results or [])
             if r and r.strip().startswith(_FACT_PREFIXES)]
    if not fresh:
        return
    if session.get("fen") != fen:                 # position changed -> start a new cache
        session["fen"] = fen
        session["facts"] = []
    facts = session.setdefault("facts", [])
    for f in fresh:
        if f not in facts:
            facts.append(f)
    del facts[:-_MAX_FACTS]                        # keep the most recent few


def render(session: dict, current_fen: str) -> str:
    """The session-note block to inject — ONLY when the cached facts pertain to the CURRENT
    position (freshness guard). '' when empty or the board has moved on."""
    if not session or session.get("fen") != current_fen:
        return ""
    facts = session.get("facts") or []
    if not facts:
        return ""
    return ("ESTABLISHED THIS SESSION (already computed for the current position — reuse "
            "these, do not re-call the tool):\n- " + "\n- ".join(facts))


def clear(session: dict) -> None:
    session.clear()
