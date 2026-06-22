"""Episodic 'how-to-operate' memory — the fourth tier: the system LEARNS to use tools better
across turns/runs/machines, with the model FROZEN.

The signal is free: the serve loop already produces a correction whenever the model fires a tool
the wrong way, reads the error, and fixes it (bare `<tool>scale_recipe>` -> error -> `<tool>
scale_recipe from_servings=12 to_servings=30>` -> grounded answer). observe() harvests that as an
episode (trigger request -> the call that WORKED); episodic_block() recalls the closest past episode
for a new request and injects a one-line HINT, so next time the model goes one-shot.

Design:
- GLOBAL, not per-user — a lesson is about operating a TOOL, so it helps every user/machine. The
  store is a single JSON file under CHESS_MEMORY_DIR (sync that dir => cross-machine learning).
- Gated like the profile store (extract.py's lesson): only harvest a turn that REACHED a grounded
  answer (reply present) via an error->fix on the SAME tool; reject PII/board-state (reuse _REJECT);
  one lesson per tool (newest refreshes); bounded (oldest-by-use evicted). Poison control is the job.
- FLAG-gated: off unless CHESS_EPISODIC=1, so default serve behavior is byte-identical.
- Recall is cheap LEXICAL similarity (no model, no embeddings) and only surfaces a tool that is in
  the LIVE manifest — domain-general (keys on request text + manifest, not chess)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .store import _REJECT          # reuse the PII / board-state reject gate

_TOOL = re.compile(r"<tool>\s*([A-Za-z0-9_][\w-]*)([^<]*)</tool>")
_STOP = {"the", "a", "an", "to", "of", "my", "is", "it", "me", "for", "and", "how", "do", "can",
         "you", "with", "this", "that", "into", "from", "please", "pls", "up", "down", "i"}
_CAP = 50                            # max episodes (bounded; least-used evicted)
_SIM_MIN = 0.34                      # Jaccard floor to recall (avoids spurious cross-domain hits)
_TRIGGER_MAX = 160


def _enabled() -> bool:
    return os.environ.get("CHESS_EPISODIC", "0") == "1"


def _path() -> Path:
    root = Path(os.environ.get("CHESS_MEMORY_DIR")
                or Path(__file__).resolve().parents[3] / "data" / "memory")
    return root / "episodic.json"


def _load() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save(eps: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(eps, ensure_ascii=False, indent=2), encoding="utf-8")


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) > 1 and w not in _STOP}


def _sim(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _tool_calls(result: dict) -> list[tuple[str, str, str]]:
    """(tool_name, canonical_call, result_string) for each <tool> call in the turn's display
    history. Skill loads (<skill>…) are ignored — episodes are about TOOL usage."""
    out: list[tuple[str, str, str]] = []
    for call, res in zip(result.get("tool_calls") or [], result.get("tool_results") or []):
        m = _TOOL.search(call or "")
        if m:
            out.append((m.group(1), f"<tool>{m.group(1)}{m.group(2).rstrip()}</tool>", res or ""))
    return out


def _correction(result: dict) -> tuple[str, str] | None:
    """A tool that ERRORED then later SUCCEEDED in the same turn -> (tool, the working call).
    That's a self-correction the model made and should not have to repeat next time."""
    errored: set[str] = set()
    for name, call, res in _tool_calls(result):
        if res.startswith("error"):
            errored.add(name)
        elif name in errored:                       # success after an earlier error of this tool
            return name, call
    return None


def add_episode(eps: list[dict], trigger: str, tool: str, lesson: str) -> bool:
    """The write gate: validate + PII-reject, then ONE episode per tool (refresh + bump hits),
    bounded (evict least-used). Mutates `eps`; returns True only when it changed."""
    val = " ".join((trigger or "").split())
    if not val or not tool or not lesson or _REJECT.search(val) or _REJECT.search(lesson):
        return False
    for e in eps:
        if e.get("tool") == tool:                   # dedupe by tool — refresh the canonical lesson
            e["lesson"], e["trigger"] = lesson, val[:_TRIGGER_MAX]
            e["hits"] = int(e.get("hits", 1)) + 1
            return True
    eps.append({"tool": tool, "trigger": val[:_TRIGGER_MAX], "lesson": lesson, "hits": 1})
    if len(eps) > _CAP:
        eps.sort(key=lambda e: int(e.get("hits", 1)))   # least-used first
        del eps[0]
    return True


def observe(user_message: str, result: dict, plugin_context: dict | None = None) -> bool:
    """Harvest a correction episode from a completed turn. No-op unless enabled; stores nothing
    unless the turn REACHED a grounded answer (reply present) via a same-tool error->fix."""
    if not _enabled() or not (result.get("reply") or "").strip():
        return False
    corr = _correction(result)
    if not corr:
        return False
    tool, lesson = corr
    eps = _load()
    if add_episode(eps, user_message, tool, lesson):
        _save(eps)
        return True
    return False


def recall(user_message: str, live_names: set[str] | None) -> dict | None:
    """The episode most similar to `user_message` whose tool is callable now, or None."""
    if not _enabled():
        return None
    inc = _tokens(user_message)
    best, best_s = None, 0.0
    for e in _load():
        if e.get("tool") not in (live_names or set()):
            continue
        s = _sim(inc, _tokens(e.get("trigger", "")))
        if s > best_s:
            best, best_s = e, s
    return best if best_s >= _SIM_MIN else None


def episodic_block(user_message: str, plugin_context: dict | None = None) -> str:
    """The system-prompt RECALLED block for the closest past lesson ('' when none / disabled).
    Framed as a HINT (not a rule) and kept to one line — the model already tolerates injected
    profile + live-board context in this idiom, so this stays on-distribution."""
    if not _enabled():
        return ""
    from ..manifest_view import live_tool_names      # import here to avoid a load-time cycle
    e = recall(user_message, live_tool_names(plugin_context))
    if not e:
        return ""
    return ("RECALLED (a similar past request — a hint, not a rule; only use if it fits): "
            f"{e['lesson']} worked for \"{e['trigger']}\".")
