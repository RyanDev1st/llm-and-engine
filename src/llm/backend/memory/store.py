"""Disk-backed persistent user memory + the write gate. JSON is the source of truth;
render_profile() turns it into the short USER PROFILE block injected into the system prompt
every turn (the CLAUDE.md pattern — re-injected each turn so it survives compaction).

The gate is what keeps auto-capture from rotting (Cursor removed un-gated auto-memory):
categories are a whitelist, `rating` is single-valued (newest wins), the other lists are
capped (oldest drops), values are length-bounded, and anything resembling transient board
state / PII is rejected even via the explicit "remember…" note path. Curated, bounded —
never an unbounded dump. Keyed by user_id so multi-user is free later (single-user demo
defaults to "default")."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ROOT = Path(os.environ.get("CHESS_MEMORY_DIR")
             or Path(__file__).resolve().parents[3] / "data" / "memory")
ALLOWED = ("rating", "style", "weakness", "pref", "note")
_SINGLE = {"rating"}        # single-valued: newest replaces
_CAP = 6                    # max entries per list-category (oldest drops when full)
_VAL_MAX = 140              # chars; reject a runaway value
# never store transient board state or PII, even via the opted-in note path.
_REJECT = re.compile(
    r"[pnbrqkPNBRQK1-8]{2,}/[pnbrqkPNBRQK1-8/]+\s+[wb]\b"      # a FEN
    r"|\b[\w.+-]+@[\w-]+\.\w+\b"                                # an email
    r"|\b\d{7,}\b")                                             # a long digit run (phone/card/id)


def _path(user_id: str) -> Path:
    uid = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "default") or "default"
    return _ROOT / uid / "profile.json"


def load_profile(user_id: str = "default") -> dict:
    p = _path(user_id)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_profile(user_id: str, profile: dict) -> None:
    p = _path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def add_fact(profile: dict, cat: str, value: str) -> bool:
    """The write gate. Validate category, reject board-state/PII/over-long, normalize,
    dedupe, then supersede (rating) or append-with-cap (lists). Mutates `profile`; returns
    True only when it actually changed."""
    if cat not in ALLOWED:
        return False
    val = " ".join((value or "").split())
    if not val or len(val) > _VAL_MAX or _REJECT.search(val):
        return False
    if cat in _SINGLE:
        if profile.get(cat) == val:
            return False
        profile[cat] = val
        return True
    lst = profile.setdefault(cat, [])
    if not isinstance(lst, list):                      # tolerate a hand-edited file
        lst = profile[cat] = [lst]
    if any(str(v).lower() == val.lower() for v in lst):   # dedupe (case-insensitive)
        return False
    lst.append(val)
    if len(lst) > _CAP:                                # bounded: oldest drops
        del lst[0]
    return True


_RENDER = (("style", "plays"), ("weakness", "watch-outs"),
           ("pref", "preferences"), ("note", "remember"))


def render_profile(profile: dict) -> str:
    """The short USER PROFILE block injected into the system prompt; '' when empty."""
    if not profile:
        return ""
    parts: list[str] = []
    if profile.get("rating"):
        parts.append(f"rating {profile['rating']}")
    for cat, label in _RENDER:
        vals = profile.get(cat) or []
        if isinstance(vals, str):
            vals = [vals]
        if vals:
            parts.append(f"{label}: " + "; ".join(str(v) for v in vals))
    if not parts:
        return ""
    return ("USER PROFILE (persists across sessions — tailor coaching to it; do not restate "
            "it verbatim):\n- " + "\n- ".join(parts))


def capture(user_message: str, user_id: str = "default") -> int:
    """Mine one user message and persist any NEW typed facts (idempotent — re-capturing the
    same message writes nothing). Returns the number of facts written."""
    from .extract import extract_facts
    profile = load_profile(user_id)
    written = 0
    for cat, val in extract_facts(user_message):
        written += int(add_fact(profile, cat, val))
    if written:
        save_profile(user_id, profile)
    return written


def memory_block(user_id: str = "default") -> str:
    """The system-prompt memory block for `user_id` ('' when the profile is empty)."""
    return render_profile(load_profile(user_id))
