"""Deterministic fact extraction for persistent user memory — the WRITE-DISCIPLINE GATE.

Auto-capture rots without a gate: Cursor shipped auto-"Memories", then removed it and
reverted to curated rules. So we only ever emit TYPED, durable facts from a small whitelist
of categories, via explicit patterns — never free text, never transient board state. Each
turn's user message is mined for: approximate rating, playing style, a recurring weakness,
and a stated preference. An explicit "remember that I…" is the ONE path that may store a
free-form note (the user opted in), and store.py still re-validates it against the gate.

No model call (matches the deterministic-compaction philosophy); pure regex over the message.
"""
from __future__ import annotations

import re

# rating: a plausible chess number tied to a rating word or a ~NNNN form. The word forms get
# a leading \b; the ~ form stands alone (a \b can't sit between a space and '~', which would
# otherwise drop "rated ~1400").
_RATING = re.compile(r"(?:\b(?:rated|rating|elo|around|about|roughly)\s*|~\s*)(\d{3,4})\b"
                     r"|\b(\d{3,4})\s*(?:elo|rated|rating)\b", re.I)
# recurring weakness: an "I always/keep/tend to <blunder-verb> …" self-report.
_WEAKNESS = re.compile(
    r"\bi (?:always|often|keep|tend to|usually|frequently|constantly|habitually)\s+"
    r"((?:hang|hangs|hanging|blunder|blundering|drop|dropping|lose|losing|miss|missing|"
    r"forget|forgetting|overlook|overlooking)\b[^.,;!?]{0,40})", re.I)
# stated preference: phrase -> canonical typed value.
_PREF = [
    (re.compile(r"\bkeep it (?:short|brief|terse|concise|simple)\b", re.I), "prefers terse replies"),
    (re.compile(r"\b(?:be|stay|reply|answer) (?:concise|brief|terse|short)\b", re.I), "prefers terse replies"),
    (re.compile(r"\b(?:be|go|more) (?:detailed|thorough|in.?depth|verbose)\b", re.I), "prefers detailed replies"),
    (re.compile(r"\bexplain (?:it |this )?(?:simply|slowly|like i'?m (?:new|five|5|a beginner|a kid))\b", re.I),
     "prefers simple explanations"),
    (re.compile(r"\bi'?m (?:new|a beginner|just starting|just learning)\b", re.I), "is a beginner"),
]
# playing style: "I play/prefer/open with the <opening>" — kept opening-ish, not a sentence.
_STYLE = re.compile(
    r"\bi (?:play|prefer|like|love|always open|open|main)\s+(?:the\s+|with\s+|as\s+)?"
    r"([A-Za-z][\w' -]{1,30}?(?:\s+(?:opening|defense|defence|system|gambit|attack))?)\b", re.I)
_STYLE_OK = re.compile(r"opening|defense|defence|system|gambit|attack|"
                       r"\b(?:e4|d4|c4|nf3|the london|caro|sicilian|french|italian|ruy|pirc|king'?s)\b", re.I)
# explicit opt-in: "remember (that) I …" — the one free-form path.
_EXPLICIT = re.compile(r"\bremember(?:\s+that)?\s+(?:i\s+)?(.+)", re.I)


def extract_facts(user_message: str) -> list[tuple[str, str]]:
    """Return typed (category, value) facts mined from one user message. Conservative —
    emits only durable, whitelisted categories; noise / '' yields []."""
    msg = (user_message or "").strip()
    if not msg:
        return []
    out: list[tuple[str, str]] = []

    m = _RATING.search(msg)
    if m:
        n = int(m.group(1) or m.group(2))
        if 100 <= n <= 3200:                              # a rating, not a year/quantity
            out.append(("rating", f"~{n}"))

    for m in _WEAKNESS.finditer(msg):
        out.append(("weakness", "tends to " + " ".join(m.group(1).split()).lower()))

    for pat, val in _PREF:
        if pat.search(msg):
            out.append(("pref", val))

    m = _STYLE.search(msg)
    if m:
        val = " ".join(m.group(1).split())
        if _STYLE_OK.search(val) or len(val.split()) <= 2:   # opening-ish, not free prose
            out.append(("style", "plays " + val.lower()))

    m = _EXPLICIT.search(msg)
    if m:
        note = " ".join(m.group(1).split()).rstrip(".!?")
        if 3 <= len(note) <= 120:
            out.append(("note", note))                    # explicit opt-in (re-gated in store)

    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for cat, val in out:                                  # dedupe within this message
        key = (cat, val.lower())
        if key not in seen:
            seen.add(key)
            uniq.append((cat, val))
    return uniq
