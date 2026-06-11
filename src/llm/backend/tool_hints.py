"""Deterministic routing hints for the harness.

A small model occasionally fails to route an obvious request to its tool — it
narrates intent without calling, or stops one tool short. This layer scans the
user's words for unambiguous intent keywords and prompt-injects an explicit
reminder of the matching tool into the system prompt for that turn. It does NOT
execute anything; it only nudges. The model still decides — the hint just makes
the right tool salient.

Covers the default harness tools. `move` also extracts the SAN the user named so
the reminder is concrete ("call move san=b3"), the single most common slip.
"""
from __future__ import annotations

import re

# SAN / castling token the user explicitly named (used for the move hint).
_SAN = re.compile(
    r"\b(O-O-O|O-O|[KQRBN][a-h1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][1-8])\b")
_PLAY = re.compile(r"\b(play|make|do|push|advance|move)\b", re.I)
_CASTLE = re.compile(r"\bcastl", re.I)
_QUEENSIDE = re.compile(r"\b(queenside|long)\b", re.I)

# tool -> (human phrase, canonical call shown, trigger). Order = display order;
# move/best_move are made mutually exclusive in routing_hints().
_TRIGGERS: list[tuple[str, str, str, re.Pattern]] = [
    ("best_move", "find the engine's best move or a hint", "<tool>best_move depth=18</tool>",
     re.compile(r"\b(best move|what should i play|what do i play|what to play|hint|suggest a move|recommend a move|best line|best continuation|strongest move)\b", re.I)),
    ("eval", "evaluate who stands better", "<tool>eval depth=18</tool>",
     re.compile(r"\beval\b|\bevaluat\w*|\bassess\w*|\b(who'?s winning|am i winning|am i losing|how am i doing|how'?s my (game|position)|how do i stand|is (this|it) lost|am i better|am i worse)\b", re.I)),
    ("review_move", "review the move just played", "<tool>review_move depth=15</tool>",
     re.compile(r"\b(was that (a )?(good|ok|bad|blunder)|did i blunder|rate my (last )?move|how was (that|my) move|review (my|the|that) move|good move\??$)\b", re.I)),
    ("threats", "check the opponent's strongest threat", "<tool>threats depth=12</tool>",
     re.compile(r"\b(threat|threats|what'?s the opponent|opponent'?s plan|in danger|under attack|am i attacked|any danger)\b", re.I)),
    ("legal_moves", "list the legal moves", "<tool>legal_moves square=<sq></tool>",
     re.compile(r"\b(legal moves|possible moves|available moves|where can .*(go|move)|what can (this|my) .* (do|move))\b", re.I)),
    ("undo", "take back the last move", "<tool>undo</tool>",
     re.compile(r"\b(undo|take ?back|takeback|revert (that|the|my) move)\b", re.I)),
    ("list_pieces", "list the remaining pieces", "<tool>list_pieces color=<white|black></tool>",
     re.compile(r"\b(what pieces|my pieces|list .* pieces|material count|what do i have left)\b", re.I)),
    ("load_fen", "set up the position from a FEN", "<tool>load_fen fen=<FEN></tool>",
     re.compile(r"\b(load fen|set up (the|this) (position|board)|use this fen)\b|[pnbrqkPNBRQK1-8]{2,}/[pnbrqkPNBRQK1-8/]+ [wb] ", re.I)),
]


def _move_san(msg: str) -> str:
    if _CASTLE.search(msg):
        return "O-O-O" if _QUEENSIDE.search(msg) else "O-O"
    m = _SAN.search(msg)
    return m.group(1) if m else ""


def _move_hint(msg: str) -> tuple[str, str, str] | None:
    """The user named a specific move to play (imperative + SAN, or 'castle')."""
    san = _move_san(msg)
    if _CASTLE.search(msg) or (san and _PLAY.search(msg)):
        call = f"<tool>move san={san}</tool>" if san else "<tool>move san=<SAN></tool>"
        return ("move", f"play the move {san}".strip(), call)
    return None


def routing_hints(user_message: str) -> str:
    """Return a system-prompt addendum reminding the model of the tool(s) the
    user's words map to, or '' if nothing matched. Empty by default — no nudge,
    no change."""
    msg = user_message or ""
    hits: list[tuple[str, str, str]] = []
    mv = _move_hint(msg)
    if mv:
        hits.append(mv)
    for tool, phrase, call, pat in _TRIGGERS:
        if tool == "best_move" and mv:
            continue  # naming a specific move overrides "what should I play"
        if pat.search(msg):
            hits.append((tool, phrase, call))
    if not hits:
        return ""
    lines = [f"- to {phrase}, call `{tool}`: {call}" for tool, phrase, call in hits]
    return ("\n\nROUTING HINT (the user's words map to these tools — call the tool, "
            "do not just describe it; ground your reply in the result):\n" + "\n".join(lines))
