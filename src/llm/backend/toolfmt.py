"""Parsing of `<tool>NAME arg=value ...</tool>` calls and spec score formatting.

The final `query=` arg (ask_chessbot) is free text and may contain spaces, so it
captures the rest of the line; all other args are single tokens."""
from __future__ import annotations

import re

# Name allows hyphens/digits so a skill called as a tool (<tool>chess-coach</tool>,
# <tool>plugin-echo-73</tool>) parses whole instead of truncating at the hyphen;
# real tool names are a subset (a-z + _). The dispatcher then routes a known
# skill name to load_skill rather than dead-ending.
_CALL = re.compile(r"<tool>\s*([a-z0-9_-]+)(.*?)</tool>", re.DOTALL)


def parse_call(tool_call: str) -> tuple[str | None, dict[str, str]]:
    m = _CALL.search(tool_call.strip())
    if not m:
        return None, {}
    name = m.group(1)
    rest = m.group(2).strip()
    # One free-text arg per tool captures the rest of the call (it may contain
    # spaces / '='): ask_chessbot's query=, load_fen's fen=. Keyed BY TOOL so a
    # 'fen=' inside a question (or odd chars in a FEN) can't truncate the other.
    free = {"ask_chessbot": "query=", "load_fen": "fen="}.get(name)
    if free and free in rest:
        head, tail = rest.split(free, 1)
        args = _kv(head)
        args[free[:-1]] = tail.strip()
        return name, args
    return name, _kv(rest)


def _kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def clamp_depth(args: dict[str, str], default: int) -> int:
    try:
        d = int(args.get("depth", default))
    except ValueError:
        d = default
    return max(8, min(20, d))


def fmt_white_score(kind: str, val, depth: int) -> str:
    if kind == "mate":
        side, n = val
        return f"score: mate in {n} for {side}, depth={depth}"
    return f"score: {int(val) / 100:+.2f} pawns from white POV, depth={depth}"
