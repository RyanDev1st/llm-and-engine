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

# A bare move token (SAN, castling, or UCI) for the move-arg coercion below. Same shape as
# tool_hints._SAN/_UCI — kept local so toolfmt stays dependency-free.
_MOVE_TOK = re.compile(
    r"(O-O-O[+#]?|O-O[+#]?|[KQRBN][a-h1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h][1-8][a-h][1-8][qrbnQRBN]?)")


def parse_call(tool_call: str) -> tuple[str | None, dict[str, str]]:
    m = _CALL.search(tool_call.strip())
    if not m:
        return None, {}
    name = m.group(1)
    rest = m.group(2).strip()
    # One free-text arg per tool captures the rest of the call (it may contain
    # spaces / '='): ask_chessbot's query=, load_fen's fen=. Keyed BY TOOL so a
    # 'fen=' inside a question (or odd chars in a FEN) can't truncate the other.
    # Case-INSENSITIVE match so QUERY=/Fen=/CODE= slips don't fall through.
    free = {"ask_chessbot": "query=", "load_fen": "fen=", "python": "code="}.get(name)
    if free:
        idx = rest.lower().find(free)
        if idx != -1:
            head, tail = rest[:idx], rest[idx + len(free):]
            args = _kv(head)
            args[free[:-1]] = tail.strip()
            return name, args
    args = _kv(rest)
    # move arg coercion: `<tool>move e4</tool>` / `<tool>move Nf3</tool>` — the model's single
    # most common slip is omitting `san=`. When move has no san but a bare move-like token sits
    # in the call, fill it, so a clearly-valid move runs instead of bouncing to a corrective error
    # (which the model then sometimes mishandles into a broken reply). UCI/SAN both handled by move.
    if name == "move" and not args.get("san"):
        mv = _MOVE_TOK.fullmatch(rest)     # ONLY a clean single token (`move e4`); 'move rook f8'
        if mv:                             # (multi-word) stays a corrective error, not a wrong coerce
            args["san"] = mv.group(1)
    return name, args


def _kv(text: str) -> dict[str, str]:
    # Arg KEYS are case-folded (every catalog arg is lowercase: san, fen, depth, color, …) so a
    # `move SAN=Rd1#` / `eval DEPTH=12` slip resolves instead of bouncing to a corrective error.
    # VALUES keep their case (SAN like 'Rd1#'/'Nf3' is case-sensitive: B=bishop vs b=file).
    out: dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip().lower()] = v
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
