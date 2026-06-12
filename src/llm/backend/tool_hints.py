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

# Benign fillers allowed between a request verb/count and "moves" (any run, incl. none).
# WHITELIST only — so "legal/possible/available moves" is NOT swallowed by best_move and
# stays routed to legal_moves. Keeps the matcher flexible without over-firing.
_FILL = (r"(?:the\s+|a\s+|some\s+|my\s+|\d+\s+|next\s+|best\s+|good\s+|top\s+"
         r"|consecutive\s+|few\s+|other\s+|more\s+|several\s+|strong\s+|candidate\s+)*")

# tool -> (human phrase, canonical call shown, trigger). Order = display order;
# move/best_move are made mutually exclusive in routing_hints().
_TRIGGERS: list[tuple[str, str, str, re.Pattern]] = [
    ("best_move", "find the engine's best move or a hint", "<tool>best_move depth=18</tool>",
     re.compile(
         r"\b(best moves?|candidate moves?|strongest moves?|best line|best continuation"
         r"|what should i play|what do i play|what to play|hint"
         r"|top \d+ moves?"                                          # "top 5 moves"
         # a request verb (or a count) followed by any run of benign fillers then "moves".
         # The filler class is whitelisted (the/a/some/N/next/best/good/consecutive/few/...)
         # so it does NOT swallow "legal/possible/available moves" -> those stay legal_moves.
         r"|(?:suggest|recommend|show|give|list|play)\s+(?:me\s+)?" + _FILL + r"moves?"
         r"|\d+\s+" + _FILL + r"moves?)\b",                          # "3 consecutive moves"
         re.I)),
    ("eval", "evaluate who stands better", "<tool>eval depth=18</tool>",
     re.compile(
         r"\beval\b|\bevaluat\w*|\bassess\w*"
         r"|\b(who'?s winning|am i winning|am i losing|how am i doing|how'?s my (game|position)"
         r"|how do i stand|is (this|it) lost|am i better|am i worse)\b"
         # casual / slang self-assessment: "am I doing bad", "am I fucked/screwed/cooked",
         # "how's it looking", "is my game good", "rate my position", "doing well?"
         r"|\bam i (doing |playing )?(bad|badly|good|well|ok|okay|fine|poorly|great|terrible"
         r"|cooked|fucked|screwed|toast|done|lost|winning|losing)\b"
         r"|\b(how'?s|how is) (it|my game|my position|this) (going|looking)\b"
         r"|\bis my (game|position) (good|bad|ok|okay|fine|winning|losing|lost)\b"
         r"|\brate my (game|position|standing)\b"
         r"|\b(doing|playing) (bad|badly|well|good|ok|poorly)\??$", re.I)),
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
    ("random_position", "set up a fresh position (puzzle/scramble/random)", "<tool>random_position kind=puzzle</tool>",
     re.compile(r"\b(give me a puzzle|make me a puzzle|a tactical puzzle|generate a puzzle|new puzzle"
                r"|random (position|board|fen|game)|scramble (the|this)? ?(board|position)?"
                r"|set up a (random|tactical) position|practice a tactic)\b", re.I)),
]


# Generic words that appear in many skill names / the always-on coach skill —
# matching on these would fire the skill hint on almost every chess turn. We key
# only on DISTINCTIVE name tokens, so a broad skill (chess-coach) stays silent
# while a specialised drop-in (tactical-puzzles, endgame-drills) fires when named.
_SKILL_STOP = {"chess", "coach", "skill", "official", "plugin", "move", "moves",
               "game", "play", "board", "position", "help", "assistant", "tool"}


def skill_hints(user_message: str, skills: list[dict]) -> str:
    """Deterministic skill-routing layer. If the user's words contain a distinctive
    token from an installed skill's name, remind the model to load that skill via
    the load_skill protocol (progressive disclosure) instead of answering blind.
    Default silent — only fires on a clearly named specialised skill, so it never
    over-triggers on the broad always-loaded coach skill. Generalises to any
    dropped-in SKILL.md (no per-skill code)."""
    msg = (user_message or "").lower()
    hits: list[tuple[str, str]] = []
    for s in skills:
        tokens = [t for t in re.split(r"[-_\s]+", str(s.get("name", "")).lower())
                  if len(t) >= 4 and t not in _SKILL_STOP]
        # drop a trailing plural 's' and match by prefix so "puzzles" fires on
        # "puzzle" and vice versa (deterministic, no full stemmer needed).
        stems = [t[:-1] if t.endswith("s") and len(t) > 4 else t for t in tokens]
        if any(re.search(r"\b" + re.escape(st), msg) for st in stems):
            hits.append((s["name"], s.get("description", "")))
    if not hits:
        return ""
    lines = [f"- for this, load the `{name}` skill: <tool>load_skill name={name}</tool>"
             f"  ({desc})" for name, desc in hits]
    return ("\n\nSKILL HINT (the user's request matches an installed skill — load it "
            "first with load_skill, then follow what it tells you):\n" + "\n".join(lines))


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


def routing_hints(user_message: str, game_over: str = "") -> str:
    """Return a system-prompt addendum reminding the model of the tool(s) the
    user's words map to, or '' if nothing matched. Empty by default — no nudge,
    no change. `game_over` (e.g. "checkmate"/"stalemate"/"draw") short-circuits to
    a state hint so the model states the result instead of calling analysis tools
    on a finished game."""
    if game_over:
        return ("\n\nGAME STATE: the game is over (" + game_over + "). Do NOT call "
                "analysis tools — state the result plainly and offer a new game.")
    hits = _match(user_message or "")
    if not hits:
        return ""
    lines = [f"- to {phrase}, call `{tool}`: {call}" for tool, phrase, call in hits]
    return ("\n\nROUTING HINT (the user's words map to these tools — call the tool, "
            "do not just describe it; ground your reply in the result):\n" + "\n".join(lines))


def _match(msg: str) -> list[tuple[str, str, str]]:
    """The intent matches as (tool, human phrase, canonical call). Shared by
    routing_hints (formats them) and matched_calls (the coverage set)."""
    hits: list[tuple[str, str, str]] = []
    mv = _move_hint(msg)
    if mv:
        hits.append(mv)
    for tool, phrase, call, pat in _TRIGGERS:
        if tool == "best_move" and mv:
            continue  # naming a specific move overrides "what should I play"
        if pat.search(msg):
            hits.append((tool, phrase, call))
    return hits


_COUNT = re.compile(r"\b([2-5])\s+" + _FILL + r"moves?\b"     # "3 moves", "3 consecutive moves"
                    r"|\b(?:next|top)\s+([2-5])\b", re.I)    # "next 3", "top 3"
_WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5}
# "consecutive / in a row / a line / the sequence / continuation" => the user wants a
# LINE (move, reply, move...) => best_move series=N, NOT N alternative candidate moves.
_LINE = re.compile(r"\b(consecutive|in a row|sequence|continuation|line|"
                   r"principal variation|next few moves)\b", re.I)


def _move_count(msg: str) -> int | None:
    """The number of moves the user asked for ("5 next best moves" / "top 3"), 2-5."""
    m = _COUNT.search(msg)
    if m:
        return int(m.group(1) or m.group(2))
    for word, n in _WORDNUM.items():
        if re.search(rf"\b{word}\s+(?:next\s+|best\s+|good\s+|consecutive\s+)*moves?\b", msg, re.I):
            return n
    return None


def matched_calls(user_message: str) -> dict[str, str]:
    """tool name -> the canonical `<tool>…</tool>` call the user's words map to.
    The deterministic coverage set: every detected intent must be gathered before
    the loop narrates. For best_move, honor the requested COUNT and distinguish a LINE
    (consecutive moves -> series=N) from ALTERNATIVES (N candidate moves -> top=N)."""
    msg = user_message or ""
    calls = {tool: call for tool, _phrase, call in _match(msg)}
    if "best_move" in calls:
        n = _move_count(msg)
        if _LINE.search(msg):                      # consecutive line: move -> reply -> move
            calls["best_move"] = f"<tool>best_move depth=18 series={n or 3}</tool>"
        elif n:                                     # N alternative candidate moves
            calls["best_move"] = f"<tool>best_move depth=18 top={n}</tool>"
    return calls


def matched_tools(user_message: str) -> set[str]:
    return set(matched_calls(user_message))
