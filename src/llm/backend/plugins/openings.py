"""openings plugin: identify the opening being played and give its typical plans.
Real (deterministic) — matches the longest known opening line against the game's move
history. Reflects a real chess-platform 'opening explorer'. Tests routing: "what
opening is this?" must go to name_opening, NOT eval/best_move."""
from __future__ import annotations

NAME = "openings"

# Longest-match wins. SAN sequence (white POV move list) -> (opening name, one-line plan).
_BOOK: list[tuple[list[str], str, str]] = [
    (["e4", "c5", "Nf3", "d6"], "Sicilian Defence, Najdorf-bound",
     "Black fights for the centre asymmetrically; White often plays d4 then attacks the kingside."),
    (["e4", "c5"], "Sicilian Defence",
     "Black unbalances the game; expect an open d-file and opposite-wing play."),
    (["e4", "e5", "Nf3", "Nc6", "Bb5"], "Ruy Lopez",
     "White pressures the e5 pawn via the c6 knight; plan slow central expansion with c3, d4."),
    (["e4", "e5", "Nf3", "Nc6", "Bc4"], "Italian Game",
     "Both sides develop fast and eye f7; consider c3+d4 or a quiet d3 setup."),
    (["e4", "e5"], "Open Game",
     "Classical king-pawn battle; rapid development and central control decide it."),
    (["e4", "e6"], "French Defence",
     "Black accepts a cramped but solid centre; the c8 bishop is the problem piece."),
    (["e4", "c6"], "Caro-Kann Defence",
     "Solid for Black with a sound structure; White grabs space and plays for the centre."),
    (["d4", "d5", "c4"], "Queen's Gambit",
     "White offers the c-pawn to deflect Black's centre; classical, strategic play."),
    (["d4", "Nf6", "c4", "g6"], "King's Indian Defence",
     "Black cedes the centre then strikes with ...e5 and a kingside pawn storm."),
    (["d4", "Nf6"], "Indian Defence",
     "Black delays ...d5, controlling the centre with pieces; flexible and modern."),
    (["d4", "d5"], "Closed Game",
     "Queen-pawn structures; manoeuvring and pawn breaks (c4/e4 vs ...c5/...e5) matter."),
    (["c4"], "English Opening",
     "Flank strategy; White fights for d5 and often transposes to a reversed Sicilian."),
    (["Nf3"], "Réti / King's Indian Attack",
     "Hypermodern: develop first, strike the centre later."),
]

TOOLS = [
    {"name": "name_opening", "description": "Identify the opening being played from the move history.",
     "args": {}, "applies_when": "has_history"},
    {"name": "opening_ideas", "description": "Give the typical plans and ideas for the current opening.",
     "args": {}, "applies_when": "has_history"},
]

SKILLS = [{
    "name": "opening-advisor",
    "description": "Use when the user asks what opening this is, or for opening plans, theory, or a repertoire.",
    "body": ("---\nname: opening-advisor\ndescription: Opening identification and plans.\n---\n\n"
             "# opening-advisor\n\nWhen the user asks about the opening, act this turn — don't ask which "
             "opening they mean. Call `name_opening` to identify it, then `opening_ideas` for the typical "
             "plans, and explain in one or two plain sentences. Never invent an opening name the tool did "
             "not return."),
}]


def _identify(san_stack: list[str]) -> tuple[str, str] | None:
    for seq, name, plan in _BOOK:
        if san_stack[:len(seq)] == seq:
            return name, plan
    return None


def handle(name: str, args: dict, executor) -> str | None:
    if name not in ("name_opening", "opening_ideas"):
        return None
    stack = list(executor.game.san_stack)
    if not stack:
        return "opening: none yet — no moves played."
    hit = _identify(stack)
    if not hit:
        return "opening: unrecognised line (out of book)."
    opening, plan = hit
    if name == "name_opening":
        return f"opening: {opening}"
    return f"opening_ideas: {opening} — {plan}"
