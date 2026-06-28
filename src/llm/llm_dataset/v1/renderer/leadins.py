"""Conversational shape, shared by the chess and universality renderers: a
short lead-in sentence streams before each tool call (so the user isn't waiting
on a silent model), and coaching/analysis finals end with one guiding question
that suggests what to ask next — the way a coding agent narrates its work."""
from __future__ import annotations

import random

LEAD = {
    "load_skill": ("Let me load the skill that fits this.", "First, the right coaching skill.",
                   "Loading the matching skill now."),
    "board_state": ("First, the current position.", "Let me read the board.",
                    "Checking the board before I say anything."),
    "move": ("Playing that now.", "Let me make the move.", "Sending it to the board."),
    "eval": ("Now the engine's read.", "Let me check the evaluation.", "Asking the engine where this stands."),
    "best_move": ("Let me pull the engine's line.", "Now the candidate moves.", "Getting the best continuation."),
    "review_move": ("Let me grade that move.", "Reviewing what you just played.", "Checking how that move scored."),
    "threats": ("Let me check the threats.", "Now your opponent's ideas.", "Scanning for what they're threatening."),
    "legal_moves": ("Let me list the legal moves there.", "Checking what's legal here.", "Listing your options."),
    "list_pieces": ("Let me list your pieces from the board.", "Reading the material off the board.",
                    "Checking what's still on."),
    "ask_chessbot": ("Let me look that up.", "Checking the knowledge base.", "Pulling that from the chess KB."),
    "python": ("Let me compute that exactly.", "Running a quick script now.", "Let me get the exact figure with Python."),
}

GUIDING = (
    "Want me to map the plan to convert it, or check your opponent's threats first?",
    "Should I go deeper on the main line, or look at the alternatives?",
    "Want the follow-up plan, or your opponent's best reply first?",
    "Want me to keep going from here, or focus on one square?",
    "Should I show the next few moves, or explain the idea behind it?",
)


def lead(seed: int, action: str, i: int = 0) -> str:
    return random.Random(seed * 31 + i).choice(LEAD.get(action, ("",)))


def guiding(seed: int, i: int = 0) -> str:
    return random.Random(seed * 17 + i).choice(GUIDING)


def ask(text: str, seed: int, i: int = 0) -> str:
    """Append a guiding question to a coaching final (idempotent on '?')."""
    text = text.rstrip()
    if text.endswith("?"):
        return text
    return f"{text} {guiding(seed, i)}"
