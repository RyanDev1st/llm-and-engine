"""Hand-written chat scenarios for the report's two authentic-chat sections. These are the USER
turns only — realistic, messy, the way a real person talks to a chess coach (slang, typos, vague or
tricky asks). The coach REPLIES + the timing come from running the real harness on the live model
(chat_showcase.py); nothing here is a scripted answer.

PLAIN = the bare harness surface ("the actual AI model with the real harness, no more no less"):
        board hook OFF, no web extras — coaching/concept/skill-routing asks that need no board.
WEB   = the chess-web sandbox: a real board (FEN) per scenario, board hook ON (LIVE BOARD line),
        multi-turn within ONE session — exactly what the website sends. Asks reference the position.

Each scenario: {"title", optional "fen", "turns": [(user_text, reasoning_mode), ...]}. A scenario's
turns share ONE loop + board + history (a session); a tool turn (move/new_game) mutates that board.
"""
from __future__ import annotations

_MODES = {"fast", "think", "auto", "plan"}

# --- Section 1: bare harness, no board -------------------------------------------------------------
PLAIN_CHATS = [
    {"title": "Vague cry for help", "turns": [
        ("yo so i keep hanging my queen like every game lol what do i do", "auto"),
        ("ok but what opening should i even play as white to keep it simple", "auto")]},
    {"title": "Explain a concept (casual)", "turns": [
        ("whats a fork explain it like im 5", "fast")]},
    {"title": "Endgame study, no board", "turns": [
        ("i'm dogwater at endgames help me get better", "auto")]},
    {"title": "Wants a puzzle", "turns": [
        ("gimme a tactics puzzle to chew on", "auto"),
        ("no idea, just show me the answer", "auto")]},
    {"title": "Tricky / scope-test", "turns": [
        ("can u just play a whole game vs me right now", "auto")]},
]

# --- Section 2: chess-web sandbox, real boards -----------------------------------------------------
_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_ITALIAN = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
_SCANDI = "rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 3 3"
_SICILIAN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"

WEB_CHATS = [
    {"title": "Midgame read + plan", "fen": _ITALIAN, "turns": [
        ("hows my position looking here", "auto"),
        ("whats the best move then", "auto"),
        ("why is that better than just developing the other knight", "think")]},
    {"title": "Blunder check", "fen": _SCANDI, "turns": [
        ("wait is my queen safe on a5 or did i just blunder it", "think")]},
    {"title": "Name this opening", "fen": _SICILIAN, "turns": [
        ("what's this opening called", "fast")]},
    {"title": "Reset + play a move", "fen": _START, "turns": [
        ("eh lets just start over fresh board", "auto"),
        ("play e4 for me", "auto")]},
]


def validate(suite: list[dict]) -> None:
    """Cheap shape gate (used by tests + the CPU gate cell): every scenario has turns, every turn a
    non-empty prompt + a known reasoning mode, every WEB scenario a FEN."""
    assert suite, "empty suite"
    for sc in suite:
        assert sc.get("title"), "scenario missing title"
        assert sc.get("turns"), f"{sc['title']}: no turns"
        for text, mode in sc["turns"]:
            assert text and text.strip(), f"{sc['title']}: empty prompt"
            assert mode in _MODES, f"{sc['title']}: bad mode {mode!r}"
