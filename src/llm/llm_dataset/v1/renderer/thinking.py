"""Grounded inline reasoning (`<think>…</think>`) prepended to each assistant step, so
the model learns the harness DECISION PROCESS — goal -> state -> route/verify -> act —
generalizing across ANY skill/tool, not memorizing one. This is the trained "thinking
loop": before acting it states what the user wants, what it already has, and why it
picks the next tool (or why it answers now).

Hard rule: the think NEVER contains facts (no eval numbers, no SAN, no claims) — only
intent + plan + state. So it can't fabricate, and the grounding validators (which scan
for numbers/moves) ignore it. Seeded variation keeps it from being one rote sentence.

Schema (compact): goal (what the user wants) ; state (what I have) ; decision (next tool
by fit, or answer). Serve strips <think> from the visible reply (the 'thinking' panel)."""
from __future__ import annotations

import random

# Why each step happens, keyed by the tool being called — PLAN language, no facts.
_DECIDE = {
    "load_skill": ("load the skill that fits this", "pull the matching skill first",
                   "bring up the right skill before acting"),
    "board_state": ("read the board — I can't see it, so no claims until I do",
                    "check the live position first", "ground myself in the board before judging"),
    "move": ("play the move they named", "send that move to the board"),
    "eval": ("ask the engine where this stands", "get the evaluation to answer that"),
    "best_move": ("pull the engine's best line", "get candidate moves from the engine"),
    "review_move": ("grade the move that was just played", "score that last move"),
    "threats": ("check the opponent's best threat", "see what they're threatening"),
    "legal_moves": ("list the legal moves there", "see what's actually legal"),
    "list_pieces": ("read the remaining pieces off the board", "list the material"),
    "ask_chessbot": ("look that up in the knowledge base", "pull that general fact"),
    "normalize_human_chat": ("normalize the messy phrasing before I route it",
                             "clean up what they said so I route it right"),
}
_HAVE = {
    0: ("nothing gathered yet", "haven't read anything yet", "starting fresh"),
    "skill": ("skill is loaded", "have the skill's guidance now"),
    "board": ("board is read", "have the position now"),
    "results": ("I have the engine's result", "the tool result is in hand"),
}


def _goal(seed: int, goal: str) -> str:
    g = (goal or "help with the position").strip().rstrip("?.!")
    return random.Random(seed * 7 + 1).choice(
        (f"they want me to {g}", f"the ask is to {g}", f"goal: {g}"))


def think(seed: int, action: str, step: int = 0, *, goal: str = "", have: str = "") -> str:
    """`<think>goal ; state ; decision</think>` before a tool step (action=tool name).
    `have` is a short state key (''/'skill'/'board'/'results'). No facts ever."""
    r = random.Random(seed * 101 + step)
    state = r.choice(_HAVE[have]) if have in _HAVE else r.choice(_HAVE[0])
    decide = r.choice(_DECIDE.get(action, (f"call {action}",)))
    parts = [_goal(seed, goal), state, f"so I'll {decide}"]
    return "<think>" + "; ".join(parts) + ".</think>"


def think_fix(seed: int, action: str) -> str:
    """`<think>` for SELF-CORRECTION: the previous call errored — diagnose and retry.
    Teaches recovery instead of giving up or fabricating a result."""
    r = random.Random(seed * 107 + 5)
    return "<think>" + r.choice(
        (f"that call errored — fix the arguments and retry {action}",
         f"the tool rejected my input; correct it and call {action} again",
         f"bad call last step — adjust and re-run {action}, don't fabricate a result")) + ".</think>"


def think_answer(seed: int, goal: str = "", *, enough: bool = True) -> str:
    """`<think>` before the FINAL reply: verify the goal is met, decide to answer (not
    call another tool). This is the self-check that stops 'rookie' re-routing."""
    r = random.Random(seed * 103 + 9)
    g = (goal or "their question").strip().rstrip("?.!")
    if enough:
        return "<think>" + r.choice(
            (f"I have what I need to {g} — answer now, no more tools",
             f"that covers it; time to answer {g} directly",
             f"goal met — reply to {g}, don't call anything else")) + ".</think>"
    return "<think>" + r.choice(
        (f"already covered {g} earlier — reference it, don't re-run a tool",
         f"nothing new needed for {g}; answer from what's established")) + ".</think>"


# --- reasoning MODE gating (fast / think / auto) -----------------------------
# The model is taught three behaviors via a system-prompt signal, so fast/think
# is a real toggle (not always-on): THINK = <think> on every reasoning step;
# FAST = no <think> anywhere (snappy, direct); AUTO = <think> ONLY before a hard
# decision (routing/selection, recovery, deciding you're done) — interleaved, so
# the model learns WHEN to think, not just to obey a flag. Same slices appear in
# all three modes so the SIGNAL, not the slice, drives the behavior.
MODES = ("fast", "think", "auto")
# Step kinds that count as a "hard decision" -> AUTO keeps the <think>.
# "routine" (forced read, named move, rote fetch) is skipped in AUTO.
_AUTO_THINK_KINDS = frozenset({"select", "decide", "recover", "answer", "clarify"})
# Rote chain execution after a plan is already made: NEVER thinks, in ANY mode
# (even "think"). A budget/multi-tool task reasons once up front, then executes
# the planned fetches silently — this is the budget lesson and keeps the longest
# slice under the train seq ceiling. Distinct from "routine" (skipped only in AUTO).
_NEVER_THINK_KINDS = frozenset({"execute"})


def pick_mode(seed: int) -> str:
    """Per-row mode, seeded + independent of slice. ~35% fast / 40% auto / 25% think."""
    r = (seed * 991 + 7) % 20
    return "fast" if r < 7 else ("auto" if r < 15 else "think")


def _emit(mode: str, kind: str) -> bool:
    if kind in _NEVER_THINK_KINDS:
        return False                   # rote chain execution: silent in every mode
    if mode == "fast":
        return False
    if mode == "think":
        return True
    return kind in _AUTO_THINK_KINDS  # auto


def gated_think(seed: int, action: str, step: int, *, mode: str, kind: str,
                goal: str = "", have: str = "") -> str:
    """`<think>` for a tool step, gated by mode+kind. '' when suppressed."""
    return think(seed, action, step, goal=goal, have=have) if _emit(mode, kind) else ""


def gated_fix(seed: int, action: str, *, mode: str) -> str:
    """Self-correction `<think>` (kind='recover'): present in think+auto, '' in fast."""
    return think_fix(seed, action) if _emit(mode, "recover") else ""


def gated_answer(seed: int, goal: str = "", *, mode: str, enough: bool = True) -> str:
    """Final-step `<think>` (kind='answer', the goal-met self-check): '' only in fast."""
    return think_answer(seed, goal, enough=enough) if _emit(mode, "answer") else ""
