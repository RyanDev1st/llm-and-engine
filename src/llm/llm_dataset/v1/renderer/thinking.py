"""Grounded inline reasoning (`<think>…</think>`) prepended to each assistant step, so
the model learns the harness DECISION PROCESS — goal -> state -> route/verify -> act —
generalizing across ANY skill/tool, not memorizing one. This is the trained "thinking
loop": before acting it states what the user wants, what it already has, and why it
picks the next tool (or why it answers now).

Hard rule: the think NEVER contains facts (no eval numbers, no SAN, no claims) — only
intent + plan + state. So it can't fabricate, and the grounding validators (which scan
for numbers/moves) ignore it. Seeded variation keeps it from being one rote sentence.

Schema (compact): state (what I have) ; decision (next tool by fit, or answer). The
OBJECTIVE is committed ONCE in the leading <goal> tag (open_goal) at the top of a
thinking-mode turn, not restated every step — so the goal is held across the loop
(anti-early-stop) and multi-step rows stay under the seq ceiling. Serve strips <think>
from the visible reply (the 'thinking' panel) and <goal> to the plan panel."""
from __future__ import annotations

import random

from .planning import goal_block

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
    "python": ("verify this with a script before I claim it, not guess the number",
               "let the tool settle it — run the numbers, don't assert them",
               "check the real figure by running it instead of trusting my head"),
}
_HAVE = {
    0: ("nothing gathered yet", "haven't read anything yet", "starting fresh"),
    "skill": ("skill is loaded", "have the skill's guidance now"),
    "board": ("board is read", "have the position now"),
    "results": ("I have the engine's result", "the tool result is in hand"),
}


def think(seed: int, action: str, step: int = 0, *, goal: str = "", have: str = "") -> str:
    """`<think>state ; decision</think>` before a tool step (action=tool name).
    `have` is a short state key (''/'skill'/'board'/'results'). No facts ever. The
    objective is NOT restated here — it lives in the leading <goal> tag (open_goal);
    `goal` is kept for signature compat with the callers."""
    r = random.Random(seed * 101 + step)
    state = r.choice(_HAVE[have]) if have in _HAVE else r.choice(_HAVE[0])
    decide = r.choice(_DECIDE.get(action, (f"call {action}",)))
    return "<think>" + "; ".join((state, f"so I'll {decide}")) + ".</think>"


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


# --- leading <goal> (thinking modes only) ------------------------------------
# The objective is committed ONCE per turn in a <goal> tag, for the reasoning
# modes (think/auto/plan) — NOT fast. Fast is the snappy single-shot path with no
# early-stop problem, so it stays bare. Promoted out of per-step <think> (see
# think()), so the goal is held across the loop (anti-early-stop) at ~neutral seq.
_GOAL_MODES = frozenset({"think", "auto", "plan"})


def open_goal(seed: int, mode: str, goal) -> str:
    """Leading `<goal>…</goal>` for a thinking-mode turn (think/auto/plan); '' for
    fast. `goal` is a str (one objective) or a list/tuple (a compound request —
    every ask enumerated). Routes to the plan panel at serve time."""
    if (mode or "").strip().lower() not in _GOAL_MODES:
        return ""
    return goal_block(seed, goal)


def prepend_open_goal(messages: list, seed: int, mode: str, goal) -> list:
    """Prepend the leading `<goal>` to the FIRST trained assistant turn (skips
    train:False context turns, e.g. multi-turn turn 1). No-op for fast mode and
    when the goal is empty. Mutates and returns `messages`."""
    g = open_goal(seed, mode, goal)
    if not g:
        return messages
    for m in messages:
        if m.get("role") == "assistant" and m.get("train", True):
            m["content"] = g + "\n" + m["content"]
            break
    return messages


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


def think_direct(seed: int) -> str:
    """`<think>` for answering directly when NO listed skill fits (V1_Q). A clean,
    natural self-check — not the goal-substituted gated_answer template, which read
    awkwardly here ('time to answer reply directly … directly')."""
    r = random.Random(seed * 109 + 3)
    return "<think>" + r.choice((
        "no listed skill fits this — answer directly from what I know",
        "nothing in the skill list matches; just reply directly",
        "this needs no skill or tool — answer it straight",
        "none of the available skills apply here, so I'll answer plainly")) + ".</think>"


def gated_direct(seed: int, *, mode: str) -> str:
    """Direct-answer `<think>` (kind='answer'): present in think/auto, '' in fast."""
    return think_direct(seed) if _emit(mode, "answer") else ""
