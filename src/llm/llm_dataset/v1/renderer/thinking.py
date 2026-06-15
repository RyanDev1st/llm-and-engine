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
# Each pool is paraphrase-rich so the thinking panel never shows one rote sentence
# thousands of times (memorization). Every phrase stays fact-free (intent/state only)
# and short (<= the longest legacy phrase, to keep the seq ceiling unmoved).
_DECIDE = {
    "load_skill": ("load the skill that fits this", "pull the matching skill first",
                   "bring up the right skill before acting", "open the skill whose description matches",
                   "grab the skill that covers this request", "reach for the fitting skill before I act"),
    "board_state": ("read the board — I can't see it, so no claims until I do",
                    "check the live position first", "ground myself in the board before judging",
                    "snapshot the position before I say anything", "look at the board before any claim"),
    "move": ("play the move they named", "send that move to the board", "make the move they asked for"),
    "eval": ("ask the engine where this stands", "get the evaluation to answer that",
             "let the engine score the position", "pull the eval before I judge it"),
    "best_move": ("pull the engine's best line", "get candidate moves from the engine",
                  "ask the engine for the strongest move", "see what the engine prefers here"),
    "review_move": ("grade the move that was just played", "score that last move",
                    "have the engine check that move", "judge the move against the engine"),
    "threats": ("check the opponent's best threat", "see what they're threatening",
                "find the threat before I answer", "look for what's hanging first"),
    "legal_moves": ("list the legal moves there", "see what's actually legal",
                    "pull the legal moves before guessing", "check which moves are allowed"),
    "list_pieces": ("read the remaining pieces off the board", "list the material",
                    "count what's left on the board", "see the pieces before I judge material"),
    "ask_chessbot": ("look that up in the knowledge base", "pull that general fact",
                     "check the reference before answering", "get the fact from the KB, not my head"),
    "normalize_human_chat": ("normalize the messy phrasing before I route it",
                             "clean up what they said so I route it right",
                             "untangle the slang before picking a skill",
                             "restate this clearly before routing"),
    "python": ("verify this with a script before I claim it, not guess the number",
               "let the tool settle it — run the numbers, don't assert them",
               "check the real figure by running it instead of trusting my head",
               "compute it for real rather than estimating", "run it to get the exact value, no guessing"),
}
# Generic decision verbs for any tool not in _DECIDE (the cross-domain tools:
# validate_json, regex_test, sum_spend, diff_view, search_kb, ...). Keeps those
# steps from all collapsing onto one "call X" string.
_DECIDE_GENERIC = ("call {a} to get that", "run {a} for the detail I need",
                   "use {a} to settle this", "check it with {a}", "lean on {a} for the specifics",
                   "let {a} surface the answer")
_HAVE = {
    0: ("nothing gathered yet", "haven't read anything yet", "starting fresh", "no context yet",
        "nothing in hand so far", "blank slate right now", "no results read yet",
        "I've pulled nothing yet"),
    "skill": ("skill is loaded", "have the skill's guidance now", "the skill body is in front of me",
              "guidance from the skill is loaded", "skill steps are in hand now"),
    "board": ("board is read", "have the position now", "the position is in front of me",
              "I've read the live board"),
    "results": ("I have the engine's result", "the tool result is in hand", "the result came back",
                "got the output I called for"),
}


def think(seed: int, action: str, step: int = 0, *, goal: str = "", have: str = "") -> str:
    """`<think>state ; decision</think>` before a tool step (action=tool name).
    `have` is a short state key (''/'skill'/'board'/'results'). No facts ever. The
    objective is NOT restated here — it lives in the leading <goal> tag (open_goal);
    `goal` is kept for signature compat with the callers."""
    r = random.Random(seed * 101 + step)
    state = r.choice(_HAVE[have]) if have in _HAVE else r.choice(_HAVE[0])
    options = _DECIDE.get(action) or tuple(g.format(a=action) for g in _DECIDE_GENERIC)
    decide = r.choice(options)
    return "<think>" + "; ".join((state, f"so I'll {decide}")) + ".</think>"


def think_fix(seed: int, action: str) -> str:
    """`<think>` for SELF-CORRECTION: the previous call errored — diagnose and retry.
    Teaches recovery instead of giving up or fabricating a result."""
    r = random.Random(seed * 107 + 5)
    return "<think>" + r.choice(
        (f"that call errored — fix the arguments and retry {action}",
         f"the tool rejected my input; correct it and call {action} again",
         f"bad call last step — adjust and re-run {action}, don't fabricate a result",
         f"wrong args that time — repair them and call {action} once more",
         f"that failed; read the error, fix the call, retry {action}",
         f"my input was off — adjust and re-run {action}, no faking the output",
         f"the call bounced — correct the arguments and try {action} again",
         f"error on that step; diagnose it, then re-issue {action} properly")) + ".</think>"


def think_answer(seed: int, goal: str = "", *, enough: bool = True) -> str:
    """`<think>` before the FINAL reply: verify the goal is met, decide to answer (not
    call another tool). This is the self-check that stops 'rookie' re-routing. Goal-FREE:
    the objective is already committed in the leading <goal> tag (open_goal), so restating
    it here is redundant AND ungrammatical (goals are verb phrases — 'time to answer decide
    between the options' broke). `goal` kept for call-site compat, unused."""
    r = random.Random(seed * 103 + 9)
    if enough:
        return "<think>" + r.choice(
            ("I have what I need — answer now, no more tools",
             "that covers it; time to answer directly",
             "goal met — reply now, don't call anything else",
             "enough gathered — write the answer, no more calls",
             "got the result I needed; answer from it now",
             "that's the piece I was missing — reply now",
             "no gap left; answer directly instead of calling again",
             "the result settles it — respond, don't re-run",
             "I can answer this now without another tool",
             "everything needed is in hand — reply",
             "done gathering; time to give the answer",
             "that's sufficient — answer straight, no extra calls",
             "the tool gave what I needed; answer now",
             "nothing else to fetch — write the reply",
             "I'm set — answer the question directly",
             "have the basis to answer now; reply, don't loop")) + ".</think>"
    return "<think>" + r.choice(
        ("already covered this earlier — reference it, don't re-run a tool",
         "nothing new needed; answer from what's established",
         "this was settled earlier — point back to it, no new call",
         "I already have this from before; reuse it, don't re-fetch",
         "no fresh tool needed — the earlier result still holds",
         "answer from what's already on the table",
         "covered above — restate it, skip another call",
         "the prior step answers this; lean on it")) + ".</think>"


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
        "none of the available skills apply here, so I'll answer plainly",
        "general enough to answer without loading anything",
        "the list has nothing for this — answer from what I know",
        "skip the skills here; a direct reply fits",
        "nothing to load for this one — answer plainly",
        "no tool or skill applies; I'll just respond",
        "this is straightforward — answer it without routing")) + ".</think>"


def gated_direct(seed: int, *, mode: str) -> str:
    """Direct-answer `<think>` (kind='answer'): present in think/auto, '' in fast."""
    return think_direct(seed) if _emit(mode, "answer") else ""
