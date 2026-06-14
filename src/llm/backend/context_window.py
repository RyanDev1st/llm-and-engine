"""Token-budgeted context window — the model's session memory manager.

The serving prompt is `system + conversation history + the new user turn`, and a
transformer can only attend to a fixed number of tokens (`n_ctx`). Left
unbounded, a long chat overflows the window: the backend silently drops the
oldest tokens (the model "forgets" mid-session and contradicts itself) or errors
out. This module keeps the prompt inside a token budget by evicting the OLDEST
turns first, while always retaining the system prompt and the current user turn —
so recent context survives and the window never overflows.

Drop-oldest alone loses the thread. So when `compress=True`, the evicted prefix is
not discarded silently: `compact()` distills it into ONE small memory note —
pinning the objective (`<goal>`/`<plan>`) and recording what was already done
(skills loaded, tools run + their result, the last conclusion) — folded into the
system prompt. The model keeps continuity (and, crucially, keeps the GOAL anchor
that prevents mid-loop drift) instead of forgetting. Deterministic: no model call.

Pure and backend-agnostic: it takes a `count(text) -> int` function so the real
tokenizer drives the math (HF `tok`, llama.cpp `tokenize`), with a chars/4
fallback when none is available. Nothing here calls a model.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class WindowConfig:
    n_ctx: int                    # hard token ceiling of the model
    reserve_output: int = 192     # tokens kept free for the model's reply
    reserve_thinking: int = 1024  # tokens kept free for in-turn tool calls + results
    safety_margin: int = 64       # slack for chat-template / special tokens
    reserve_digest: int = 256     # max tokens the compacted-history note may use


@dataclass(frozen=True)
class WindowStats:
    n_ctx: int
    budget: int           # tokens available for system + history + user
    system_tokens: int
    history_tokens: int   # tokens of the history actually kept
    user_tokens: int
    used_tokens: int      # system + kept-history + user
    turns_total: int
    turns_kept: int
    turns_evicted: int
    overflow: bool        # system + user alone already exceed the budget
    digest: str = ""      # compacted note for the evicted turns ("" when none/disabled)

    def as_payload(self) -> dict:
        return asdict(self)


# Per-message overhead for chat-template role tags / delimiters. Approximate but
# consistent, so the budget stays conservative rather than optimistic.
_MSG_FRAMING = 4

# Harness verbs / panels to mine from the evicted turns when compacting.
_GOAL = re.compile(r"<goal>(.*?)</goal>", re.DOTALL)
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_SKILL = re.compile(r"<skill>\s*([A-Za-z0-9_][A-Za-z0-9_-]*)\s*</skill>")
_TOOLNAME = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")
_ANYTAG = re.compile(r"<[^>]+>")


def estimate_tokens(text: str) -> int:
    """Fallback counter when no tokenizer is available (~4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def compact(evicted: list[dict], count: Callable[[str], int], max_tokens: int) -> str:
    """Distill the EVICTED (oldest) turns into ONE compact memory note so the model
    keeps continuity instead of forgetting mid-session. Pins the objective
    (`<goal>`/`<plan>`) and records what's already done — skills loaded, tools run +
    their result signal, the last conclusion — so the model neither reloads a skill
    nor contradicts an earlier answer. Deterministic (no model call); bounded to
    ~`max_tokens` by adding the highest-value lines first (objective wins)."""
    if not evicted:
        return ""
    goal = plan = last_answer = first_user = None
    skills: list[str] = []
    ran: list[str] = []
    pending: str | None = None          # tool name awaiting its result, for pairing
    for m in evicted:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            if first_user is None:
                first_user = content
        elif role == "assistant":
            if (g := _GOAL.findall(content)):
                goal = g[-1].strip()
            if (p := _PLAN.findall(content)):
                plan = p[-1].strip()
            for s in _SKILL.findall(content):
                if s not in skills:
                    skills.append(s)
            names = [n for n in _TOOLNAME.findall(content) if n != "load_skill"]
            pending = names[0] if names else None
            if not _SKILL.search(content) and not names:   # a plain final reply
                txt = _ANYTAG.sub("", content).strip()
                if txt:
                    last_answer = txt
        elif role == "tool" and pending:
            sig = (content.strip().splitlines() or [""])[0]
            ran.append(f"{pending} -> {_clip(sig, 70)}")
            pending = None

    # Priority order: the objective anchor first, then what's done, then the thread.
    header = "[Earlier context, compacted - treat as established facts]"
    candidates = []
    if goal:
        candidates.append(f"Goal: {_clip(goal, 200)}")
    if plan:
        candidates.append("Plan: " + _clip(plan.replace("\n", " "), 240))
    if skills:
        candidates.append("Skills already loaded (do not reload): " + ", ".join(skills))
    if ran:
        candidates.append("Tools already run: " + "; ".join(ran[-6:]))
    if last_answer:
        candidates.append(f'Last answer given: "{_clip(last_answer, 200)}"')
    elif first_user:
        candidates.append(f'Started with: "{_clip(first_user, 140)}"')
    out = [header]
    for c in candidates:
        if count("\n".join(out + [c])) > max_tokens:
            break
        out.append(c)
    return "\n".join(out) if len(out) > 1 else ""


class ContextWindow:
    def __init__(self, count: Callable[[str], int], config: WindowConfig) -> None:
        self._count = count
        self.config = config

    def budget(self) -> int:
        c = self.config
        return max(0, c.n_ctx - c.reserve_output - c.reserve_thinking - c.safety_margin)

    def _msg_tokens(self, msg: dict) -> int:
        return self._count(msg.get("content", "")) + _MSG_FRAMING

    def _suffix(self, history: list[dict], avail: int) -> tuple[list[dict], int]:
        """Most-recent contiguous suffix of `history` fitting `avail` tokens, trimmed
        to start on a user turn so the kept history never begins mid-pair."""
        kept_rev: list[dict] = []
        hist_t = 0
        for msg in reversed(history):  # newest first
            t = self._msg_tokens(msg)
            if hist_t + t > avail:
                break
            kept_rev.append(msg)
            hist_t += t
        kept = list(reversed(kept_rev))
        if kept and kept[0].get("role") != "user":  # don't start mid-pair
            hist_t -= self._msg_tokens(kept[0])
            kept = kept[1:]
        return kept, hist_t

    def fit(self, system: str, history: list[dict], user: str,
            compress: bool = False) -> tuple[list[dict], WindowStats]:
        """Trim `history` to the most-recent turns that fit the token budget.

        Keeps a contiguous suffix (newest turns), evicting oldest-first; the
        system prompt and current user turn are always retained. If those two
        alone overflow, we still return them and flag `overflow` — a degraded
        answer beats a crash.

        With `compress=True`, the evicted prefix is distilled into `stats.digest`
        (a compact memory note pinning the goal/plan + what's already done), and
        room for it is reserved from the budget. The caller folds `digest` into the
        system prompt so continuity (and the goal anchor) survives compaction.
        Default `compress=False` keeps the pure drop-oldest behavior unchanged.
        """
        budget = self.budget()
        sys_t = self._count(system) + _MSG_FRAMING
        usr_t = self._count(user) + _MSG_FRAMING
        avail = budget - sys_t - usr_t

        kept, hist_t = self._suffix(history, avail)
        digest = ""
        if compress and len(kept) < len(history):
            # Reserve room for the note, refit tighter, then build it from whatever
            # ends up evicted (so newly-evicted turns are captured too).
            evicted = history[: len(history) - len(kept)]
            probe = compact(evicted, self._count, self.config.reserve_digest)
            if probe:
                kept, hist_t = self._suffix(history, avail - (self._count(probe) + _MSG_FRAMING))
                evicted = history[: len(history) - len(kept)]
                digest = compact(evicted, self._count, self.config.reserve_digest)
        dig_t = (self._count(digest) + _MSG_FRAMING) if digest else 0

        stats = WindowStats(
            n_ctx=self.config.n_ctx, budget=budget,
            system_tokens=sys_t + dig_t, history_tokens=hist_t, user_tokens=usr_t,
            used_tokens=sys_t + dig_t + hist_t + usr_t,
            turns_total=len(history), turns_kept=len(kept),
            turns_evicted=len(history) - len(kept),
            overflow=avail < 0, digest=digest,
        )
        return kept, stats
