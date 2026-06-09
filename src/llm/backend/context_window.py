"""Token-budgeted context window — the model's session memory manager.

The serving prompt is `system + conversation history + the new user turn`, and a
transformer can only attend to a fixed number of tokens (`n_ctx`). Left
unbounded, a long chat overflows the window: the backend silently drops the
oldest tokens (the model "forgets" mid-session and contradicts itself) or errors
out. This module keeps the prompt inside a token budget by evicting the OLDEST
turns first, while always retaining the system prompt and the current user turn —
so recent context survives and the window never overflows.

Pure and backend-agnostic: it takes a `count(text) -> int` function so the real
tokenizer drives the math (HF `tok`, llama.cpp `tokenize`), with a chars/4
fallback when none is available. Nothing here calls a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class WindowConfig:
    n_ctx: int                    # hard token ceiling of the model
    reserve_output: int = 192     # tokens kept free for the model's reply
    reserve_thinking: int = 1024  # tokens kept free for in-turn tool calls + results
    safety_margin: int = 64       # slack for chat-template / special tokens


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

    def as_payload(self) -> dict:
        return asdict(self)


# Per-message overhead for chat-template role tags / delimiters. Approximate but
# consistent, so the budget stays conservative rather than optimistic.
_MSG_FRAMING = 4


def estimate_tokens(text: str) -> int:
    """Fallback counter when no tokenizer is available (~4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


class ContextWindow:
    def __init__(self, count: Callable[[str], int], config: WindowConfig) -> None:
        self._count = count
        self.config = config

    def budget(self) -> int:
        c = self.config
        return max(0, c.n_ctx - c.reserve_output - c.reserve_thinking - c.safety_margin)

    def _msg_tokens(self, msg: dict) -> int:
        return self._count(msg.get("content", "")) + _MSG_FRAMING

    def fit(self, system: str, history: list[dict], user: str) -> tuple[list[dict], WindowStats]:
        """Trim `history` to the most-recent turns that fit the token budget.

        Keeps a contiguous suffix (newest turns), evicting oldest-first; the
        system prompt and current user turn are always retained. If those two
        alone overflow, we still return them and flag `overflow` — a degraded
        answer beats a crash. A kept suffix that would start on an assistant
        turn is trimmed to start on a user turn, so the history stays coherent.
        """
        budget = self.budget()
        sys_t = self._count(system) + _MSG_FRAMING
        usr_t = self._count(user) + _MSG_FRAMING
        avail = budget - sys_t - usr_t

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

        stats = WindowStats(
            n_ctx=self.config.n_ctx, budget=budget,
            system_tokens=sys_t, history_tokens=hist_t, user_tokens=usr_t,
            used_tokens=sys_t + hist_t + usr_t,
            turns_total=len(history), turns_kept=len(kept),
            turns_evicted=len(history) - len(kept),
            overflow=avail < 0,
        )
        return kept, stats
