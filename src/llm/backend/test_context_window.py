"""Unit tests for the token-budgeted context window (session memory manager).

Uses a deterministic char-count tokenizer so budgets are exact and the
invariants are easy to reason about: the kept prompt never exceeds budget, only
the OLDEST turns are evicted, and the system + current user turn always survive.
"""
from backend.context_window import ContextWindow, WindowConfig, estimate_tokens

FRAMING = 4  # must match context_window._MSG_FRAMING


def _win(n_ctx, count=len):
    # reserves zeroed so `budget == n_ctx`, making the arithmetic transparent.
    return ContextWindow(count, WindowConfig(n_ctx=n_ctx, reserve_output=0,
                                             reserve_thinking=0, safety_margin=0))


def _pairs(n):
    """n user/assistant pairs, oldest first, content tagged with its index."""
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


def _convo_tokens(win, system, kept, user):
    total = win._count(system) + FRAMING + win._count(user) + FRAMING
    for m in kept:
        total += win._count(m["content"]) + FRAMING
    return total


def test_estimate_tokens_positive_and_scaling():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_everything_kept_when_under_budget():
    win = _win(10_000)
    history = _pairs(5)
    kept, stats = win.fit("system", history, "hello")
    assert kept == history
    assert stats.turns_evicted == 0
    assert stats.turns_kept == len(history)
    assert stats.used_tokens <= stats.budget


def test_oldest_evicted_and_budget_never_exceeded():
    history = _pairs(20)             # 40 messages
    win = _win(120)                 # tight budget forces eviction
    kept, stats = win.fit("sys", history, "now")
    # kept is a suffix (most-recent) of history
    assert kept == history[len(history) - len(kept):]
    assert stats.turns_evicted > 0
    assert stats.turns_kept + stats.turns_evicted == stats.turns_total == 40
    # the actual prompt token sum stays within budget
    assert _convo_tokens(win, "sys", kept, "now") <= stats.budget
    assert stats.used_tokens == _convo_tokens(win, "sys", kept, "now")


def test_kept_history_starts_on_a_user_turn():
    history = _pairs(20)
    win = _win(140)
    kept, _ = win.fit("sys", history, "q")
    assert kept and kept[0]["role"] == "user"


def test_system_and_user_always_returned_even_on_overflow():
    history = _pairs(3)
    # budget smaller than system+user alone -> overflow, history fully evicted
    win = _win(8)
    kept, stats = win.fit("a_very_long_system_prompt", history, "a_long_user_message")
    assert kept == []
    assert stats.overflow is True
    assert stats.turns_evicted == len(history)


def test_real_tokenizer_swappable_via_count_fn():
    # word-count "tokenizer" — proves the budget math is driven by the injected fn
    win = ContextWindow(lambda t: len(t.split()),
                        WindowConfig(n_ctx=50, reserve_output=0, reserve_thinking=0, safety_margin=0))
    history = [{"role": "user", "content": "one two three four five"},
               {"role": "assistant", "content": "six seven eight nine ten"}]
    kept, stats = win.fit("sys words here", history, "the final question text")
    assert stats.used_tokens <= stats.budget
    assert stats.turns_kept <= len(history)
