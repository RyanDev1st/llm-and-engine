"""v5 NATIVE-FORMAT serve smoke. With CHESS_NATIVE_FORMAT=1 the model emits Gemma's native
wire form (`<|channel>thought…<channel|>` + `<|tool_call>call:NAME{args}<tool_call|>`). The
loop must parse those calls (native_fmt.parse_native_call), execute the real tools, lift the
native thought out of the final reply, and never leak a native marker to the user — all while
reusing the unchanged CoachLoop. The flag is read at import, so this module sets it FIRST and
imports inside the tests (kept isolated from the v4 suite, which imports with the flag off).

Start-position tools avoid Stockfish, so this runs fast and deterministically (no GPU/engine)."""
import importlib
import os
import re

import pytest

Q = '<|"|>'   # the native string-value quote token (id 52)


@pytest.fixture()
def native_loop(monkeypatch):
    """Re-import the backend with CHESS_NATIVE_FORMAT=1 so the module-level _NATIVE_FMT flag is on,
    then restore the default import afterwards (so later tests see the v4 path)."""
    monkeypatch.setenv("CHESS_NATIVE_FORMAT", "1")
    import backend.inference as inf
    importlib.reload(inf)
    from backend.game import Game
    from backend.tools import ToolExecutor
    assert inf._NATIVE_FMT is True
    yield inf, Game, ToolExecutor
    monkeypatch.delenv("CHESS_NATIVE_FORMAT", raising=False)
    importlib.reload(inf)   # back to v4 default for the rest of the session


class NativeModel:
    """Emits scripted NATIVE-format generations (one per loop step)."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0
        self.seen_messages = []

    def generate(self, messages, max_new_tokens, stop):
        self.seen_messages.append([dict(m) for m in messages])
        out = self.steps[self.i]
        self.i += 1
        return out


def _names(inf, calls):
    """Action names from the loop's stored (display) calls — load_skill shows as <skill>."""
    out = []
    for c in calls:
        sk = re.search(r"<skill>\s*([A-Za-z0-9_-]+)\s*</skill>", c)
        out.append("load_skill" if sk else re.search(r"<tool>\s*([a-z_]+)", c).group(1))
    return out


def test_native_think_route_ground_sequence(native_loop):
    # think mode: native thought -> load skill -> read board -> engine -> grounded final.
    inf, Game, ToolExecutor = native_loop
    steps = [
        f"<|channel>thought\nLoad my coaching skill first.\n<channel|>"
        f"<|tool_call>call:load_skill{{name:{Q}chess-coach{Q}}}<tool_call|>",
        f"<|channel>thought\nNow the position.\n<channel|>"
        f"<|tool_call>call:board_state{{fields:{Q}basic{Q}}}<tool_call|>",
        f"<|tool_call>call:eval{{depth:12}}<tool_call|>",
        "You're set up fine at the start. Want the main plan, or Black's likely reply first?",
    ]
    events = []
    # coverage=False isolates the NATIVE-FORMAT plumbing from the v4 coverage layer (which is
    # already tested elsewhere and would force-route best_move — an engine-only tool — here).
    out = inf.CoachLoop(NativeModel(steps), ToolExecutor(Game(), None)).respond(
        [], "what should I play?", reasoning_mode="think", coverage=False, on_event=events.append)

    assert _names(inf, out["tool_calls"]) == ["load_skill", "board_state", "eval"]
    assert "best_move" in out["tool_results"][0]                       # real condensed skill body
    assert out["tool_results"][1].startswith("board_state:") and "turn=white" in out["tool_results"][1]
    assert out["tool_results"][2].startswith("score:")                # start-pos eval (engine-free)
    # the final reply is clean prose — NO native markers leaked to the user
    for marker in ("<|tool_call>", "<tool_call|>", "<|channel>", "call:", Q):
        assert marker not in out["reply"], marker
    assert out["reply"].rstrip().endswith("?")


def test_native_fast_single_tool_then_answer(native_loop):
    # fast mode: a direct native tool call (no thought channel) then a grounded one-liner.
    # eval is engine-free at the start position (static score), so this needs no Stockfish.
    inf, Game, ToolExecutor = native_loop
    steps = [f"<|tool_call>call:eval{{depth:12}}<tool_call|>",
             "It's level at the start — develop a center pawn like e4 or d4."]
    out = inf.CoachLoop(NativeModel(steps), ToolExecutor(Game(), None)).respond(
        [], "how do I stand?", reasoning_mode="fast", coverage=False)
    assert _names(inf, out["tool_calls"]) == ["eval"]
    assert out["tool_results"][0].startswith("score:")
    assert "<|tool_call>" not in out["reply"] and out["reply"]


def test_native_leaked_call_in_final_is_not_shown(native_loop):
    # If a final generation leaks a native call, contains_tool_call must catch it so the loop
    # narrates the real fact instead of surfacing raw markers (the budget-forced fallback path).
    inf, Game, ToolExecutor = native_loop
    assert inf.contains_tool_call(f"<|tool_call>call:eval{{}}<tool_call|>") is True


def test_native_history_rerenders_as_structured_tool_calls(native_loop, monkeypatch):
    # The model must SEE its prior tool steps as native structured tool_calls on re-render (so the
    # 2nd+ turn matches training). to_native_messages is unit-tested directly here against the loop's
    # own stored history shape (<skill>/<tool> text -> tool_calls), the contract model_hf relies on.
    inf, Game, ToolExecutor = native_loop
    from backend.native_fmt import to_native_messages
    hist = [{"role": "system", "content": "C"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "<skill>chess-coach</skill>"},
            {"role": "tool", "content": "skill body"},
            {"role": "assistant", "content": "Let me look.\n<tool>eval depth=12</tool>"},
            {"role": "tool", "content": "score: +0.20"}]
    nm = to_native_messages(hist)
    assert nm[2]["content"] == "" and nm[2]["tool_calls"][0]["function"]["name"] == "load_skill"
    assert nm[3] == {"role": "tool", "name": "load_skill", "content": "skill body"}
    assert nm[4]["tool_calls"][0]["function"] == {"name": "eval", "arguments": {"depth": 12}}
    assert nm[5]["name"] == "eval"
