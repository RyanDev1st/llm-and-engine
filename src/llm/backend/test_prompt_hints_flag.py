"""Off-distribution prompt nudges are OFF by default. The model trained on the BARE contract and
benchmarks at ~96% routing; the live serve used to append 'SKILL HINT' / 'ROUTING HINT' blocks the
model never saw, and routed worse (loaded the wrong skill despite the hint). Default: the served
system prompt is the bare contract (no hint blocks); CHESS_PROMPT_HINTS=1 restores them. CPU, scripted
model that records the system prompt it received."""
from backend import inference
from backend.game import Game
from backend.inference import CoachLoop, PLUGIN_CONTEXT
from backend.tools import ToolExecutor


class CapturingModel:
    """Records the system prompt of each generate() call; answers plainly (no tools)."""
    def __init__(self):
        self.systems = []

    def generate(self, messages, max_new_tokens, stop):
        self.systems.append(messages[0]["content"] if messages and messages[0]["role"] == "system" else "")
        return "Sure — here is a tactical puzzle for you to solve."


def _run(monkeypatch, on):
    monkeypatch.setattr(inference, "_PROMPT_HINTS", on)
    m = CapturingModel()
    pc = {k: list(v) for k, v in PLUGIN_CONTEXT.items()}
    CoachLoop(m, ToolExecutor(Game(), None), plugin_context=pc).respond(
        [], "load up the puzzle and guide me to solve it", reasoning_mode="auto")
    return m.systems[0]


def test_hints_absent_by_default(monkeypatch):
    sysp = _run(monkeypatch, False)
    assert "SKILL HINT" not in sysp and "ROUTING HINT" not in sysp     # bare contract == benchmark
    assert "tactical-puzzles" in sysp                                  # the skill is still in the catalog


def test_hints_present_when_flag_on(monkeypatch):
    sysp = _run(monkeypatch, True)
    assert "SKILL HINT" in sysp                                        # opt-in restores the nudge
    assert "tactical-puzzles" in sysp
