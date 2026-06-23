"""The "load the skill twice then deflect" failure (observed live on 'gen me a puzzle'): the model
loads a skill, then re-emits the SAME <skill> load instead of USING it. The old loop recorded the
duplicate, displayed "Loaded X" a second time, and BROKE to a generic budget-forced reply — so the
task (the puzzle) was never done. The loop must instead STEER the model forward ("you already loaded
it — act now") and continue, bounded. CPU, scripted model + real load_skill executor (no engine)."""
from backend.game import Game
from backend.inference import CoachLoop
from backend.tools import ToolExecutor
from backend.toolfmt import parse_call


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def _loop(steps):
    return CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None))


def _load_count(out):
    return sum(1 for c in out["tool_calls"] if (parse_call(c)[0] == "load_skill") or ("<skill>" in c))


def test_reloading_a_loaded_skill_steers_forward_not_deflect():
    events = []
    out = _loop([
        "<skill>chess-coach</skill>",                    # load (real)
        "<skill>chess-coach</skill>",                    # re-load the SAME skill -> nudge + continue
        "Here is your answer using the loaded skill.",   # post-nudge: the real answer
    ]).respond([], "coach me", on_event=events.append)
    assert _load_count(out) == 1                          # loaded ONCE — the duplicate is not recorded
    assert out["reply"].startswith("Here is your answer")
    skill_events = [e for e in events if e.get("type") == "tool" and e.get("name") == "skill"]
    assert len(skill_events) == 1                         # "Loaded chess-coach" shown once, not twice


def test_persistent_reload_is_bounded_not_infinite():
    # If the model NEVER stops re-loading, the loop must still terminate (nudge cap + step cap),
    # not spin forever. The skill is still loaded exactly once.
    out = _loop(["<skill>chess-coach</skill>"]).respond([], "coach me")
    assert _load_count(out) == 1
    assert isinstance(out["reply"], str) and out["reply"]


def test_duplicate_fact_tool_still_stops():
    # A repeated FACT tool (not a skill) yields the same result, so the loop should still stop and
    # answer — the steer-forward path is for skill re-loads only.
    g = Game()
    out = CoachLoop(ScriptedModel([
        "<tool>legal_moves</tool>",     # a fact tool (no engine needed)
        "<tool>legal_moves</tool>",     # duplicate -> stop, answer
        "done",
    ]), ToolExecutor(g, None)).respond([], "what are my legal moves?")
    assert sum(1 for c in out["tool_calls"] if parse_call(c)[0] == "legal_moves") == 1
