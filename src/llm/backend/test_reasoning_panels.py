"""Goal/think panel surfacing. The model sometimes omits the closing </goal> and runs straight
into <think>/<skill>, so a strict <goal>...</goal> match missed it: the objective only flashed in
the token stream ("only got air of it") and never pinned. _split_reasoning + the loop's early
surfacing use the tolerant _GOAL_OPEN, so an unclosed <goal> still pins to the panel and is stripped
from the chat bubble. CPU, no model for the split tests; scripted model for the loop surfacing."""
from backend.game import Game
from backend.inference import CoachLoop, _split_reasoning
from backend.tools import ToolExecutor


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps); self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]; self.i += 1
        return out


def test_split_extracts_and_strips_a_closed_goal():
    visible, thinks, goal = _split_reasoning("<goal>win the endgame</goal>You are better; push the pawn.")
    assert goal == "win the endgame"
    assert "<goal>" not in visible and visible == "You are better; push the pawn."


def test_split_handles_an_unclosed_goal_before_a_think():
    # The reported bug: no </goal>, runs into <think>. Goal must still surface and be stripped,
    # and the <think> block must be left intact for its own panel.
    visible, thinks, goal = _split_reasoning(
        "<goal>generate a tactical puzzle<think>load the generator</think>Here it is.")
    assert goal == "generate a tactical puzzle"
    assert thinks == ["load the generator"]
    assert "<goal>" not in visible and visible == "Here it is."


def test_split_handles_an_unclosed_goal_to_end():
    visible, thinks, goal = _split_reasoning("<goal>just analyze the position")
    assert goal == "just analyze the position" and visible == ""


def test_loop_pins_an_unclosed_goal_to_the_panel():
    # End-to-end: the model emits an unclosed <goal> then a skill load. The loop must emit a
    # 'goal' event (panel pin), not let it vanish in the stream.
    events = []
    CoachLoop(ScriptedModel([
        "<goal>generate a tactical puzzle<skill>chess-coach</skill>",   # unclosed goal + action
        "Here is your puzzle.",
    ]), ToolExecutor(Game(), None)).respond([], "give me a puzzle", on_event=events.append)
    goals = [e for e in events if e.get("type") == "goal"]
    assert goals and goals[0]["content"] == "generate a tactical puzzle"
