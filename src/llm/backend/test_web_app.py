"""Audit #2 regression: in dual mode the base (untrained) loop must NEVER mutate
the real, displayed board. It runs on a private mirror; a base move/undo/load_fen
stays off APP.game. No model or engine needed — scripted loops + the `move` tool
(which doesn't touch Stockfish)."""
import chess

from backend.inference import CoachLoop
from backend.web_app import App


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def test_base_loop_never_mutates_real_board():
    app = App(adapter=None)
    # SFT just talks; base tries to PLAY a move. If base shared the real board,
    # APP.game would advance — it must not.
    app.loop = CoachLoop(ScriptedModel(["You're set up fine at the start."]), app.executor)
    app.loop_base = CoachLoop(
        ScriptedModel(["I'll grab the centre.\n<tool>move san=e4", "e4 looks good."]),
        app.base_executor)

    out = app.chat("what should I do?", variant="both")

    assert set(out) == {"sft", "base", "state"}
    assert app.game.board.fen() == chess.STARTING_FEN          # real board untouched
    assert app.base_executor.game.san_stack == ["e4"]          # base moved on its OWN board
    assert out["state"]["fen"].split()[0] == chess.STARTING_FEN.split()[0]  # display = real board


def test_coverage_compare_returns_both_and_isolates_board():
    app = App(adapter=None)
    # ON run force-routes the required eval; OFF run just replies without it.
    app.loop = CoachLoop(ScriptedModel(["I'll just say it.", "<tool>eval depth=18", "ON: equal."]), app.executor)
    app.loop_mirror = CoachLoop(ScriptedModel(["OFF: just play e4."]), app.base_executor)
    out = app.chat("how am I doing?", variant="coverage", coverage=True)
    assert set(out) == {"on", "off", "state"}
    assert out["on"]["reply"] == "ON: equal." and out["on"]["tool_calls"]      # coverage ran eval
    assert out["off"]["reply"] == "OFF: just play e4." and out["off"]["tool_calls"] == []  # off: no tool
    assert app.game.board.fen() == chess.STARTING_FEN     # neither run advanced the real board
