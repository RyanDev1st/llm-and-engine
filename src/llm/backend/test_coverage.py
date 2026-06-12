"""The coverage layer on the single loop: the model must not finish a turn while a
detected intent is ungathered. It is steered once (s1-style "Wait"), then the tool
is force-routed as a backstop. coverage=False disables it (the ablation)."""
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


def _names(out):
    return [parse_call(c)[0] for c in out["tool_calls"]]


def _loop(steps, game=None):
    return CoachLoop(ScriptedModel(steps), ToolExecutor(game or Game(), None))


def test_wait_steer_then_model_complies():
    # "best move and the evaluation" -> required {best_move, eval}.
    # Model gathers best_move, tries to answer, gets steered to eval, complies.
    out = _loop([
        "<tool>best_move top=3",      # gather best_move
        "Here are the moves.",         # tries to answer (eval still outstanding) -> Wait steer
        "<tool>eval depth=18",        # complies with the steer
        "Final summary.",              # all covered -> final reply
    ]).respond([], "give me the best move and the evaluation")
    assert _names(out) == ["best_move", "eval"]
    assert out["reply"] == "Final summary."


def test_backstop_force_routes_when_model_ignores_steer():
    # required {eval}. Model never calls it, even after the steer -> force-routed.
    out = _loop([
        "Just play e4.",               # tries to answer (eval outstanding) -> Wait steer
        "Still just play e4.",         # ignores the steer -> backstop force-routes eval
        "Okay, evaluation noted.",     # all covered -> final reply
    ]).respond([], "how am I doing?")
    assert "eval" in _names(out)
    assert out["reply"] == "Okay, evaluation noted."


def test_coverage_off_lets_the_model_stop_early():
    out = _loop(["Just play e4."]).respond([], "how am I doing?", coverage=False)
    assert out["tool_calls"] == [] and out["reply"] == "Just play e4."


def test_no_required_intent_returns_first_reply():
    out = _loop(["Hello there!"]).respond([], "hi")
    assert out["tool_calls"] == [] and out["reply"] == "Hello there!"


def test_game_over_skips_coverage():
    g = Game()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        g.move(san)
    out = _loop(["That's checkmate — Black wins."], game=g).respond([], "how am I doing?")
    assert out["tool_calls"] == [] and "checkmate" in out["reply"].lower()


def test_dedup_is_by_full_call_not_name():
    # best_move with different args must BOTH run (name-dedup would have blocked the 2nd).
    out = _loop([
        "<tool>best_move depth=1",
        "<tool>best_move top=3 series=2",
        "Done.",
    ]).respond([], "show me moves")     # no required intent; pure model-driven
    assert _names(out) == ["best_move", "best_move"]
