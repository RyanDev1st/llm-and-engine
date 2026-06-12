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


def test_model_proactive_multi_tool_then_reply():
    # "best move and the evaluation" -> required {best_move, eval}.
    # Model proactively gathers BOTH itself, then replies — no forcing needed.
    out = _loop([
        "<tool>best_move top=3",      # gather best_move
        "<tool>eval depth=18",        # gather eval (model-driven)
        "Final summary.",              # all covered -> final reply (mentions no fact)
    ]).respond([], "give me the best move and the evaluation")
    assert _names(out) == ["best_move", "eval"]
    # the vague reply mentions neither fact -> answer-coverage appends both, grounded
    assert out["reply"].startswith("Final summary.")


def test_force_routes_outstanding_when_model_stops_early():
    # required {eval}. Model stops without it -> force-routed directly (no nudge round-trip).
    out = _loop([
        "Just play e4.",               # stops; eval outstanding -> force-route eval
        "Okay, evaluation noted.",     # all covered -> final reply (states no number)
    ]).respond([], "how am I doing?")
    assert "eval" in _names(out)
    assert out["reply"].startswith("Okay, evaluation noted.")  # eval fact appended after


def test_on_event_fires_per_tool_for_streaming():
    # Streaming progress: on_event must fire once per executed tool, in order, each
    # carrying the tool name + result — so the UI can show steps live.
    events = []
    out = _loop([
        "<tool>eval depth=18",
        "<tool>best_move top=3",
        "Done.",
    ]).respond([], "give me the best move and the evaluation", on_event=events.append)
    assert [e["type"] for e in events] == ["tool", "tool"]
    assert [e["name"] for e in events] == ["eval", "best_move"]
    assert all(e["result"] for e in events)        # each event carries the tool result
    assert _names(out) == ["eval", "best_move"]     # non-streaming return shape unchanged


def test_plural_best_moves_force_routed_after_eval():
    # The screenshot: "eval and the 5 next best moves" — model evals then stops to ask.
    # Coverage must force-route best_move (top=5) so both are gathered.
    out = _loop([
        "<tool>eval depth=18",                       # eval gathered
        "Want me to suggest the top 5 moves?",        # tries to answer; best_move outstanding -> Wait
        "Sure, here they are.",                        # still no best_move -> backstop force-routes it
    ]).respond([], "can you eval and give me the 5 next best moves?")
    assert set(_names(out)) == {"eval", "best_move"}
    assert any("top=5" in c for c in out["tool_calls"])   # honored the requested count


def test_coverage_off_lets_the_model_stop_early():
    out = _loop(["Just play e4."]).respond([], "how am I doing?", coverage=False)
    assert out["tool_calls"] == [] and out["reply"] == "Just play e4."


def test_no_required_intent_returns_first_reply():
    out = _loop(["Hello there!"]).respond([], "hi")
    assert out["tool_calls"] == [] and out["reply"] == "Hello there!"


def test_leadin_only_terminal_reply_narrates_tool_result():
    # the live "it didn't finish" bug: model ran ask_chessbot, then its final reply was
    # just "Loading the chess-coach skill." (a dangling lead-in) -> narrate the real result.
    out = _loop([
        "<tool>ask_chessbot query=explain chess",          # runs ask_chessbot
        "Loading the chess-coach skill.",                   # terminal lead-in, no tool -> non-answer
    ]).respond([], "explain chess in 8 sentences")
    assert "ask_chessbot" in _names(out)
    assert "Loading the chess-coach skill" not in out["reply"]      # the dangling lead-in is replaced
    assert out["reply"]                                             # with a real narrated answer


def test_leadin_only_kept_when_no_tools_ran():
    # without tools, a lead-in is the model's actual (if weak) reply — don't fabricate.
    out = _loop(["Let me check that for you."]).respond([], "hi")
    assert out["reply"] == "Let me check that for you."


def test_game_over_skips_coverage():
    g = Game()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        g.move(san)
    out = _loop(["That's checkmate — Black wins."], game=g).respond([], "how am I doing?")
    assert out["tool_calls"] == [] and "checkmate" in out["reply"].lower()


def test_answer_coverage_appends_dropped_eval():
    # The screenshot bug: model gathers eval AND best_move, but the final reply only
    # narrates the moves. Answer-coverage must append the eval fact (grounded).
    out = _loop([
        "<tool>eval depth=18",          # required eval gathered
        "<tool>best_move top=3",        # required best_move gathered
        "The engine suggests e4, then d4 and c4. Want me to play e4?",  # drops the eval
    ]).respond([], "suggest 3 next best moves and the eval")
    assert set(_names(out)) == {"eval", "best_move"}
    assert "e4" in out["reply"]                      # moves still there
    # eval result at the start position is "score: 0.00 ... equal" -> appended fact
    assert "0.00" in out["reply"] or "equal" in out["reply"].lower()


def test_answer_coverage_no_double_when_already_mentioned():
    # If the reply already states the eval number, don't append it again.
    out = _loop([
        "<tool>eval depth=18",
        "<tool>best_move top=3",
        "Position is 0.00 (equal); best is e4, then d4, c4.",
    ]).respond([], "best moves and the eval")
    assert out["reply"].count("0.00") == 1           # not duplicated


def test_dedup_is_by_full_call_not_name():
    # best_move with different args must BOTH run (name-dedup would have blocked the 2nd).
    out = _loop([
        "<tool>best_move depth=1",
        "<tool>best_move top=3 series=2",
        "Done.",
    ]).respond([], "show me moves")     # no required intent; pure model-driven
    assert _names(out) == ["best_move", "best_move"]
