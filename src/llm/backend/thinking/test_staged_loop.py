import chess

from backend.game import Game
from backend.thinking.prompts import board_facts, facts_summary, build_controller_system, build_narrator_system


def test_board_facts_reads_live_board():
    g = Game()
    bf = board_facts(g)
    assert "turn=white" in bf and "legal_moves=20" in bf and "last_move=none" in bf


def test_facts_summary_compacts_results():
    assert facts_summary([]) == "(none yet)"
    assert facts_summary([("eval", "score: +0.30")]) == "eval→score: +0.30"


def test_controller_system_has_manifest_and_outstanding():
    s = build_controller_system("", None, "best move and eval", "", ["eval"])
    assert "AVAILABLE TOOLS" in s          # full manifest present (it can route)
    assert "DONE" in s and "OUTSTANDING" in s and "eval" in s


def test_narrator_system_has_no_tool_manifest():
    s = build_narrator_system("")
    assert "AVAILABLE TOOLS" not in s      # narrator cannot route
    assert "grounded" in s.lower()


from backend.thinking.parse import parse_controller


def test_parse_controller_tool_done_and_recovery():
    kind, call = parse_controller("<tool>eval depth=18</tool>")
    assert kind == "tool" and "eval" in call
    assert parse_controller("DONE") == ("done", None)
    assert parse_controller("done.") == ("done", None)
    # Gemma's native wrapper is recovered into a tool action
    k, c = parse_controller("Let me check. <tool_code>eval depth=18</tool_code>")
    assert k == "tool" and "<tool>eval" in c
    # prose that is neither a call nor DONE -> fail toward narrating (done)
    assert parse_controller("I think we are good here") == ("done", None)
    assert parse_controller("") == ("done", None)


from backend.tools import ToolExecutor
from backend.thinking.loop import StagedLoop, MAX_STEPS


class ScriptedModel:
    """Returns scripted stage outputs in order; final extra output is the Narrator."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def _loop(steps, game=None):
    return StagedLoop(ScriptedModel(steps), ToolExecutor(game or Game(), None))


def _names(out):
    from backend.toolfmt import parse_call
    return [parse_call(c)[0] for c in out["tool_calls"]]


def test_one_tool_then_done():
    out = _loop(["<tool>eval depth=18", "DONE", "Equal here. Want the plan?"]).run([], "how am I doing?")
    assert _names(out) == ["eval"]
    assert out["reply"].endswith("?") and "<tool>" not in out["reply"]
    assert out["trace"] and out["trace"][-1]["stage"] == "narrator"


def test_multi_tool_model_driven():
    # empty required (no recognized intent) — the model chains tools itself
    out = _loop(["<tool>eval depth=18", "<tool>best_move depth=18", "DONE", "Here you go."]).run([], "tell me everything")
    assert _names(out) == ["eval", "best_move"]


def test_guaranteed_coverage_forces_missing_tool():
    # "best move and the evaluation" -> required {best_move, eval}; model DONEs early
    out = _loop(["<tool>best_move depth=18", "DONE", "DONE", "Summary."]).run([], "best move and the evaluation")
    assert set(_names(out)) == {"best_move", "eval"}   # eval force-routed despite premature DONE


def test_immediate_done_no_tools():
    out = _loop(["DONE", "Hello! Ask me about the position."]).run([], "hi there")
    assert out["tool_calls"] == [] and out["reply"]


def test_malformed_controller_recovers_then_done():
    out = _loop(["<tool_code>eval depth=18</tool_code>", "DONE", "Equal."]).run([], "evaluate it")
    assert _names(out) == ["eval"]


def test_dedup_with_nothing_outstanding_stops():
    out = _loop(["<tool>eval depth=18", "<tool>eval depth=18", "Done analysing."]).run([], "tell me everything")
    assert _names(out) == ["eval"]                     # repeat broke the loop


def test_cap_stops_at_max_steps():
    distinct = ["<tool>eval depth=18", "<tool>best_move depth=18", "<tool>threats depth=12",
                "<tool>review_move depth=15", "<tool>legal_moves", "<tool>list_pieces color=white",
                "<tool>board_state fields=all", "<tool>ask_chessbot query=hi",
                "<tool>load_fen fen=8/8/8/8/8/8/8/8 w - - 0 1", "<tool>undo"]
    out = _loop(distinct + ["reply"]).run([], "tell me everything")
    assert len(out["tool_calls"]) == MAX_STEPS


def test_game_over_no_analysis():
    g = Game()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        g.move(san)
    out = _loop(["DONE", "That's checkmate — Black wins. New game?"], game=g).run([], "how am I doing?")
    assert out["tool_calls"] == [] and "checkmate" in out["reply"].lower()
