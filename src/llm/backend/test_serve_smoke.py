"""Phase 3 Task 14 — end-to-end serve smoke (no GGUF, no Stockfish needed):
a scripted model drives the real CoachLoop + ToolExecutor through the trained
shape (lead-in + ONE <tool> per step), the tools actually execute, and a
dropped-in SKILL.md is discoverable in the catalog and loadable. Start-position
tools avoid the engine, so this runs fast and deterministically."""
import re
import shutil

from backend.game import Game
from backend.inference import CoachLoop, build_system_prompt
from backend.skills import SKILLS_DIR, load_skills
from backend.tools import ToolExecutor


class ScriptedModel:
    """Emits the trained shape: a lead-in sentence then one <tool> (no closing
    tag — the loop stops at </tool> and normalizes), final = plain text."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[self.i]
        self.i += 1
        return out


def test_coach_loop_executes_leadin_then_tool_sequence():
    steps = [
        "Let me load my coaching skill.\n<tool>load_skill name=chess-coach",
        "First, the position.\n<tool>board_state fields=basic",
        "Now the engine read.\n<tool>eval depth=12",
        "You're set up fine at the start. Want the plan, or Black's reply first?",
    ]
    loop = CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None))
    out = loop.respond([], "how's my game?")

    # all three actions ran, in order (lead-in did NOT swallow the call). load_skill
    # is displayed via the trained <skill> verb, fact tools as <tool> — the serve
    # display contract (inference._to_skill_verb).
    def _action_name(c):
        sk = re.search(r"<skill>\s*([A-Za-z0-9_-]+)\s*</skill>", c)
        return "load_skill" if sk else re.search(r"<tool>\s*([a-z_]+)", c).group(1)
    names = [_action_name(c) for c in out["tool_calls"]]
    assert names == ["load_skill", "board_state", "eval"]
    # the stored assistant turn keeps the lead-in narration before the call
    assert out["tool_calls"][0].startswith("Let me load")
    assert "chess-coach" in out["tool_results"][0]          # real skill body
    assert out["tool_results"][1].startswith("board_state:") and "turn=white" in out["tool_results"][1]
    assert out["tool_results"][2].startswith("score:")      # start-pos eval (no engine)
    assert out["reply"].rstrip().endswith("?")
    assert "<tool>" not in out["reply"]
    # context-window stats are reported and the prompt stays within budget
    ctx = out["context"]
    assert ctx["n_ctx"] == 4096 and ctx["budget"] > 0
    assert ctx["used_tokens"] <= ctx["budget"]
    assert ctx["turns_kept"] + ctx["turns_evicted"] == ctx["turns_total"]


def test_game_over_is_detected_and_loop_states_result():
    # Fool's mate: after 1.f3 e5 2.g4 Qh4# White is checkmated.
    game = Game()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        assert game.move(san).startswith("success:"), san
    assert game.over_status() == "checkmate"          # deterministic, board-derived
    # On a finished game the loop must still run; the model states the result
    # (no analysis tool is forced — the routing layer short-circuits to a state hint).
    loop = CoachLoop(ScriptedModel(["That's checkmate — Black wins. Want a new game?"]),
                     ToolExecutor(game, None))
    out = loop.respond([], "how am I doing?")
    assert out["tool_calls"] == []                     # no eval/analysis call on a dead game
    assert "<tool>" not in out["reply"]
    assert out["reply"]


def test_load_uci_moves_is_atomic_on_a_bad_list():
    game = Game()
    assert game.load_uci_moves(["e2e4", "e7e5"]) is True
    assert game.san_stack == ["e4", "e5"]
    good_fen = game.board.fen()
    # a list with an illegal move must NOT half-replay onto the live board
    assert game.load_uci_moves(["e2e4", "e7e5", "zzzz"]) is False
    assert game.board.fen() == good_fen and game.san_stack == ["e4", "e5"]


def test_dropped_in_skill_is_discoverable_and_loadable():
    demo = SKILLS_DIR / "_smoke_demo"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "SKILL.md").write_text(
        "---\nname: _smoke_demo\ndescription: A throwaway smoke skill for the serve test.\n---\n"
        "# _smoke_demo\nWhen to use: never in production.\nSteps:\n1. Prove discovery works.\n",
        encoding="utf-8")
    try:
        assert any(s.name == "_smoke_demo" for s in load_skills())
        assert "_smoke_demo" in build_system_prompt()                 # in the catalog
        body = ToolExecutor(Game(), None).execute("<tool>load_skill name=_smoke_demo</tool>")
        assert "Prove discovery works." in body                       # loadable body
    finally:
        shutil.rmtree(demo)
