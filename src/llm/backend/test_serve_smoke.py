"""Phase 3 Task 14 — end-to-end serve smoke (no GGUF, no Stockfish needed):
a scripted model drives the real CoachLoop + ToolExecutor through the trained
shape (lead-in + ONE <tool> per step), the tools actually execute, and a
dropped-in SKILL.md is discoverable in the catalog and loadable. Start-position
tools avoid the engine, so this runs fast and deterministically."""
import re
import shutil

from backend.game import Game
from backend.inference import CoachLoop, build_system_prompt, is_plan_panel
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
    assert "best_move" in out["tool_results"][0]            # real (condensed) skill body
    assert out["tool_results"][1].startswith("board_state:") and "turn=white" in out["tool_results"][1]
    assert out["tool_results"][2].startswith("score:")      # start-pos eval (no engine)
    assert out["reply"].rstrip().endswith("?")
    assert "<tool>" not in out["reply"]
    # context-window stats are reported and the prompt stays within budget
    ctx = out["context"]
    assert ctx["n_ctx"] == 4096 and ctx["budget"] > 0
    assert ctx["used_tokens"] <= ctx["budget"]
    assert ctx["turns_kept"] + ctx["turns_evicted"] == ctx["turns_total"]


class StopHonoringModel:
    """Like a real backend: returns scripted text but TRUNCATES at the first stop
    substring (inclusive of the stop), and records the stop lists it was handed.
    Lets us prove the loop's action stop tokens actually bound the generation."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0
        self.stops_seen = []

    def generate(self, messages, max_new_tokens, stop):
        self.stops_seen.append(list(stop))
        out = self.steps[self.i]
        self.i += 1
        cut = len(out)
        for s in stop:
            j = out.find(s)
            if j != -1:
                cut = min(cut, j + len(s))   # keep the stop token, drop the tail
        return out[:cut]


def test_skill_load_generation_stops_at_one_action():
    # Regression: </skill> must be in the action stop list. The model over-generates a
    # skill load followed by a SECOND <skill> tag and a partial tail; the loop must stop
    # the generation at the first </skill> so exactly ONE action is recorded and displayed
    # ONCE — no auto double-emit, no partial <skill>, no missing close tag.
    over = ("<skill>chess-coach</skill> <skill>chess-coach</skill> then I'll also load analysi")
    model = StopHonoringModel([over,
                               "You're set up fine at the start. Want the plan?",
                               "DONE"])  # self-verify verdict after a context-only turn
    out = CoachLoop(model, ToolExecutor(Game(), None)).respond([], "coach me")
    assert "</skill>" in model.stops_seen[0]                 # action gen was given </skill> to stop on
    assert [_act_name(c) for c in out["tool_calls"]] == ["load_skill"]
    assert out["tool_calls"][0].count("<skill>") == 1        # displayed once, not doubled
    assert "best_move" in out["tool_results"][0]             # real (condensed) skill body loaded
    assert "<skill>" not in out["reply"] and out["reply"].rstrip().endswith("?")


def _act_name(c):
    sk = re.search(r"<skill>\s*([A-Za-z0-9_-]+)\s*</skill>", c)
    return "load_skill" if sk else re.search(r"<tool>\s*([a-z_]+)", c).group(1)


def test_plan_panel_is_not_final_and_loop_works_the_boxes():
    # A plan-mode model emits the <goal>/<plan> panel first. The loop must surface it
    # and KEEP GOING (work both boxes), not show the raw panel as the answer.
    panel = ("<goal>two things</goal>\n<plan>\n- [ ] load the coach (chess-coach)\n"
             "- [ ] read the board (board_state)\n- [ ] synthesize (none)\n</plan>")
    steps = [panel,
             "Loading the coach.\n<tool>load_skill name=chess-coach",
             "Now the board.\n<tool>board_state fields=basic",
             "You're set up fine at the start. Want the plan, or Black's reply first?"]
    out = CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None)).respond([], "help with two things")
    assert "<plan>" not in out["reply"] and "<goal>" not in out["reply"]   # panel not leaked as reply
    assert out["reply"].rstrip().endswith("?")
    fired = {_act_name(c) for c in out["tool_calls"]}
    assert "load_skill" in fired and "board_state" in fired                 # both boxes ran
    assert any("<plan>" in t.get("content", "") for t in out["turns"])      # panel kept in history


def test_plan_box_backstop_fills_an_unfilled_box_before_final():
    # The model tries to finalize after only one box. The plan-box backstop must steer it
    # to fill the remaining box before the answer is accepted.
    panel = ("<goal>g</goal>\n<plan>\n- [ ] load the coach (chess-coach)\n"
             "- [ ] read the board (board_state)\n- [ ] synthesize (none)\n</plan>")
    steps = [panel,
             "Loading.\n<tool>load_skill name=chess-coach",
             "I'll just answer now.",                              # premature: board box unfilled
             "Now the board.\n<tool>board_state fields=basic",     # produced by the backstop nudge
             "All set at the start. Anything else to check?"]
    out = CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None)).respond([], "do two things")
    fired = {_act_name(c) for c in out["tool_calls"]}
    assert "board_state" in fired                                  # backstop forced the unfilled box
    assert "<plan>" not in out["reply"]


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


def test_reasoning_is_stripped_from_reply_and_sent_to_panels():
    # G1: <think>/<goal> must NOT appear in the visible reply; they go to the panels as events.
    events = []
    steps = ['<goal>tell them where the game stands</goal>\n'
             '<think>no skill needed - answer plainly</think> You are fine; it is your move.']
    out = CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None)).respond(
        [], 'give me general advice', coverage=False, on_event=events.append)
    assert '<think>' not in out['reply'] and '<goal>' not in out['reply']
    assert out['reply'].startswith('You are fine')
    goals = [e for e in events if e['type'] == 'goal']
    thinks = [e for e in events if e['type'] == 'think']
    assert goals and goals[0]['content'] == 'tell them where the game stands'
    assert thinks and 'no skill needed' in thinks[0]['content']


def test_is_plan_panel_requires_a_plan_checklist():
    # G2: a bare <goal> + prose answer is NOT a plan panel (else the answer is discarded).
    assert is_plan_panel('<goal>g</goal>\n<think>t</think> You are slightly better.') is False
    # a real plan-mode panel carries a <plan> checklist.
    assert is_plan_panel('<goal>g</goal>\n<plan>\n- [ ] read the board (board_state)\n</plan>') is True


def test_hf_truncate_cuts_at_first_action_close_inclusive():
    # The HF backend post-truncates a full generation. It must cut at the first action
    # close of ANY kind, INCLUSIVE — </skill> exactly like </tool>. Omitting </skill> dropped
    # the close tag (the "missing </skill>") and let skill gens run on (the row-4 over-gen).
    import pytest
    pytest.importorskip("torch")          # model_hf imports torch at module load
    from backend.model_hf import _truncate
    A = ["</tool>", "</tool_code>", "</skill>"]
    assert _truncate('<think>t</think> <skill>chess-coach</skill> <think>more</think>', A) \
        == '<think>t</think> <skill>chess-coach</skill>'                  # keep </skill>, drop tail
    assert _truncate('Let me check.\n<tool>eval depth=15</tool> junk', A) \
        == 'Let me check.\n<tool>eval depth=15</tool>'                    # keep </tool>, drop junk
    assert _truncate('<think>done</think> You are slightly better.', A) \
        == '<think>done</think> You are slightly better.'                 # no action close -> intact
    assert _truncate('<skill>chess-coach</skill> then <tool>eval</tool>', A) \
        == '<skill>chess-coach</skill>'                                   # earliest close wins


def test_reasoning_mode_threads_into_the_system_prompt():
    # G3: the reasoning mode reaches build_system, so fast and think render different prompts.
    class Rec:
        def __init__(self): self.sys = None
        def generate(self, messages, mx, stop):
            self.sys = messages[0]['content']; return 'All set. Anything else?'
    rf, rt = Rec(), Rec()
    CoachLoop(rf, ToolExecutor(Game(), None)).respond([], 'hello there', coverage=False, reasoning_mode='fast')
    CoachLoop(rt, ToolExecutor(Game(), None)).respond([], 'hello there', coverage=False, reasoning_mode='think')
    assert rf.sys and rt.sys and rf.sys != rt.sys


def test_skill_load_deflection_is_forced_to_a_real_answer():
    # Regression: model loads a skill then DEFLECTS with a capability blurb ("I'm here to
    # help... what's on your mind?") instead of answering. The whiff guard misses it (fluent)
    # and self-verify accepts the model's own "DONE". The deterministic deflection guard must
    # catch it and force a direct answer from the loaded skill — round-trip-neutral (the force
    # gen replaces the verify gen). This is the e2b->e4b regression the user flagged.
    DEFLECT = "I'm here to help with your game, whether tactics, positions, or planning. What's on your mind?"
    REAL = "You are playing White. Develop your knights and a central pawn to start."

    class Defl:
        def __init__(self): self.i = 0
        def generate(self, messages, mx, stop):
            last = messages[-1]["content"] if messages else ""
            if "answer my question directly" in last.lower():
                return REAL                                  # the force nudge -> real answer
            if "Self-check" in last:
                return "DONE"                                # would wrongly accept the blurb
            self.i += 1
            return "<skill>chess-coach</skill>" if self.i == 1 else DEFLECT

    out = CoachLoop(Defl(), ToolExecutor(Game(), None)).respond([], "what am I playing as? how do I play chess?")
    assert out["reply"] == REAL                              # blurb replaced, not accepted
    assert "what's on your mind" not in out["reply"].lower()


def test_good_skill_load_answer_is_not_force_rerolled():
    # The deflection guard must NOT fire on a real, position-specific answer that ends with a
    # guiding question (a trained-good final). force_answer would never be called here.
    GOOD = "You hold a slight edge — the bishop pair helps. Want the attacking plan, or to shore up the defense first?"

    class Good:
        def __init__(self): self.i = 0
        def generate(self, messages, mx, stop):
            last = messages[-1]["content"] if messages else ""
            assert "answer my question directly" not in last.lower(), "force_answer wrongly fired on a good reply"
            if "Self-check" in last:
                return "DONE"
            self.i += 1
            return "<skill>chess-coach</skill>" if self.i == 1 else GOOD

    out = CoachLoop(Good(), ToolExecutor(Game(), None)).respond([], "how am I doing strategically?", coverage=False)
    assert out["reply"] == GOOD


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
