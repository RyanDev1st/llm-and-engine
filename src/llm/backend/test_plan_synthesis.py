"""Plan mode follow-through: the model emits a <goal>/<plan>, works the TOOL boxes, but its terminal
box is 'synthesize and reply' (no tool) — and it deflected ("what would you like to look at?") instead
of answering, so the plan never completed (live failure). The loop must FORCE the synthesis from the
gathered results. CPU: scripted model + engine-free tools (board_state, legal_moves)."""
from backend.game import Game
from backend.inference import CoachLoop
from backend.tools import ToolExecutor
from backend.toolfmt import parse_call


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps); self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]; self.i += 1
        return out


_PLAN = ("<goal>understand the position</goal><plan>\n"
         "- [ ] get the board state (board_state)\n"
         "- [ ] list the legal moves (legal_moves)\n"
         "- [ ] synthesize and reply (synthesize)\n</plan>")


def _loop(steps):
    return CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None))


def test_plan_forces_synthesis_instead_of_deflecting():
    out = _loop([
        _PLAN,                          # emit the plan panel (registers the 2 TOOL boxes)
        "<tool>board_state</tool>",     # box 1 (engine-free)
        "<tool>legal_moves</tool>",     # box 2 (engine-free)
        "What aspect interests you?",   # tries to finalize with a bare ASK-BACK -> force synthesis
        "You're fine; develop your knights and castle.",  # forced synthesis -> the real answer
    ]).respond([], "give me a full read of the position", coverage=False, reasoning_mode="plan")
    # The plan completes with a real synthesis, not the ask-back deflection.
    assert "develop your knights" in out["reply"]
    assert "What aspect" not in out["reply"]
    names = [parse_call(c)[0] for c in out["tool_calls"]]       # both tool boxes actually ran
    assert "board_state" in names and "legal_moves" in names


def test_a_grounded_answer_with_a_trailing_offer_is_NOT_forced():
    # A real synthesis that ends with an optional offer must NOT be treated as an ask-back.
    out = _loop([
        _PLAN, "<tool>board_state</tool>", "<tool>legal_moves</tool>",
        "You're slightly better; the centre is yours. Want the attacking plan?",
    ]).respond([], "read the position", coverage=False, reasoning_mode="plan")
    assert out["reply"].startswith("You're slightly better")  # kept as-is, no forced regen
