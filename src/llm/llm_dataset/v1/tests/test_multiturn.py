"""Multi-turn follow-up slice (Stockfish-free): rows are a 2-turn conversation
where turn 1 is masked context and turn 2 is the trained follow-up. Verifies the
rows validate, the ephemeral shape (turn-1 answer marked train:false, no turn-1
tools), grounded AND clarify archetypes both appear, and — the v5 fix — every
follow-up that ASSERTS a position fact RE-GROUNDS it with a tool call (the
followup_grounded gate), never "same read as before" from memory.

v5-native: actions are STRUCTURED tool_calls (read via tags.tool_calls_of); a skill
reload is the native load_skill{name} call."""
import copy

from llm_dataset.v1.renderer.multiturn import render_multiturn_row
from llm_dataset.v1.renderer.multiturn_followups import BACKREF
from llm_dataset.v1.renderer.tags import tool_calls_of
from llm_dataset.v1.sampler import MULTITURN_SLICE, plan_scenarios
from llm_dataset.v1.validate import validate_row

FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"


class _FakePos:
    def __init__(self, fen: str) -> None:
        self.fen = fen
        self.score_cp = 150
        self.score_kind = "cp"
        self.best_san = "Nf3"
        self.best_line_sans = ("Nf3", "Bc4", "Qe2")
        self.depth = 12
        self.threats_san = None
        # top-k candidates for the alternatives archetype (san, white-POV cp)
        self.top_moves = (("Nf3", 150), ("Bc4", 120), ("d4", 100))


class _FakeAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _FakePos:
        return _FakePos(fen)


def _rows(n: int):
    scenarios = plan_scenarios({MULTITURN_SLICE: n}, seed=7)   # deterministic (Random(7))
    return [render_multiturn_row(s, _FakeAnnotator()) for s in scenarios]


def _tool_names(row):
    """Non-skill tool calls across the row, in order."""
    return [tc["name"] for m in row["messages"] if m["role"] == "assistant"
            for tc in tool_calls_of(m) if tc["name"] != "load_skill"]


def _skill_loads(row):
    return [tc["arguments"].get("name") for m in row["messages"] if m["role"] == "assistant"
            for tc in tool_calls_of(m) if tc["name"] == "load_skill"]


def test_every_multiturn_row_validates():
    # validate_row now includes followup_grounded, so this also proves no archetype
    # asserts a fact without re-grounding.
    for row in _rows(60):
        assert validate_row(row) == [], (row["id"], validate_row(row))


def test_turn1_is_masked_context_and_has_no_tools():
    for row in _rows(60):
        msgs = row["messages"]
        assert msgs[0]["role"] == "user"
        # the first assistant turn (turn-1 answer) is context-only
        assert msgs[1]["role"] == "assistant" and msgs[1].get("train") is False
        # turn-1 carries NO tool scratchpad (ephemeral-serving shape)
        assert not msgs[1].get("tool_calls")
        # the final turn (turn-2 answer) IS trained (no train:false) and is plain text
        assert msgs[-1]["role"] == "assistant" and msgs[-1].get("train") is not False
        assert not msgs[-1].get("tool_calls")


def test_grounded_and_clarify_archetypes_both_appear():
    rows = _rows(60)
    has_tools = [bool(_tool_names(r)) for r in rows]
    assert any(has_tools), "expected grounded (tool) follow-ups"
    assert not all(has_tools), "expected at least one tool-free clarify follow-up"


def test_grounded_followup_reloads_skill_and_grounds():
    saw_grounded = saw_clarify = False
    for row in _rows(60):
        names = _tool_names(row)
        if names:  # grounded archetype: reloads skill via load_skill, then grounds on board first
            saw_grounded = True
            assert names[0] == "board_state"
            assert "chess-coach" in _skill_loads(row)
            assert "chess-coach" in row["selected_skills"]
            assert "narration_grounded" in row["acceptance_rules"]
            # the grounded final connects back to the prior turn
            assert any(c in row["messages"][-1]["content"] for c in BACKREF)
        else:       # clarify archetype: no tool, no skill reload, ASKS (ends with '?')
            saw_clarify = True
            assert row["selected_skills"] == []
            assert row["messages"][-1]["content"].rstrip().endswith("?")
    assert saw_grounded and saw_clarify


def test_followup_grounded_gate_catches_memory_answer():
    """The exact v1-v4 confab pattern must FAIL: a follow-up that asserts a standing
    with no tool call in the turn ("same read as before — White is on top")."""
    row = next(r for r in _rows(60) if _tool_names(r))
    bad = copy.deepcopy(row)
    # strip the turn-2 tool scratchpad, keep only [user1, asst1(train:false), user2, final]
    user_turns = [i for i, m in enumerate(bad["messages"]) if m["role"] == "user"]
    bad["messages"] = (bad["messages"][:user_turns[1] + 1]
                       + [{"role": "assistant",
                           "content": "Same read as before — White is still on top, no guess."}])
    bad["selected_skills"] = []
    rules = [v.rule for v in validate_row(bad)]
    assert "followup_grounded" in rules, rules


def test_followup_grounded_allows_toolfree_clarify():
    """A tool-free follow-up that ASKS (asserts no fact) must still pass."""
    row = next((r for r in _rows(60) if not _tool_names(r)), None)
    assert row is not None
    assert validate_row(row) == []


class _MatePos(_FakePos):
    def __init__(self, fen: str) -> None:
        super().__init__(fen)
        self.score_cp = 2          # mate distance, not centipawns
        self.score_kind = "mate"
        self.top_moves = (("Nf3", 100000), ("Bc4", 100000), ("d4", 100000))


class _MateAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _MatePos:
        return _MatePos(fen)


def test_multiturn_mate_final_is_tool_grounded():
    """Grounded archetypes whose final claims 'mate in N' (why/plan/line/alts use
    eval_magnitude) must carry 'mate' in a tool result — the [G] preflight hole."""
    scenarios = plan_scenarios({MULTITURN_SLICE: 60}, seed=7)
    saw = False
    for s in scenarios:
        row = render_multiturn_row(s, _MateAnnotator())
        assert validate_row(row) == [], (row["id"], validate_row(row))
        final = row["messages"][-1]["content"].lower()
        tools = " ".join(m["content"] for m in row["messages"] if m["role"] == "tool").lower()
        if "mate" in final:
            saw = True
            assert "mate" in tools, f"mate claim not grounded in {row['id']}"
    assert saw, "expected at least one mate final across the grounded archetypes"
