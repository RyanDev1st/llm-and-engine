"""Multi-turn follow-up slice (Stockfish-free): rows are a 2-turn conversation
where turn 1 is masked context and turn 2 is the trained follow-up. Verifies the
rows validate, the ephemeral shape (turn-1 answer marked train:false, no turn-1
tools), both archetypes appear, and turn 2 back-references the prior turn.

v5-native: actions are STRUCTURED tool_calls (read via tags.tool_calls_of); a skill
reload is the native load_skill{name} call."""
from llm_dataset.v1.renderer.multiturn import BACKREF_A, BACKREF_B, render_multiturn_row
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
        self.best_line_sans = ["Nf3", "Bc4", "Qe2"]
        self.depth = 12
        self.threats_san = None
        self.mate = None


class _FakeAnnotator:
    def annotate(self, fen: str, depth: int = 12) -> _FakePos:
        return _FakePos(fen)


def _rows(n: int):
    scenarios = plan_scenarios({MULTITURN_SLICE: n}, seed=7)
    return [render_multiturn_row(s, _FakeAnnotator()) for s in scenarios]


def _tool_names(row):
    """Non-skill tool calls across the row, in order."""
    return [tc["name"] for m in row["messages"] if m["role"] == "assistant"
            for tc in tool_calls_of(m) if tc["name"] != "load_skill"]


def _skill_loads(row):
    return [tc["arguments"].get("name") for m in row["messages"] if m["role"] == "assistant"
            for tc in tool_calls_of(m) if tc["name"] == "load_skill"]


def test_every_multiturn_row_validates():
    for row in _rows(24):
        assert validate_row(row) == [], (row["id"], validate_row(row))


def test_turn1_is_masked_context_and_has_no_tools():
    for row in _rows(24):
        msgs = row["messages"]
        assert msgs[0]["role"] == "user"
        # the first assistant turn (turn-1 answer) is context-only
        assert msgs[1]["role"] == "assistant" and msgs[1].get("train") is False
        # turn-1 carries NO tool scratchpad (ephemeral-serving shape)
        assert not msgs[1].get("tool_calls")
        # the final turn (turn-2 answer) IS trained (no train:false) and is plain text
        assert msgs[-1]["role"] == "assistant" and msgs[-1].get("train") is not False
        assert not msgs[-1].get("tool_calls")


def test_final_backreferences_prior_turn():
    connectives = tuple(BACKREF_A) + tuple(BACKREF_B)
    for row in _rows(24):
        final = row["messages"][-1]["content"]
        assert any(c in final for c in connectives), final


def test_both_archetypes_present():
    rows = _rows(24)
    has_tools = [bool(_tool_names(r)) for r in rows]
    assert any(has_tools) and not all(has_tools), "expected both reference-only and tool follow-ups"


def test_tool_followup_reloads_skill_and_grounds():
    for row in _rows(24):
        names = _tool_names(row)
        if names:  # tool archetype: reloads skill via load_skill, then grounds on board first
            assert names[0] == "board_state"
            assert "chess-coach" in _skill_loads(row)
            assert "chess-coach" in row["selected_skills"]
            assert "narration_grounded" in row["acceptance_rules"]
        else:       # reference/clarify archetype: no tool, no skill reload
            assert row["selected_skills"] == []
