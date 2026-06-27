"""V1_U specialist routing: each row reads the intent, picks the RIGHT specialist from the
flat catalog, loads exactly that one, calls its tool, and grounds the answer. Stockfish-free."""
from collections import Counter

from llm_dataset.v1.renderer.specialist_routing import render_specialist_routing_row
from llm_dataset.v1.renderer.tags import tool_calls_of
from llm_dataset.v1.validate import validate_row


def _rows(n):
    return [render_specialist_routing_row(s) for s in range(n)]


def test_rows_validate_clean():
    fails = [(s, validate_row(render_specialist_routing_row(s)))
             for s in range(150) if validate_row(render_specialist_routing_row(s))]
    assert not fails, fails[:3]


def test_routes_to_each_specialist_fairly():
    selected = Counter(r["selected_skills"][0] for r in _rows(120))
    assert {"game-reviewer", "opening-advisor", "tactical-puzzles"} <= set(selected)
    assert all(c > 20 for c in selected.values())          # round-robin by seed, no starvation


def test_loads_exactly_one_skill_the_selected_one():
    for r in _rows(60):
        skill = r["selected_skills"][0]
        loaded = [tc["arguments"].get("name") for m in r["messages"] if m["role"] == "assistant"
                  for tc in tool_calls_of(m) if tc["name"] == "load_skill"]
        assert loaded == [skill]                                  # exactly one, the selected one
        assert skill in {s["name"] for s in r["skills_index"]}   # catalog lists it


def test_history_specialists_read_the_board_first():
    for r in _rows(60):
        if r["selected_skills"][0] in ("game-reviewer", "opening-advisor"):
            tools = [m["content"] for m in r["messages"] if m["role"] == "tool"]
            assert any(t.startswith("board_state:") and "last_move=" in t and "none" not in t
                       for t in tools), r["id"]
