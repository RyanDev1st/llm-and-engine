from llm_dataset.v1.catalog import (
    OFFICIAL_TOOLS, OFFICIAL_SKILL, alt_skills, alt_tools, synthetic_skill_name, synthetic_tool_name,
)


def test_official_catalog_has_plugin_provenance():
    assert OFFICIAL_SKILL["plugin"] == "chess-official"
    assert OFFICIAL_SKILL["source"] == "official_plugin"
    assert OFFICIAL_SKILL["enabled"] is True


def test_official_chess_tools_have_applies_when():
    names = {tool["name"] for tool in OFFICIAL_TOOLS}
    assert {"move", "eval", "best_move", "review_move", "threats",
            "legal_moves", "undo", "list_pieces", "ask_chessbot",
            "load_skill", "board_state"} <= names
    for tool in OFFICIAL_TOOLS:
        assert "applies_when" in tool


def test_alt_pools_have_at_least_eight_entries():
    assert len(alt_skills()) >= 8
    assert len(alt_tools()) >= 8


def test_synthetic_names_are_unfamiliar():
    name = synthetic_skill_name(seed=7)
    assert "-" in name or "_" in name
    assert any(ch.isdigit() for ch in name)
    tool = synthetic_tool_name(seed=7)
    assert "_" in tool
