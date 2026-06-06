"""Phase 3 Task 11: serving builds its system prompt with the SAME build_system()
the trainer uses (train == serve), and via progressive disclosure — the catalog
lists skill NAMES+DESCRIPTIONS, never pre-stuffed bodies. Kills the old
skill_prompt() body-injection path."""
from backend.inference import build_system_prompt
from backend.skills import load_skills

BACKEND_TOOLS = ("load_skill", "board_state", "move", "undo", "legal_moves",
                 "list_pieces", "ask_chessbot", "eval", "best_move",
                 "review_move", "threats")


def test_serving_system_lists_load_skill_and_every_backend_tool():
    s = build_system_prompt()
    for tool in BACKEND_TOOLS:
        assert tool in s, tool


def test_serving_system_lists_skill_catalog_names_not_bodies():
    s = build_system_prompt()
    skills = load_skills()
    assert skills, "expected at least chess-coach SKILL.md on disk"
    for skill in skills:
        assert skill.name in s                       # catalog name present
        assert skill.content.strip() not in s        # full body NOT pre-stuffed


def test_serving_system_applies_overlay():
    s = build_system_prompt(agent_overlay="Always answer in exactly one sentence.")
    assert "CUSTOMIZATION" in s
    assert "Always answer in exactly one sentence." in s


def test_serving_default_has_no_customization_block():
    assert "CUSTOMIZATION" not in build_system_prompt()


def test_agent_overlay_reads_env_default_empty(monkeypatch):
    from backend.server import agent_overlay
    monkeypatch.delenv("CHESS_AGENT_OVERLAY", raising=False)
    assert agent_overlay() == ""
    assert "CUSTOMIZATION" not in build_system_prompt(agent_overlay())
    monkeypatch.setenv("CHESS_AGENT_OVERLAY", "Be encouraging and brief.")
    assert agent_overlay() == "Be encouraging and brief."
    assert "Be encouraging and brief." in build_system_prompt(agent_overlay())
