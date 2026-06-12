"""Plugin bundles: each contributes tools + skills that grow the served surface, gated
by plugin_context["enabled"]. Tests the contract + cross-bundle routing surface (the
deterministic parts; the MODEL's routing across them is what the bundles exist to test)."""
import chess

from backend import plugins
from backend.game import Game
from backend.tools import ToolExecutor
from backend.inference import serving_tool_manifest, serving_skills_index, PLUGIN_CONTEXT

PC = PLUGIN_CONTEXT
OFF = {"installed": ["chess-official"], "enabled": ["chess-official"], "marketplace": []}


def _ruy() -> Game:
    g = Game()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        g.move(san)
    return g


def test_enabled_plugins_grow_the_manifest_and_catalog():
    tools = {t["name"] for t in serving_tool_manifest(PC)}
    assert {"name_opening", "opening_ideas", "accuracy_report", "find_blunders"} <= tools
    skills = {s["name"] for s in serving_skills_index(PC)}
    assert {"opening-advisor", "game-reviewer"} <= skills


def test_disabled_plugin_contributes_nothing():
    tools = {t["name"] for t in serving_tool_manifest(OFF)}
    assert "name_opening" not in tools and "accuracy_report" not in tools
    assert ToolExecutor(_ruy(), None, plugin_context=OFF).execute("<tool>name_opening</tool>").startswith("error")


def test_openings_plugin_identifies_the_line():
    ex = ToolExecutor(_ruy(), None, plugin_context=PC)
    assert ex.execute("<tool>name_opening</tool>") == "opening: Ruy Lopez"
    assert "Ruy Lopez" in ex.execute("<tool>opening_ideas</tool>")
    # a different line resolves to a different opening (not a fixed string)
    g2 = Game()
    for san in ["e4", "c5"]:
        g2.move(san)
    assert "Sicilian" in ToolExecutor(g2, None, plugin_context=PC).execute("<tool>name_opening</tool>")


def test_plugin_skill_body_loads_via_load_skill():
    ex = ToolExecutor(_ruy(), None, plugin_context=PC)
    body = ex.execute("<tool>load_skill name=opening-advisor</tool>")
    assert "opening-advisor" in body and "name_opening" in body
    # disabled -> the body is not available
    assert ToolExecutor(_ruy(), None, plugin_context=OFF).execute(
        "<tool>load_skill name=opening-advisor</tool>") == "error: unknown_skill"


def test_plugin_skill_called_as_a_tool_gets_corrective_error():
    out = ToolExecutor(_ruy(), None, plugin_context=PC).execute("<tool>opening-advisor</tool>")
    assert "is a skill, not a tool" in out and "load_skill name=opening-advisor" in out


def test_analysis_accuracy_report_runs_on_a_real_game():
    # real engine — replays Fool's-mate-ish moves and scores them
    from backend.engine import Engine
    g = Game()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        if not g.move(san).startswith("success"):
            break
    ex = ToolExecutor(g, Engine(), plugin_context=PC)
    rep = ex.execute("<tool>accuracy_report depth=10</tool>")
    assert rep.startswith("accuracy:") and "white=" in rep and "black=" in rep
    bl = ex.execute("<tool>find_blunders depth=10</tool>")
    assert bl.startswith("blunders:")
