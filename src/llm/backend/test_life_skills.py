"""life-skills bundle: the tools are REAL deterministic executors (not mocks), they dispatch
through the real ToolExecutor when the bundle is enabled, and the skills load real bodies. Guards
the math + the cross-bundle routing the benchmark's STRESS suite relies on."""
import math

from backend.game import Game
from backend.plugins import life_skills as L
from backend.tools import ToolExecutor

PC = {"installed": ["life-skills"], "enabled": ["life-skills"], "marketplace": []}


def test_convert_units_real_math():
    out = L.handle("convert_units", {"value": "5", "from_unit": "miles", "to_unit": "km"}, None)
    assert "8.047" in out                                   # 5 * 1.60934
    assert "212" in L.handle("convert_units", {"value": "100", "from_unit": "C", "to_unit": "F"}, None)
    assert "error" in L.handle("convert_units", {"value": "5", "from_unit": "miles", "to_unit": "kg"}, None)


def test_scale_metronome_breathing_real():
    assert "2.5" in L.handle("scale_recipe", {"from_servings": "12", "to_servings": "30"}, None)
    assert "500.0 ms" in L.handle("metronome_bpm", {"bpm": "120"}, None)   # 60000/120
    assert "90s" in L.handle("breathing_timer", {"seconds": "90"}, None)
    assert "error" in L.handle("scale_recipe", {"from_servings": "0", "to_servings": "10"}, None)


def test_not_my_tool_returns_none():
    assert L.handle("eval", {"depth": "18"}, None) is None    # registry routes on to another bundle


def test_dispatches_through_real_executor_when_enabled():
    ex = ToolExecutor(Game(), None, PC)
    assert ex.execute("<tool>convert_units value=5 from_unit=miles to_unit=km</tool>").startswith("convert:")
    # disabled bundle -> the tool is NOT callable (unknown_tool), proving the gate is real
    ex_off = ToolExecutor(Game(), None, {"installed": [], "enabled": [], "marketplace": []})
    assert "unknown_tool" in ex_off.execute("<tool>convert_units value=5 from_unit=miles to_unit=km</tool>")


def test_skills_have_real_bodies_and_load():
    ex = ToolExecutor(Game(), None, PC)
    for name in ("recipe-scaler", "guitar-tutor", "breathing-coach", "tax-filing-helper"):
        body = ex.execute(f"<tool>load_skill name={name}</tool>")
        assert not body.startswith("error") and name in body and len(body) > 80


def test_tool_loaded_as_skill_gets_corrective_error():
    # Symmetric to skill-as-tool: <skill>metronome_bpm</skill> (a TOOL emitted as a skill, seen
    # on OOD routing) must NOT dead-end at unknown_skill -> name the right verb so the loop self-
    # corrects to <tool>, instead of flailing back to the training-dominant skill.
    ex = ToolExecutor(Game(), None, PC)
    out = ex.execute("<tool>load_skill name=metronome_bpm</tool>")
    assert "is a tool, not a skill" in out and "<tool>metronome_bpm" in out
    # a genuinely unknown name still reports unknown_skill (no false coercion)
    assert ex.execute("<tool>load_skill name=not_a_real_thing</tool>") == "error: unknown_skill"
