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


def test_arg_taking_skill_bodies_say_extract_dont_re_ask():
    # Transcript bug: "scale my cookie recipe from 12 up to 30 servings" -> recipe-scaler loaded ->
    # the model ASKED for the servings the user already gave. The body's "ask if not given" wording
    # induced over-asking; it must now tell the model to EXTRACT args from the message and not re-ask.
    bodies = {s["name"]: s["body"] for s in L.SKILLS}
    rs = bodies["recipe-scaler"].lower()
    assert "do not ask for numbers the user already gave" in rs
    assert "from_servings=12" in rs                     # a concrete extract-from-message example
    assert "don't ask for a number they gave" in bodies["guitar-tutor"].lower()
    assert "default to 60" in bodies["breathing-coach"].lower()


def test_dropped_breathing_result_is_grounded_in_the_loop():
    # The transcript dropped the breathing_timer result from the final answer ("Breathing helps
    # reset the nervous system. Ready to try it?" with NO duration). Consumer C grounding must
    # append the real fact. Full live loop, engine-free (breathing_timer needs no engine).
    from backend.inference import CoachLoop

    class Scripted:
        def __init__(self, steps):
            self.steps = list(steps); self.i = 0
        def generate(self, messages, max_new_tokens, stop):
            out = self.steps[min(self.i, len(self.steps) - 1)]; self.i += 1
            return out

    loop = CoachLoop(Scripted(["<tool>breathing_timer seconds=10",
                               "Breathing helps reset the nervous system. Ready to try it?"]),
                     ToolExecutor(Game(), None, PC), plugin_context=PC)
    out = loop.respond([], "stressed af rn need to chill n breathe for a bit")
    assert any(r.startswith("breathing_timer:") for r in out["tool_results"])
    assert "10s" in out["reply"]                        # the dropped fact is now grounded


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


def test_corrective_error_shows_the_real_arg_schema():
    # The corrective error must carry the tool's REAL args from the live manifest, not a literal
    # '...' placeholder — otherwise the model guesses arg names (seen: it invented `seconds=10`).
    ex = ToolExecutor(Game(), None, PC)
    out = ex.execute("<tool>load_skill name=breathing_timer</tool>")
    assert "<tool>breathing_timer seconds=<seconds></tool>" in out
    assert "..." not in out
    # a multi-arg tool lists every required arg
    out2 = ex.execute("<tool>load_skill name=scale_recipe</tool>")
    assert "from_servings=<from_servings>" in out2 and "to_servings=<to_servings>" in out2
