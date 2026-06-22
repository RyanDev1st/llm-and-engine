"""The plugin-aware deterministic-layer foundation: a DOMAIN-NEUTRAL view of the live tool
surface (manifest_view). The chess harness helpers (recovery, corrective errors, result-
grounding) were hardcoded to the chess tool set; these primitives let them read the live
manifest for whatever plugins are enabled, WITHOUT per-domain regex."""
from backend.manifest_view import full_manifest, live_tool_names, tool_schema, generic_result_signal

_LIFE = {"enabled": ["life-skills"]}


# --- live surface: names + schema track the enabled plugins ---

def test_live_names_include_chess_always_and_plugins_when_enabled():
    chess_only = live_tool_names(None)
    assert {"eval", "best_move", "move", "python"} <= chess_only      # official + compute
    assert "convert_units" not in chess_only                          # life-skills off
    with_life = live_tool_names(_LIFE)
    assert {"convert_units", "scale_recipe", "metronome_bpm", "breathing_timer"} <= with_life
    assert {"eval", "python"} <= with_life                            # chess stays callable too


def test_full_manifest_entries_carry_name_and_args():
    by_name = {t["name"]: t for t in full_manifest(_LIFE)}
    assert by_name["breathing_timer"]["args"] == {"seconds": "required"}
    assert by_name["eval"]["args"] == {"depth": "required"}


def test_tool_schema_reads_args_from_the_live_manifest():
    assert tool_schema(_LIFE, "breathing_timer") == {"seconds": "required"}
    assert tool_schema(_LIFE, "scale_recipe") == {"from_servings": "required",
                                                   "to_servings": "required"}
    # a plugin tool is invisible to schema when its bundle is off
    assert tool_schema(None, "breathing_timer") is None
    assert tool_schema(_LIFE, "no_such_tool") is None


# --- generic_result_signal: the distinctive grounded token from ANY `name: ...` result ---

def test_signal_prefers_the_value_after_an_equals():
    # a conversion's grounded answer is the RESULT, after '=' (not the input)
    assert generic_result_signal("convert: 5 miles = 8.047 kilometers (length)") == "8.047"
    assert generic_result_signal("metronome_bpm: 120 bpm = 500.0 ms per beat") == "500.0"
    assert generic_result_signal("convert: 20 C = 68 F") == "68"


def test_signal_falls_back_to_a_compact_number_unit_then_a_number():
    assert generic_result_signal("scale_recipe: multiply every ingredient by 2x (from 2 to 4 servings)") == "2x"
    assert generic_result_signal("breathing_timer: 10s set — about 0 slow 4-7-8 breath cycle(s).") == "10s"


def test_signal_is_none_for_errors_and_numberless_results():
    assert generic_result_signal("error: breathing_timer needs a number of seconds") is None
    assert generic_result_signal("pieces: white king on e1") is None
    assert generic_result_signal("") is None
