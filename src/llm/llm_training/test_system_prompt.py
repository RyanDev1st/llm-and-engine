from llm_training.system_prompt import BASE_HARNESS, build_system

SK = [
    {"name": "chess-coach", "description": "Analyze positions, choose moves.",
     "plugin": "chess-official", "source": "official_plugin", "enabled": True},
    {"name": "hood-human-chat", "description": "Normalize slang before routing.",
     "plugin": "user-skills", "source": "user_skill", "enabled": True},
]
TM = [
    {"name": "load_skill", "description": "Load a listed skill's body.",
     "args": {"name": "required"}, "applies_when": "always", "plugin": "chess-official", "enabled": True},
    {"name": "move", "description": "Play a move.",
     "args": {"san": "required"}, "applies_when": "always", "plugin": "chess-official", "enabled": True},
]
PC = {"installed": ["chess-official", "user-skills"], "enabled": ["chess-official"], "marketplace": []}


def test_system_lists_skill_index_names_and_descriptions():
    s = build_system(SK, TM, PC)
    assert "chess-coach" in s and "Analyze positions" in s
    assert "hood-human-chat" in s


def test_system_lists_every_tool_name():
    s = build_system(SK, TM, PC)
    assert "load_skill" in s
    assert "move" in s


def test_system_includes_plugin_context():
    s = build_system(SK, TM, PC)
    assert "installed=" in s
    assert "chess-official" in s


def test_system_starts_with_base_harness():
    s = build_system(SK, TM, PC)
    assert BASE_HARNESS.strip().splitlines()[0] in s


def test_empty_envelope_falls_back_to_base_only():
    s = build_system([], [], {})
    assert BASE_HARNESS.strip().splitlines()[0] in s
    assert "AVAILABLE SKILLS" not in s
    assert "AVAILABLE TOOLS" not in s


def test_customization_overlay_renders_when_present():
    s = build_system(SK, TM, PC, agent_overlay="Be terse and end with a fact.")
    assert "CUSTOMIZATION" in s
    assert "Be terse and end with a fact." in s
    # overlay comes AFTER the harness/catalog (instruction hierarchy + cache prefix)
    assert s.index("CUSTOMIZATION") > s.index("AVAILABLE TOOLS")


def test_customization_overlay_absent_by_default():
    assert "CUSTOMIZATION" not in build_system(SK, TM, PC)


def test_empty_overlay_is_byte_identical_to_no_overlay():
    # Option A: default-empty overlay must not change the trained system text
    # (no train/serve drift). Train-time loader calls build_system without it.
    assert build_system(SK, TM, PC, agent_overlay="") == build_system(SK, TM, PC)
    assert build_system(SK, TM, PC, agent_overlay="   ") == build_system(SK, TM, PC)
