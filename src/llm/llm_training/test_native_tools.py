from llm_training.system_prompt import build_system


def test_manifest_to_native_tools_maps_required_and_enum_args():
    from llm_training.native_tools import manifest_to_native_tools

    tools = manifest_to_native_tools([
        {"name": "best_move", "description": "Find best move.",
         "args": {"depth": "required", "top": ["1", "2", "3"]}},
        {"name": "new_game", "description": "Reset.", "args": {}},
    ])
    best = tools[0]["function"]
    assert best["name"] == "best_move"
    assert best["parameters"]["required"] == ["depth"]
    assert best["parameters"]["properties"]["depth"]["type"] == "integer"
    assert best["parameters"]["properties"]["top"]["enum"] == [1, 2, 3]
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_manifest_to_native_tools_can_compact_descriptions():
    from llm_training.native_tools import manifest_to_native_tools

    tools = manifest_to_native_tools([
        {"name": "eval", "description": "Evaluate the current chess position.",
         "args": {"depth": "required"}},
    ], include_descriptions=False)

    assert tools[0]["function"]["description"] == ""
    assert tools[0]["function"]["parameters"]["required"] == ["depth"]


def test_tools_from_system_prompt_recovers_served_manifest_text():
    from llm_training.native_tools import tools_from_system_prompt

    system = build_system([], [
        {"name": "load_skill", "description": "Load a skill.", "args": {"name": "required"}},
        {"name": "eval", "description": "Evaluate.", "args": {"depth": "required"}},
    ], {}, reasoning_mode="fast")
    names = [tool["function"]["name"] for tool in tools_from_system_prompt(system)]
    assert names == ["load_skill", "eval"]


def test_native_system_can_omit_prose_tool_list():
    system = build_system([], [
        {"name": "load_skill", "description": "Load a skill.", "args": {"name": "required"}},
        {"name": "eval", "description": "Evaluate.", "args": {"depth": "required"}},
    ], {}, reasoning_mode="auto", include_tools=False)

    assert "Reasoning mode: AUTO" in system
    assert "AVAILABLE TOOLS" not in system
    assert "eval depth" not in system
