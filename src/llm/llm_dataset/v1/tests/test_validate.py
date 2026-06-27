from llm_dataset.v1.renderer.tags import skill_call_msg, tool_call_msg
from llm_dataset.v1.validate import validate_row

# v5-native: actions are STRUCTURED tool_calls on the assistant message, built via
# the same helpers the renderers use. A skill load is the native tool load_skill{name}.


def _final(text):
    return {"role": "assistant", "content": text}


def _toolres(text):
    return {"role": "tool", "content": text}


def good_row(**overrides):
    row = {
        "id": "v1_a_001",
        "slice": "A",
        "kind": "harness_chess",
        "intent": "select chess skill",
        "plugin_context": {"installed": ["chess-official"], "enabled": ["chess-official"]},
        "skills_index": [{"name": "chess-coach", "description": "Analyze chess positions."}],
        "selected_skills": ["chess-coach"],
        "tool_manifest": [
            {"name": "load_skill", "description": "Load skill.", "args": {"name": "required"}, "applies_when": "always"},
            {"name": "board_state", "description": "Read board.", "args": {"fields": ["basic", "all"]}, "applies_when": "always"},
        ],
        "expected_tool_calls": ["load_skill", "board_state"],
        "grounding_sources": ["board_state"],
        "messages": [
            {"role": "user", "content": "What is happening on the board?"},
            skill_call_msg("chess-coach"),
            _toolres("Use board tools before board claims."),
            tool_call_msg("board_state", {"fields": "basic"}),
            _toolres("board_state: turn=white, check=no"),
            _final("White to move, and the king is not in check."),
        ],
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema", "selected_skill_exists", "board_claim_grounded"],
    }
    row.update(overrides)
    return row


def rules(row):
    return {v.rule for v in validate_row(row)}


def test_accepts_valid_grounded_row():
    assert validate_row(good_row()) == []


def test_rejects_unknown_tool():
    row = good_row(messages=[tool_call_msg("move", {"san": "e4"})])
    assert "known_tool_only" in rules(row)


def test_rejects_duplicate_and_over_budget_calls():
    calls = [tool_call_msg("board_state", {"fields": "basic"}) for _ in range(7)]
    row = good_row(messages=calls)
    assert {"max_six_tool_calls", "no_exact_duplicate_call"} <= rules(row)


def test_rejects_final_xml():
    row = good_row(messages=good_row()["messages"][:-1] + [_final("Here: <tool>eval</tool>")])
    assert "final_no_xml" in rules(row)


def test_rejects_overstated_close_eval():
    row = good_row(
        acceptance_rules=["close_eval_equal_language"],
        messages=[_final("White is slightly better at +0.12.")],
    )
    assert "close_eval_equal_language" in rules(row)


def test_rejects_tool_outside_manifest():
    row = good_row()
    row["tool_manifest"] = [t for t in row["tool_manifest"] if t["name"] != "board_state"]
    assert "plugin_only_tools" in rules(row)


def test_rejects_review_without_history():
    row = good_row(
        tool_manifest=[
            {"name": "load_skill", "description": "...", "args": {"name": "required"}, "applies_when": "always"},
            {"name": "review_move", "description": "...", "args": {"depth": "required"}, "applies_when": "has_history"},
        ],
        messages=[
            {"role": "user", "content": "review my move"},
            skill_call_msg("chess-coach"),
            _toolres("lesson"),
            tool_call_msg("review_move", {"depth": 12}),
            _toolres("review: e4, label=good, delta=+0.05 pawns, best_was=e4"),
            _final("Solid."),
        ],
    )
    assert "applies_when_respected" in rules(row)


def test_accepts_review_with_move_history():
    row = good_row(
        tool_manifest=[
            {"name": "load_skill", "description": "...", "args": {"name": "required"}, "applies_when": "always"},
            {"name": "move", "description": "...", "args": {"san": "required"}, "applies_when": "game_in_progress"},
            {"name": "review_move", "description": "...", "args": {"depth": "required"}, "applies_when": "has_history"},
        ],
        expected_tool_calls=["load_skill", "move", "review_move"],
        grounding_sources=[],
        messages=[
            {"role": "user", "content": "play e4 then review it"},
            skill_call_msg("chess-coach"),
            _toolres("lesson"),
            tool_call_msg("move", {"san": "e4"}),
            _toolres("move: success san=e4"),
            tool_call_msg("review_move", {"depth": 12}),
            _toolres("review: e4, label=good, delta=+0.05 pawns, best_was=e4"),
            _final("Solid."),
        ],
        acceptance_rules=["final_no_xml", "known_tool_only", "args_match_schema", "selected_skill_exists", "applies_when_respected"],
    )
    assert validate_row(row) == []


def test_accepts_review_with_success_colon_move_history():
    row = good_row(
        tool_manifest=[
            {"name": "load_skill", "description": "...", "args": {"name": "required"}, "applies_when": "always"},
            {"name": "move", "description": "...", "args": {"san": "required"}, "applies_when": "game_in_progress"},
            {"name": "review_move", "description": "...", "args": {"depth": "required"}, "applies_when": "has_history"},
        ],
        expected_tool_calls=["load_skill", "move", "review_move"],
        grounding_sources=[],
        messages=[
            {"role": "user", "content": "play e4 then review it"},
            skill_call_msg("chess-coach"),
            _toolres("lesson"),
            tool_call_msg("move", {"san": "e4"}),
            _toolres("success: e4"),
            tool_call_msg("review_move", {"depth": 12}),
            _toolres("review: e4, label=good, delta=+0.05 pawns, best_was=e4"),
            _final("Solid."),
        ],
        acceptance_rules=["final_no_xml", "known_tool_only", "args_match_schema", "selected_skill_exists", "applies_when_respected"],
    )
    assert validate_row(row) == []

    # review claimed against a lesson that only MENTIONS history (no real prior move) -> rejected
    row = good_row(
        tool_manifest=[
            {"name": "load_skill", "description": "...", "args": {"name": "required"}, "applies_when": "always"},
            {"name": "review_move", "description": "...", "args": {"depth": "required"}, "applies_when": "has_history"},
        ],
        messages=[
            {"role": "user", "content": "review my move"},
            skill_call_msg("chess-coach"),
            _toolres("lesson success criteria mention move history"),
            tool_call_msg("review_move", {"depth": 12}),
            _toolres("review: none"),
            _final("No prior move."),
        ],
        acceptance_rules=["final_no_xml", "known_tool_only", "args_match_schema", "selected_skill_exists", "applies_when_respected"],
    )
    assert "applies_when_respected" in rules(row)


def test_rejects_tool_from_disabled_plugin():
    row = good_row(
        plugin_context={"installed": ["chess-official", "market-tactics"], "enabled": ["chess-official"]},
        tool_manifest=good_row()["tool_manifest"] + [
            {"name": "market_scan", "description": "Scan tactics.", "args": {}, "applies_when": "always", "plugin": "market-tactics", "source": "marketplace_plugin", "enabled": False}
        ],
        messages=[tool_call_msg("market_scan", {})],
    )
    assert "plugin_only_tools" in rules(row)


def test_rejects_selected_skill_from_uninstalled_plugin():
    row = good_row(
        plugin_context={"installed": ["chess-official"], "enabled": ["chess-official"]},
        skills_index=good_row()["skills_index"] + [
            {"name": "market-tactics", "description": "Marketplace tactics.", "plugin": "market-tactics", "source": "marketplace_plugin", "enabled": True}
        ],
        selected_skills=["market-tactics"],
        messages=[skill_call_msg("market-tactics")],
    )
    assert "selected_skill_exists" in rules(row)


def test_rejects_helper_tool_before_skill_loaded():
    row = good_row(
        skills_index=good_row()["skills_index"] + [
            {"name": "hood-human-chat", "description": "Normalize chat.", "plugin": "user-skills", "source": "user_skill", "enabled": True}
        ],
        selected_skills=["hood-human-chat", "chess-coach"],
        plugin_context={"installed": ["chess-official", "user-skills"], "enabled": ["chess-official", "user-skills"]},
        tool_manifest=good_row()["tool_manifest"] + [
            {"name": "normalize_human_chat", "description": "Normalize chat.", "args": {"text": "required"}, "applies_when": "always"}
        ],
        messages=[
            {"role": "user", "content": "yo what's up, am I cooked?"},
            tool_call_msg("normalize_human_chat", {"text": "messy_user_chat"}),
            _toolres("normalized: chess help needed"),
            skill_call_msg("hood-human-chat"),
            _toolres("normalize first"),
            skill_call_msg("chess-coach"),
            _toolres("use board tools before claims"),
            _final("Need board state first."),
        ],
        acceptance_rules=good_row()["acceptance_rules"] + ["skill_body_strict", "skill_index_only_before_load"],
    )
    assert {"skill_body_strict", "skill_index_only_before_load"} <= rules(row)


def test_rejects_irrelevant_skill_loaded_before_selected_skills():
    row = good_row(
        skills_index=good_row()["skills_index"] + [
            {"name": "hood-human-chat", "description": "Normalize chat.", "plugin": "user-skills", "source": "user_skill", "enabled": True},
            {"name": "cooking-helper", "description": "Recipe help.", "plugin": "user-skills", "source": "user_skill", "enabled": True},
        ],
        selected_skills=["hood-human-chat", "chess-coach"],
        plugin_context={"installed": ["chess-official", "user-skills"], "enabled": ["chess-official", "user-skills"]},
        messages=[
            {"role": "user", "content": "need chess help"},
            skill_call_msg("cooking-helper"),
            _toolres("recipes only"),
            skill_call_msg("hood-human-chat"),
            _toolres("normalize vague chat"),
            skill_call_msg("chess-coach"),
            _toolres("use board tools before claims"),
            _final("Need board state first."),
        ],
        acceptance_rules=good_row()["acceptance_rules"] + ["skill_body_strict"],
    )
    assert "skill_body_strict" in rules(row)


def test_rejects_selected_helper_skill_from_disabled_plugin():
    row = good_row(
        plugin_context={"installed": ["chess-official", "user-skills"], "enabled": ["chess-official"]},
        skills_index=good_row()["skills_index"] + [
            {"name": "hood-human-chat", "description": "Normalize chat.", "plugin": "user-skills", "source": "user_skill", "enabled": True}
        ],
        selected_skills=["hood-human-chat", "chess-coach"],
        messages=[
            skill_call_msg("hood-human-chat"),
            skill_call_msg("chess-coach"),
            _final("Need enabled helper first."),
        ],
    )
    assert "selected_skill_exists" in rules(row)


def test_rejects_helper_tool_from_disabled_plugin():
    row = good_row(
        plugin_context={"installed": ["chess-official", "user-skills"], "enabled": ["chess-official"]},
        skills_index=good_row()["skills_index"] + [
            {"name": "hood-human-chat", "description": "Normalize chat.", "plugin": "user-skills", "source": "user_skill", "enabled": True}
        ],
        selected_skills=["hood-human-chat", "chess-coach"],
        tool_manifest=good_row()["tool_manifest"] + [
            {"name": "normalize_human_chat", "description": "Normalize chat.", "args": {"text": "required"}, "applies_when": "always", "plugin": "user-skills", "source": "user_skill", "enabled": True}
        ],
        messages=[
            skill_call_msg("hood-human-chat"),
            tool_call_msg("normalize_human_chat", {"text": "messy_user_chat"}),
            skill_call_msg("chess-coach"),
            _final("Need enabled helper first."),
        ],
    )
    assert "plugin_only_tools" in rules(row)


def test_rejects_missing_engine_evidence():
    row = good_row(acceptance_rules=good_row()["acceptance_rules"] + ["engine_grounded"])
    assert "engine_grounded" in rules(row)
