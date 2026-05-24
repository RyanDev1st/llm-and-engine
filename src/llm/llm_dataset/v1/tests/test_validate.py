from llm_dataset.v1.validate import validate_row


def good_row(**overrides):
    row = {
        "id": "v1_a_001",
        "slice": "V1_A_skill_index_selection",
        "kind": "universality",
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
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "Use board tools before board claims."},
            {"role": "assistant", "content": "<tool>board_state fields=basic</tool>"},
            {"role": "tool", "content": "board_state: turn=white, check=no"},
            {"role": "assistant", "content": "White to move, and the king is not in check."},
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
    row = good_row(messages=[{"role": "assistant", "content": "<tool>move san=e4</tool>"}])
    assert "known_tool_only" in rules(row)


def test_rejects_duplicate_and_over_budget_calls():
    calls = [{"role": "assistant", "content": "<tool>board_state fields=basic</tool>"}] * 7
    row = good_row(messages=calls)
    assert {"max_six_tool_calls", "no_exact_duplicate_call"} <= rules(row)


def test_rejects_final_xml():
    row = good_row(messages=good_row()["messages"][:-1] + [{"role": "assistant", "content": "Here: <tool>eval</tool>"}])
    assert "final_no_xml" in rules(row)


def test_rejects_overstated_close_eval():
    row = good_row(
        acceptance_rules=["close_eval_equal_language"],
        messages=[{"role": "assistant", "content": "White is slightly better at +0.12."}],
    )
    assert "close_eval_equal_language" in rules(row)


def test_rejects_tool_outside_manifest():
    row = good_row()
    row["tool_manifest"] = [t for t in row["tool_manifest"] if t["name"] != "board_state"]
    assert "plugin_only_tools" in rules(row)


def test_rejects_review_without_history():
    row = good_row(
        position_fen=chess_starting_fen(),
        tool_manifest=[
            {"name": "load_skill", "description": "...", "args": {"name": "required"}, "applies_when": "always"},
            {"name": "review_move", "description": "...", "args": {"depth": "required"}, "applies_when": "has_history"},
        ],
        messages=[
            {"role": "user", "content": "review my move"},
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "lesson"},
            {"role": "assistant", "content": "<tool>review_move depth=12</tool>"},
            {"role": "tool", "content": "review: e4, label=good, delta=+0.05 pawns, best_was=e4"},
            {"role": "assistant", "content": "Solid."},
        ],
    )
    assert "applies_when_respected" in rules(row)


def chess_starting_fen() -> str:
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
