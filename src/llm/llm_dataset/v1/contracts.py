from __future__ import annotations

SLICES = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "V1_A_skill_index_selection",
    "V1_B_skill_conflict_and_absence",
    "V1_C_dynamic_tool_schema",
    "V1_D_tool_unavailable_and_readonly",
    "V1_E_board_grounding",
    "V1_F_special_chess_rules",
    "V1_G_multi_tool_budget",
    "V1_H_error_recovery",
    "V1_I_eval_language",
    "V1_J_no_tool_and_mixed_intent",
    "V1_K_adversarial_injection",
    "V1_L_rejects_and_audit_fixtures",
)

FINAL_NO_XML = "final_no_xml"
KNOWN_TOOL_ONLY = "known_tool_only"
ARGS_MATCH_SCHEMA = "args_match_schema"
MAX_SIX_TOOL_CALLS = "max_six_tool_calls"
NO_EXACT_DUPLICATE_CALL = "no_exact_duplicate_call"
SKILL_INDEX_ONLY_BEFORE_LOAD = "skill_index_only_before_load"
SELECTED_SKILL_EXISTS = "selected_skill_exists"
BOARD_CLAIM_GROUNDED = "board_claim_grounded"
START_POSITION_EQUAL = "start_position_equal"
CLOSE_EVAL_EQUAL_LANGUAGE = "close_eval_equal_language"
TOOL_TEXT_IS_DATA = "tool_text_is_data"

HARNESS_RULES = (
    "applies_when_respected",
    "plugin_only_tools",
    "skill_body_strict",
    "engine_grounded",
)

ROW_KINDS = ("harness_chess", "universality")

OFFICIAL_PLUGIN = "chess-official"
USER_SKILLS_PLUGIN = "user-skills"

REAL_TOOL_NAMES = (
    "move", "eval", "best_move", "review_move", "threats",
    "legal_moves", "undo", "list_pieces", "ask_chessbot",
    "load_skill", "board_state",
)

RULES = (
    FINAL_NO_XML, KNOWN_TOOL_ONLY, ARGS_MATCH_SCHEMA, MAX_SIX_TOOL_CALLS,
    NO_EXACT_DUPLICATE_CALL, SKILL_INDEX_ONLY_BEFORE_LOAD, SELECTED_SKILL_EXISTS,
    BOARD_CLAIM_GROUNDED, START_POSITION_EQUAL, CLOSE_EVAL_EQUAL_LANGUAGE,
    TOOL_TEXT_IS_DATA,
) + HARNESS_RULES

REQUIRED_FIELDS = (
    "id", "slice", "kind", "intent",
    "plugin_context", "skills_index", "selected_skills",
    "tool_manifest", "expected_tool_calls", "grounding_sources",
    "messages", "acceptance_rules",
)

VALID_ROLES = {"system", "user", "assistant", "tool"}
MAX_TOOL_CALLS = 6
