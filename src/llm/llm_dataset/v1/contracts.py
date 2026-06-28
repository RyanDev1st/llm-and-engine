from __future__ import annotations

# v5 pure-chess slice set: chess coaching core (A-K) + multi-turn + the chess-refocused
# keystones. The cross-domain universality/marketplace/routing slices (V1_A-O, V1_Q) were
# dropped with their renderers (archived to legacy [ignore]/).
SLICES = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "V1_P_multiturn_followup",
    "V1_R_compute_grounding",
    "V1_S_compound_plan",
    "V1_T_audited_plan",
    "V1_U_specialist_routing",
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
NARRATION_GROUNDED = "narration_grounded"

HARNESS_RULES = (
    "applies_when_respected",
    "plugin_only_tools",
    "skill_body_strict",
    "engine_grounded",
    "human-chat helper accepted coverage",
    "multi-skill composition accepted coverage",
)

ROW_KINDS = ("harness_chess", "compute", "compound_plan", "audited_plan", "specialist_routing")

OFFICIAL_PLUGIN = "chess-official"
USER_SKILLS_PLUGIN = "user-skills"

# The full FLAT pure-chess tool name set (no plugin gating — the serve harness
# aggregates plugins into one flat catalog the model sees, so training mirrors that).
# Capabilities that were plugin tools (openings/analysis/puzzles) are listed here as
# plain chess tools; what_if is the one genuinely new coach tool. Kept in sync with
# catalog.OFFICIAL_TOOLS + the position tools — that catalog is the live source of truth.
REAL_TOOL_NAMES = (
    "move", "eval", "best_move", "what_if", "review_move", "threats",
    "legal_moves", "undo", "list_pieces", "ask_chessbot", "board_state",
    "new_game", "load_fen", "random_position", "fetch_puzzle",
    "name_opening", "opening_ideas", "accuracy_report", "find_blunders",
    "python",
)
# Skills are loaded with the <skill>NAME</skill> verb, NOT a load_skill tool.
SKILL_VERB_OPEN, SKILL_VERB_CLOSE = "<skill>", "</skill>"

# Stage 1/2 PLAN-mode deterministic gates: goal committed before the checklist, and
# every checkbox binding maps to a real listed skill/tool.
GOAL_BEFORE_PLAN = "goal_before_plan"
PLAN_BOXES_BOUND = "plan_boxes_bound"
# Stage 2 audit gate: a tool-checkable box must be CLOSED by a real python audit
# (the executor is the source of truth), not asserted.
AUDIT_BOXES_GROUNDED = "audit_boxes_grounded"

RULES = (
    FINAL_NO_XML, KNOWN_TOOL_ONLY, ARGS_MATCH_SCHEMA, MAX_SIX_TOOL_CALLS,
    NO_EXACT_DUPLICATE_CALL, SKILL_INDEX_ONLY_BEFORE_LOAD, SELECTED_SKILL_EXISTS,
    BOARD_CLAIM_GROUNDED, START_POSITION_EQUAL, CLOSE_EVAL_EQUAL_LANGUAGE,
    TOOL_TEXT_IS_DATA, NARRATION_GROUNDED, GOAL_BEFORE_PLAN, PLAN_BOXES_BOUND,
    AUDIT_BOXES_GROUNDED,
) + HARNESS_RULES

REQUIRED_FIELDS = (
    "id", "slice", "kind", "intent",
    "plugin_context", "skills_index", "selected_skills",
    "tool_manifest", "expected_tool_calls", "grounding_sources",
    "messages", "acceptance_rules",
)

VALID_ROLES = {"system", "user", "assistant", "tool"}
MAX_TOOL_CALLS = 6
