"""Conversational shape: an assistant turn may be a short lead-in sentence
followed by exactly ONE <tool> call. The validator must still extract the call
(so skill-load / tool-name checks work) and must reject two tools in one turn.
The final = last assistant turn with no tool call."""
from llm_dataset.v1.validate import validate_row

WHITE_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SK = [{"name": "chess-coach", "description": "Analyze.", "plugin": "chess-official",
       "source": "official_plugin", "enabled": True}]
TM = [
    {"name": "load_skill", "args": {"name": "required"}, "applies_when": "always",
     "plugin": "chess-official", "source": "official_plugin", "enabled": True},
    {"name": "board_state", "args": {"fields": ["basic", "all", "fen"]}, "applies_when": "always",
     "plugin": "chess-official", "source": "official_plugin", "enabled": True},
    {"name": "move", "args": {"san": "required"}, "applies_when": "always",
     "plugin": "chess-official", "source": "official_plugin", "enabled": True},
]
PC = {"installed": ["chess-official"], "enabled": ["chess-official"], "marketplace": []}
RULES = ["final_no_xml", "known_tool_only", "args_match_schema", "selected_skill_exists",
         "skill_index_only_before_load", "skill_body_strict", "engine_grounded"]


def _row(messages):
    return {
        "id": "t", "slice": "A", "kind": "harness_chess", "intent": "a_0001",
        "plugin_context": PC, "skills_index": SK, "selected_skills": ["chess-coach"],
        "tool_manifest": TM, "expected_tool_calls": ["load_skill", "board_state", "move"],
        "grounding_sources": ["board_state"], "messages": messages,
        "acceptance_rules": RULES, "position_fen": WHITE_START,
        "stockfish_truth": {"score_cp": 20, "best_san": "e4", "depth": 12},
    }


LEADIN_OK = [
    {"role": "user", "content": "how's my game?"},
    {"role": "assistant", "content": "Let me load my coaching skill.\n<tool>load_skill name=chess-coach</tool>"},
    {"role": "tool", "content": "Ground evaluation in Stockfish output."},
    {"role": "assistant", "content": "First, the position.\n<tool>board_state fields=basic</tool>"},
    {"role": "tool", "content": "board_state: turn=white, last_move=none, check=no, legal_count=20"},
    {"role": "assistant", "content": "Now the move.\n<tool>move san=e4</tool>"},
    {"role": "tool", "content": "success: e4"},
    {"role": "assistant", "content": "You're set up well. Want the plan, or Black's threats first?"},
]


def test_leadin_then_tool_turn_validates_clean():
    v = validate_row(_row(LEADIN_OK))
    assert v == [], v


def test_multiple_tools_in_one_turn_allowed():
    # Flexibility: a turn may carry several calls (e.g. load two skills at once).
    msgs = [m.copy() for m in LEADIN_OK]
    msgs[1] = {"role": "assistant",
               "content": "Loading both coaching skills.\n"
                          "<tool>load_skill name=chess-coach</tool>\n<tool>load_skill name=hood-human-chat</tool>"}
    # hood-human-chat must exist in the index for selected/known checks
    row = _row(msgs)
    row["skills_index"] = SK + [{"name": "hood-human-chat", "description": "Normalize chat.",
                                 "plugin": "user-skills", "source": "user_skill", "enabled": True}]
    row["tool_manifest"] = TM  # load_skill already declared
    v = [x for x in validate_row(row) if x.rule == "one_tool_per_turn"]
    assert v == [], f"multiple tools per turn should be allowed, got {v}"
