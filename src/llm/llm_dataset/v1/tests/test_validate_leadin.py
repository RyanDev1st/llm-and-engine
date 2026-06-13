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
    {"role": "assistant", "content": "Let me load my coaching skill.\n<skill>chess-coach</skill>"},
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


def test_two_tools_in_one_inference_message_rejected():
    # One tool per inference step: two <tool> in a SINGLE assistant message is invalid.
    msgs = [m.copy() for m in LEADIN_OK]
    msgs[1] = {"role": "assistant",
               "content": "Two at once.\n<skill>chess-coach</skill>\n<tool>board_state fields=basic</tool>"}
    v = validate_row(_row(msgs))
    assert any(x.rule == "one_tool_per_message" for x in v), v


def test_multiple_skills_across_messages_allowed():
    # Many calls across the agentic loop is fine — one per step.
    msgs = [
        {"role": "user", "content": "yo idk how's my game"},
        {"role": "assistant", "content": "Let me clean up the message.\n<skill>hood-human-chat</skill>"},
        {"role": "tool", "content": "Normalize messy chat. Ground in Stockfish output."},
        {"role": "assistant", "content": "Now the coaching skill.\n<skill>chess-coach</skill>"},
        {"role": "tool", "content": "Use board tools before claims."},
        {"role": "assistant", "content": "The position.\n<tool>board_state fields=basic</tool>"},
        {"role": "tool", "content": "board_state: turn=white, last_move=none, check=no, legal_count=20"},
        {"role": "assistant", "content": "You're set. Want the plan or threats first?"},
    ]
    row = _row(msgs)
    row["skills_index"] = SK + [{"name": "hood-human-chat", "description": "Normalize chat.",
                                 "plugin": "user-skills", "source": "user_skill", "enabled": True}]
    row["selected_skills"] = ["hood-human-chat", "chess-coach"]
    v = [x for x in validate_row(row) if x.rule == "one_tool_per_message"]
    assert v == [], f"one-per-step across messages should be fine, got {v}"
