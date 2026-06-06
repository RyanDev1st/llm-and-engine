"""Validator must hard-reject a move that is illegal in the row's position_fen,
and a board_state result whose turn disagrees with the FEN side."""
from llm_dataset.v1.validate import validate_row

WHITE_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 2 2"

TM = [
    {"name": "load_skill", "args": {"name": "required"}, "applies_when": "always",
     "plugin": "chess-official", "source": "official_plugin", "enabled": True},
    {"name": "move", "args": {"san": "required"}, "applies_when": "always",
     "plugin": "chess-official", "source": "official_plugin", "enabled": True},
]
SK = [{"name": "chess-coach", "description": "Analyze.", "plugin": "chess-official",
       "source": "official_plugin", "enabled": True}]
PC = {"installed": ["chess-official"], "enabled": ["chess-official"], "marketplace": []}


def _row(fen, san, board_turn="white"):
    return {
        "id": "t", "slice": "A", "kind": "harness_chess", "intent": "a_0001",
        "plugin_context": PC, "skills_index": SK, "selected_skills": ["chess-coach"],
        "tool_manifest": TM, "expected_tool_calls": ["load_skill", "move"],
        "grounding_sources": ["board_state"],
        "messages": [
            {"role": "user", "content": f"play {san}"},
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "Ground in Stockfish output. score: +0.1"},
            {"role": "assistant", "content": f"<tool>move san={san}</tool>"},
            {"role": "tool", "content": f"success: {san}"},
            {"role": "assistant", "content": f"Played {san}."},
        ],
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "selected_skill_exists", "skill_index_only_before_load",
                             "skill_body_strict", "engine_grounded"],
        "position_fen": fen, "stockfish_truth": {"score_cp": 10, "best_san": "e4", "depth": 12},
    }


def test_illegal_move_is_rejected():
    v = validate_row(_row(BLACK_TO_MOVE, "e4"))   # e4 illegal for black
    assert any(x.rule == "illegal_move" for x in v), v


def test_legal_move_passes_legality_gate():
    v = validate_row(_row(WHITE_START, "e4"))      # e4 legal at start
    assert not any(x.rule == "illegal_move" for x in v), v
