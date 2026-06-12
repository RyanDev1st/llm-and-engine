"""The deterministic routing layer must fire on clear intent and stay silent
otherwise (no nudge when the user's words don't map to a tool)."""
from backend.tool_hints import routing_hints


def _tools(msg):
    h = routing_hints(msg)
    return [ln.split("`")[1] for ln in h.splitlines() if ln.startswith("- ")]


def test_play_a_named_move_hints_move_with_san():
    h = routing_hints("play b3")
    assert "move" in _tools("play b3")
    assert "san=b3" in h                      # concrete SAN extracted
    assert "best_move" not in _tools("play b3")  # naming a move != asking what to play


def test_piece_move_san_extracted():
    assert "san=Nf3" in routing_hints("make the move Nf3")
    assert "san=O-O" in routing_hints("castle kingside")
    assert "san=O-O-O" in routing_hints("castle queenside")


def test_ask_what_to_play_hints_best_move_not_move():
    t = _tools("what should I play here?")
    assert t == ["best_move"]


def test_eval_intent_hints_eval():
    for msg in ["how am I doing?", "who's winning", "is this lost?", "evaluate the position"]:
        assert "eval" in _tools(msg), msg


def test_other_default_tools():
    assert "threats" in _tools("any threats I should worry about?")
    assert "review_move" in _tools("did I blunder?")
    assert "undo" in _tools("take back that move")
    assert "legal_moves" in _tools("what are the legal moves?")
    assert "list_pieces" in _tools("what pieces do I have left?")


def test_load_fen_on_keyword_and_raw_fen():
    assert "load_fen" in _tools("set up this position for me")
    assert "load_fen" in _tools("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


def test_silent_when_no_intent():
    assert routing_hints("hi there, nice to meet you") == ""
    assert routing_hints("thanks!") == ""
    assert routing_hints("") == ""


def test_game_over_short_circuits_to_state_hint():
    # On a finished game the model should state the result, not call analysis tools.
    h = routing_hints("how am I doing?", game_over="checkmate")
    assert "GAME STATE" in h and "checkmate" in h
    assert "ROUTING HINT" not in h          # eval hint suppressed
    assert "<tool>eval" not in h


# --- extract_call recovery (regressions from the live audit) ---
from backend.inference import extract_call


def test_extract_call_recovers_tool_code_and_echoes():
    assert "<tool>review_move" in extract_call("I'll review. <tool_code>review_move depth=12</tool_code>")
    assert "<tool>move san=Nf3" in extract_call("call: move san=Nf3</tool>")          # hint echo, opening tag dropped
    assert "<tool>move san=b3" in extract_call("Play it. <move san=b3</tool>")          # malformed wrapper
    assert extract_call("I would remove the rook and improve my position.") is None     # prose, not a call
    assert extract_call("Your best move is Nf3.") is None                                # plain reply
