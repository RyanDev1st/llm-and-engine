"""The deterministic routing layer must fire on clear intent and stay silent
otherwise (no nudge when the user's words don't map to a tool)."""
from backend.tool_hints import routing_hints, skill_hints


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


# --- skill-routing layer (generic, fires only on a distinctive skill name) ---
_TACTICS = {"name": "tactical-puzzles", "description": "solve and explain tactical puzzles"}
_ENDGAME = {"name": "endgame-drills", "description": "drill king-and-pawn endgames"}
_COACH = {"name": "chess-coach", "description": "analyze a position, choose moves, review mistakes"}


def test_skill_hint_fires_on_a_distinctively_named_skill():
    h = skill_hints("give me a tactical puzzle", [_TACTICS, _COACH])
    assert "load_skill name=tactical-puzzles" in h
    assert "chess-coach" not in h            # broad coach name tokens are stoplisted


def test_skill_hint_generalizes_to_any_dropped_in_skill():
    assert "load_skill name=endgame-drills" in skill_hints("let's practice some endgame drills", [_ENDGAME])
    assert "load_skill name=tactical-puzzles" in skill_hints("got a puzzle for me?", [_TACTICS])  # plural stem


def test_skill_hint_silent_on_broad_coach_and_off_topic():
    assert skill_hints("analyze my position and pick a move", [_COACH]) == ""   # all-generic name
    assert skill_hints("thanks, that helps!", [_TACTICS, _ENDGAME]) == ""
    assert skill_hints("", [_TACTICS]) == ""


# --- extract_call recovery (regressions from the live audit) ---
from backend.inference import extract_call


def test_extract_call_recovers_tool_code_and_echoes():
    assert "<tool>review_move" in extract_call("I'll review. <tool_code>review_move depth=12</tool_code>")
    assert "<tool>move san=Nf3" in extract_call("call: move san=Nf3</tool>")          # hint echo, opening tag dropped
    assert "<tool>move san=b3" in extract_call("Play it. <move san=b3</tool>")          # malformed wrapper
    assert extract_call("I would remove the rook and improve my position.") is None     # prose, not a call
    assert extract_call("Your best move is Nf3.") is None                                # plain reply


def test_extract_call_recovers_tagless_bare_call():
    # live audit: model emitted "review_move depth=1" as the whole reply, no tags
    assert extract_call("review_move depth=1") == "<tool>review_move depth=1</tool>"
    assert extract_call("eval depth=18") == "<tool>eval depth=18</tool>"
    # prose / one-word replies must NOT be mistaken for a bare call (no k=v args)
    assert extract_call("undo that move when you can") is None
    assert extract_call("eval looks roughly equal to me") is None
    assert extract_call("The best move here is e4.") is None


def test_extract_call_recovers_channel_token_form():
    # live leak: model emitted "<|tool_call>call:board_state fields=all" as the reply
    assert extract_call("<|tool_call>call:board_state fields=all") == "<tool>board_state fields=all</tool>"
    assert "<tool>eval depth=18" in extract_call("<|tool_call|>call: eval depth=18")
    # a plain reply that merely contains "call:" but no tool pattern stays a reply
    assert extract_call("Sure, you can call: that a solid opening.") is None


# --- coverage set (deterministic multi-tool guarantee) ---
from backend.tool_hints import matched_tools, matched_calls


def test_matched_tools_detects_compound_intent():
    t = matched_tools("give me the best move and the evaluation")
    assert t == {"best_move", "eval"}


def test_matched_calls_returns_canonical_calls():
    calls = matched_calls("play b3 and tell me the eval")
    assert calls["move"] == "<tool>move san=b3</tool>"
    assert calls["eval"].startswith("<tool>eval depth=")


def test_matched_tools_empty_on_no_intent():
    assert matched_tools("hi there") == set()
    assert matched_calls("") == {}


def test_plural_best_moves_detected_with_count():
    # the screenshot bug: "5 next best moves" must map to best_move (plural), with top=5
    t = matched_tools("can you eval and give me the 5 next best moves?")
    assert t == {"eval", "best_move"}
    assert matched_calls("give me the 5 next best moves")["best_move"] == "<tool>best_move depth=18 top=5</tool>"
    assert matched_calls("top 3 moves please")["best_move"] == "<tool>best_move depth=18 top=3</tool>"
    assert matched_calls("show me the five best moves")["best_move"] == "<tool>best_move depth=18 top=5</tool>"
    # singular still works, no spurious top
    assert matched_calls("what's the best move?")["best_move"] == "<tool>best_move depth=18</tool>"


def test_consecutive_moves_map_to_series_not_top():
    # "consecutive / line / in a row" -> a LINE (series), not N alternatives (top)
    assert "series=3" in matched_calls("give me the next 3 top moves, consecutive")["best_move"]
    assert "series=3" in matched_calls("show me the best line")["best_move"]
    assert "series=4" in matched_calls("4 moves in a row from here")["best_move"]
    # plain "3 best moves" (no line words) stays top
    assert "top=3" in matched_calls("give me the 3 best moves")["best_move"]


def test_filler_words_dont_break_the_match():
    # live leak: "suggest me the next 3 consecutive moves" matched NOTHING before
    assert matched_calls("can you suggest me the next 3 consecutive moves")["best_move"] == \
        "<tool>best_move depth=18 series=3</tool>"
    assert "top=3" in matched_calls("show me 3 moves please")["best_move"]
    assert "best_move" in matched_tools("give me a few good moves")
    # the filler whitelist must NOT swallow legal/possible/available -> stay legal_moves
    assert matched_tools("give me the legal moves") == {"legal_moves"}
    assert "best_move" not in matched_tools("what are the possible moves")
    assert matched_calls("nice moves you played") == {}      # prose, no request verb/count


def test_extract_call_recovers_loading_skill_gerund():
    # live leak: model emitted "loading_skill name=chess-coach" as the whole reply
    assert extract_call("loading_skill name=chess-coach") == "<tool>load_skill name=chess-coach</tool>"
    assert extract_call("load skill name=tactics") == "<tool>load_skill name=tactics</tool>"


def test_moves_without_the_word_best_detected():
    # the prefix-probe gap: "5 next moves" / "suggest 5 moves" (no word "best")
    assert matched_tools("suggest 5 next moves and tell me how am I doing") == {"best_move", "eval"}
    assert matched_calls("suggest 5 next moves")["best_move"] == "<tool>best_move depth=18 top=5</tool>"
    assert "best_move" in matched_tools("give me some moves")
    assert "best_move" in matched_tools("show me 3 moves to consider")
    # "legal moves" must stay legal_moves, NOT best_move
    assert matched_tools("give me the legal moves") == {"legal_moves"}
