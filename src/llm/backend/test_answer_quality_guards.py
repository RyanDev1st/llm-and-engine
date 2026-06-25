"""Serve-side answer-quality guards surfaced by the live chat showcase (2026-06-24):
  1. move arg coercion — `<tool>move e4</tool>` (no san=) must RUN, not bounce to a corrective error.
  2. degenerate final guard — a markup-only reply ('<', a dangling tag) is a non-answer (the "play e4"
     break that reached the user as 'Coach: <').
  3. result-echo strip — a raw `name: payload` tool-result line the model PARROTED verbatim into its
     reply is removed (the breathing turn leaked 'breathing_timer: 120s set — about 6 …').
Pure CPU, no model/GPU."""
from backend.inference import _is_markup_fragment, _strip_result_echo
from backend.toolfmt import parse_call
from backend.tools import validate_call


# --- 1. move arg coercion ---
def test_move_bare_san_is_coerced():
    name, args = parse_call("<tool>move e4</tool>")
    assert name == "move" and args.get("san") == "e4"
    assert validate_call("move", args) is None          # no longer a corrective-error case


def test_move_bare_piece_and_castle_and_uci():
    assert parse_call("<tool>move Nf3</tool>")[1]["san"] == "Nf3"
    assert parse_call("<tool>move O-O</tool>")[1]["san"] == "O-O"
    assert parse_call("<tool>move e2e4</tool>")[1]["san"] == "e2e4"   # UCI, move tool accepts it


def test_move_explicit_san_unchanged_and_empty_still_errors():
    assert parse_call("<tool>move san=Nf3</tool>")[1]["san"] == "Nf3"
    name, args = parse_call("<tool>move</tool>")
    assert name == "move" and "san" not in args         # nothing to coerce -> corrective error stands
    assert validate_call("move", args) is not None


def test_coercion_scoped_to_move():
    # a bare token on a NON-move tool is not turned into a san (coercion is move-only)
    _, args = parse_call("<tool>list_pieces e4</tool>")
    assert "san" not in args


def test_coercion_only_clean_single_token():
    # 'move rook f8' (multi-word, the can't-spawn-a-piece shape) must NOT coerce the loose 'f8' —
    # it stays a corrective error, not a wrong pawn move. (Guards test_reset_and_errors' expectation.)
    name, args = parse_call("<tool>move rook f8</tool>")
    assert name == "move" and "san" not in args
    assert validate_call("move", args) is not None


# --- 2. degenerate final guard ---
def test_markup_fragment_detected():
    for bad in ("<", "</", "<tool", "<tool>", "</tool>", ">", "<>", " < "):
        assert _is_markup_fragment(bad), bad


def test_real_answer_not_a_fragment():
    for ok in ("Paris.", "O-O is the move here.", "5 miles is about 8.05 km.", "No.", "e4."):
        assert not _is_markup_fragment(ok), ok


# --- 3. result-echo strip ---
def test_strips_parroted_result_line():
    reply = ("Inhale for 4, hold for 7, exhale for 8 for the set time. "
             "breathing_timer: 120s set — about 6 slow 4-7-8 breath cycle(s). How are you feeling?")
    out = _strip_result_echo(reply, ["breathing_timer: 120s set — about 6 slow 4-7-8 breath cycle(s)."])
    assert "breathing_timer:" not in out
    assert "Inhale for 4" in out and "How are you feeling?" in out


def test_keeps_reply_with_no_echo():
    reply = "5 miles is about 8.05 kilometers."
    assert _strip_result_echo(reply, ["convert: 5 miles = 8.047 kilometers (length)"]) == reply


def test_never_strips_errors_or_skill_bodies():
    reply = "There's no move to review yet — make a move first."
    # an error result must not be matched/removed (it isn't a name: fact line)
    assert _strip_result_echo(reply, ["error: no moves to review"]) == reply
    # a multi-line skill body must never be stripped from prose even if a line coincides
    body = "# chess-coach\nThe current board is shown to you each turn as a LIVE BOARD line."
    assert _strip_result_echo("Develop your pieces and castle early.", [body]) == \
        "Develop your pieces and castle early."
