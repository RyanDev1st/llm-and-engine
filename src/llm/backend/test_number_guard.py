"""Number-consistency guard: when the model fabricates an eval number (coarse-quant
tell), replace it with the real tool value. Conservative — single unmatched number vs a
single eval source; never touches legit numbers (best_move scores, move SANs) or guesses
when ambiguous."""
from backend.inference import _correct_eval_number, _correct_move_names


def test_replaces_fabricated_eval_with_real():
    # the live Q4_0 bug: tool +0.37, model wrote -0.18
    out = _correct_eval_number(
        "The position is slightly better for white (-0.18).",
        ["score: +0.37 pawns from white POV, depth=18"])
    assert "+0.37" in out and "-0.18" not in out


def test_correct_number_left_unchanged():
    r = "White is up +0.37 pawns."
    assert _correct_eval_number(r, ["score: +0.37 pawns from white POV, depth=18"]) == r


def test_unsigned_match_left_unchanged():
    r = "Roughly equal at 0.00."
    assert _correct_eval_number(r, ["score: 0.00 pawns from white POV, depth=18"]) == r


def test_best_move_scores_are_not_flagged():
    # reply legitimately quotes best_move per-move scores; none should be "corrected"
    r = "Top moves: e4 (+0.45), d4 (+0.36), c4 (+0.19). Overall +0.37."
    res = ["score: +0.37 pawns from white POV, depth=18",
           "best_moves: 1. e4 (+0.45); 2. d4 (+0.36); 3. c4 (+0.19)"]
    assert _correct_eval_number(r, res) == r          # every number is a real tool number


def test_no_eval_result_is_noop():
    r = "I'd suggest e4 (eval was around -0.18 maybe)."
    # no score: result -> nothing to correct toward -> leave it
    assert _correct_eval_number(r, ["best: e4, score: +0.45"]) == r


def test_two_evals_is_ambiguous_noop():
    r = "The eval is -0.99 now."
    res = ["score: +0.37 pawns from white POV, depth=18",
           "score: -0.12 pawns from white POV, depth=18"]
    assert _correct_eval_number(r, res) == r          # >1 eval source -> don't guess


def test_multiple_fabricated_numbers_noop():
    # two unmatched numbers -> too ambiguous to know which to fix -> leave both
    r = "Around -0.18, maybe -0.50."
    assert _correct_eval_number(r, ["score: +0.37 pawns from white POV, depth=18"]) == r


def test_move_guard_appends_real_moves_when_fabricated():
    # live bug: tool returned the line e4 c6 d4, model said "Nf3, e4, Nc3"
    out = _correct_move_names("The top three moves: Nf3, e4, and Nc3.",
                              ["best_line: e4 c6 d4, score: +0.49 pawns from white POV"])
    assert out.endswith("(Engine's actual moves: e4, c6, d4.)")


def test_move_guard_silent_when_moves_are_real():
    r = "The line is e4 c6 d4."
    assert _correct_move_names(r, ["best_line: e4 c6 d4, score: +0.49"]) == r
    r2 = "Top moves: e4, d4, Nf3."
    assert _correct_move_names(r2, ["best_moves: 1. e4 (+0.45); 2. d4 (+0.36); 3. Nf3 (+0.31)"]) == r2


def test_move_guard_noop_without_best_move_result():
    r = "Your best move is Nf3."
    assert _correct_move_names(r, ["score: +0.20 pawns from white POV"]) == r   # no best_* result
    assert _correct_move_names(r, []) == r


def test_move_numbering_not_mistaken_for_eval():
    # "1." "2." have no fractional digits -> not eval-like -> untouched; the wrong eval is fixed
    r = "Options: 1. e4 2. d4. The position is (-0.18)."
    out = _correct_eval_number(r, ["score: +0.37 pawns from white POV, depth=18"])
    assert "1. e4 2. d4" in out and "+0.37" in out and "-0.18" not in out
