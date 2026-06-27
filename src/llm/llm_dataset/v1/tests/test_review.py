"""Grounded move review (slice F) must MEASURE the move, not rubber-stamp it: an honest
label + centipawn loss from the real before/after evals, praise that names the move's
true point, and a critique that names the better move. The old slice-F final hardcoded
'label=good, delta=+0.05' for every move — these tests lock in that it can't anymore.

Pure functions only (no engine): before/after are hand-made AnnotatedPositions, so a
wrong sign, a mislabel, or an invented tactic fails here."""
import chess

from llm_dataset.v1.annotator import AnnotatedPosition
from llm_dataset.v1.renderer.review import (
    ReviewFacts, compute_review, delta_str, label_for, why_review,
)

START = chess.STARTING_FEN


def _ann(cp, best="Nf3", kind="cp", fen=START):
    return AnnotatedPosition(fen=fen, depth=12, score_cp=cp, score_kind=kind,
                             best_san=best, best_line_sans=(), threats_san=None, top_moves=())


def test_label_thresholds():
    assert label_for(0, True) == "best"
    assert label_for(200, True) == "best"        # is_best wins regardless of loss
    assert label_for(5, False) == "good"
    assert label_for(40, False) == "good"
    assert label_for(80, False) == "inaccuracy"
    assert label_for(150, False) == "mistake"
    assert label_for(300, False) == "blunder"


def test_best_move_is_zero_loss_and_labelled_best():
    rf = compute_review(_ann(30, best="Nf3"), _ann(28), "Nf3")
    assert rf.label == "best"
    assert rf.loss_cp <= 10


def test_black_to_move_sign_is_handled():
    # Black to move, eval -30 (white POV) = black slightly better. Playing the best keeps it.
    before = _ann(-30, best="Nf6", fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    rf = compute_review(before, _ann(-28), "Nf6")
    assert rf.loss_cp <= 10 and rf.label == "best"


def test_blunder_names_the_better_move_and_its_point():
    before = _ann(30, best="Nc3")                 # best develops a knight
    rf = compute_review(before, _ann(-260), "Nf3")  # played a different move; eval cratered
    assert rf.label == "blunder"
    assert rf.loss_cp >= 250
    out = why_review(rf, before, seed=1, ask_number=False)
    assert "Nc3" in out and "Nf3" in out
    assert any(w in out for w in ("blunder", "isn't best"))
    assert "<" not in out


def test_good_move_praise_is_grounded_not_a_rubber_stamp():
    before = _ann(30, best="Nf3")
    rf = compute_review(before, _ann(25), "Nf3")
    out = why_review(rf, before, seed=2, ask_number=False)
    assert "Nf3" in out
    assert any(p in out for p in ("best move", "exactly right", "Nothing better"))
    for bad in ("blunder", "mistake", "inaccuracy"):
        assert bad not in out


def test_allows_mate_is_flagged_and_phrased_honestly():
    before = _ann(30, best="e4")
    rf = compute_review(before, _ann(-1, kind="mate"), "Nf3")  # white now faces mate
    assert rf.note == "allows mate" and rf.label == "blunder"
    out = why_review(rf, before, seed=3, ask_number=False)
    assert "walks into mate" in out and "e4" in out


def test_missed_mate_is_flagged():
    before = _ann(3, best="Qh5", kind="mate")     # white had a forced mate
    rf = compute_review(before, _ann(500, kind="cp"), "Nf3")  # played a non-mating (still winning) move
    assert rf.note == "missed a forced mate" and rf.label == "blunder"
    out = why_review(rf, before, seed=4, ask_number=False)
    assert "forced mate" in out and "Qh5" in out


def test_already_lost_to_mate_is_not_blamed_as_walking_in():
    before = _ann(-2, best="Kf1", kind="mate")    # already getting mated before the move
    rf = compute_review(before, _ann(-1, kind="mate"), "Kf1")
    assert rf.note == ""                           # not a NEW mate we walked into


def test_delta_str_matches_tool_result_format():
    assert delta_str(ReviewFacts("Nf3", "Nf3", 0, "best", 30, 30, "")) == "+0.00 pawns"
    assert delta_str(ReviewFacts("Nf3", "e4", 150, "mistake", 30, -120, "")) == "-1.50 pawns"
    assert delta_str(ReviewFacts("Nf3", "e4", 9999, "blunder", 30, -9000, "allows mate")) == "allows mate"


def test_ask_number_cost_matches_delta_str():
    # When the user asks for a number, the cost the final cites must equal (sign-stripped)
    # the delta in the tool result — or narration grounding would reject it.
    before = _ann(30, best="Nc3")
    rf = compute_review(before, _ann(-120), "Nf3")  # loss 150 -> -1.50
    out = why_review(rf, before, seed=5, ask_number=True)
    assert "1.50" in out
    assert "1.50" in delta_str(rf)
