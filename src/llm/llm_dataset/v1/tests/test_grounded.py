"""The grounded 'why' composer must name the move's REAL point and never invent a
tactic. Built on hand-made AnnotatedPositions (no engine) whose FEN+move yield a
known move_facts, so a wrong or fabricated reason fails."""
import chess

from llm_dataset.v1.annotator import AnnotatedPosition
from llm_dataset.v1.renderer.grounded import why_best_move

START = chess.STARTING_FEN


def _ann(fen, best, line, cp=50, kind="cp", top=()):
    return AnnotatedPosition(
        fen=fen, depth=12, score_cp=cp, score_kind=kind, best_san=best,
        best_line_sans=tuple(line), threats_san=None, top_moves=tuple(top),
    )


def test_pin_reason_is_grounded_and_specific():
    a = _ann("4k3/3n4/8/8/8/8/8/4KB2 w - - 0 1", "Bb5", ["Bb5"], cp=60)
    out = why_best_move(a, ask_number=False, seed=1)
    assert "knight" in out and any(w in out for w in ("pin", "nails", "freez"))
    for bad in ("fork", "mate", "wins", "grabs", "develop"):
        assert bad not in out, f"pin final leaked '{bad}': {out}"
    assert "<" not in out


def test_free_capture_reason_names_the_piece():
    a = _ann("k7/8/8/3b4/8/8/8/3RK3 w - - 0 1", "Rxd5", ["Rxd5"], cp=300)
    out = why_best_move(a, ask_number=False, seed=2)
    assert "bishop" in out and any(w in out for w in ("win", "pick", "net"))


def test_quiet_move_invents_no_tactic():
    a = _ann("8/8/4k3/8/8/4K3/8/3R4 w - - 0 1", "Rd4", ["Rd4"], cp=20)
    out = why_best_move(a, ask_number=False, seed=3)
    for bad in ("fork", "pin", "mate", "checkmate", "wins", "grabs", "develop", "check"):
        assert bad not in out, f"quiet move leaked tactic '{bad}': {out}"


def test_top_form_does_not_cite_the_deep_line():
    # top=N rows return the best_moves list, not the PV — citing the deep line would be
    # ungrounded SAN. So backups appear; the continuation does not.
    a = _ann(START, "Nf3", ["Nf3", "e5", "Bc4"], cp=20, top=[("Nf3", 20), ("d4", 15), ("c4", 10)])
    out = why_best_move(a, ask_number=False, seed=4, top_form=True)
    assert "d4" in out
    assert "e5" not in out and "Bc4" not in out


def test_number_form_quotes_a_groundable_score():
    a = _ann("k7/8/8/3b4/8/8/8/3RK3 w - - 0 1", "Rxd5", ["Rxd5"], cp=300)
    out = why_best_move(a, ask_number=True, seed=5)
    assert "3.00 pawns" in out
