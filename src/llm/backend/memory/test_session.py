"""Session fact cache: carries this-session analysis facts across turns, FEN-keyed, and the
freshness guard drops them the moment the live board diverges (no stale facts)."""
from backend.memory import session as S

FEN_A = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
FEN_B = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


def test_caches_analysis_facts_and_reuses_when_fen_matches():
    sess = {}
    S.update(sess, FEN_A, ["score: +0.30 pawns from white POV, depth=18",
                           "best: e4, score: +0.30", "board_state: turn=black"])
    note = S.render(sess, FEN_A)               # same position -> facts injected
    assert "ESTABLISHED THIS SESSION" in note
    assert "+0.30" in note and "best: e4" in note
    assert "board_state" not in note           # cheap/re-derivable facts are not cached


def test_caches_position_setup_fact_for_current_fen():
    sess = {}
    S.update_setup(sess, FEN_A, ["position: puzzle set (mate in 1 back-rank). fen=" + FEN_A])
    note = S.render(sess, FEN_A)
    assert "ESTABLISHED THIS SESSION" in note
    assert "back-rank" in note and FEN_A in note


def test_freshness_guard_drops_facts_when_board_moved():
    sess = {}
    S.update(sess, FEN_A, ["score: +0.30 pawns from white POV, depth=18"])
    assert S.render(sess, FEN_B) == ""         # board moved on -> no stale facts


def test_new_position_replaces_old_cache():
    sess = {}
    S.update(sess, FEN_A, ["score: +0.30 pawns from white POV, depth=18"])
    S.update(sess, FEN_B, ["score: -0.10 pawns from white POV, depth=18"])
    assert sess["fen"] == FEN_B
    note = S.render(sess, FEN_B)
    assert "-0.10" in note and "+0.30" not in note   # only the current position's facts


def test_merge_same_position_and_cap():
    sess = {}
    for i in range(8):
        S.update(sess, FEN_A, [f"score: +0.0{i} pawns from white POV, depth=18"])
    assert len(sess["facts"]) <= 5             # bounded
    assert "board_state" not in S.render(sess, FEN_A)


def test_no_facts_is_noop():
    sess = {}
    S.update(sess, FEN_A, ["board_state: turn=black", "error: engine_unavailable"])
    assert S.render(sess, FEN_A) == ""         # nothing groundable cached


def test_setup_update_ignores_non_setup_results_after_board_change():
    sess = {}
    S.update_setup(sess, FEN_B, ["best: e5, score: +0.30", "board_state: turn=black"])
    assert S.render(sess, FEN_B) == ""         # avoids stale eval/best facts under a new FEN
