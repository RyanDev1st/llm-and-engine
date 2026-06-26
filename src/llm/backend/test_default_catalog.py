"""Regression: the DEFAULT served catalog must contain the maintained plugin skills. The live bug was
the FRONTEND clobbering this — PluginsUI pushed bogus plugin names (stockfish-engine/opening-book) to
/api/plugin on every load, and apply_plugin REPLACES installed/enabled, disabling chess-official/
openings/analysis/puzzles. Result: the model lost chess-coach, opening-advisor, game-reviewer, and
tactical-puzzles, leaving only two thin frontend-injected demo stubs -> flaky coaching. This locks the
backend truth (the real catalog is good) so a future regression of the plugin context is caught here."""
from backend.inference import PLUGIN_CONTEXT, serving_skills_index


def test_default_catalog_has_the_maintained_skills():
    names = {s["name"] for s in serving_skills_index(PLUGIN_CONTEXT)}
    assert {"chess-coach", "opening-advisor", "game-reviewer", "tactical-puzzles"} <= names, names
    # the broken frontend demo stubs are NOT part of the maintained backend catalog
    assert "tactical-puzzle-generator" not in names
    assert "chess-opening-advisor" not in names


def test_tactical_puzzles_body_drives_the_coach_loop():
    # the maintained skill (not the thin stub) tells the model to GET a puzzle position + hide the answer
    body = next(s for s in serving_skills_index(PLUGIN_CONTEXT) if s["name"] == "tactical-puzzles")
    # serving index carries description; the body lives in the plugin — assert the plugin body is the rich one
    from backend.plugins import puzzles
    b = puzzles._BODY
    assert "random_position kind=puzzle" in b      # GET a puzzle, don't scan the start board
    # the body withholds the move via the stronger setup contract: no answer appears until best_move.
    assert "do NOT reveal the solution" in b        # don't spoil the solution up front
    assert "call `best_move`" in b                  # ground the reveal/check only when needed
    assert "reveal now" in b                        # but do give it when the solver is stuck
