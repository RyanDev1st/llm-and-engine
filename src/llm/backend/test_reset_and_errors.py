"""Live-bug fixes (2026-06-25, from real serve transcripts):
- "reset board" had no tool, so the model hand-typed a start FEN and botched it -> add new_game.
- `error: invalid_fen` was a dead end -> make it actionable (field spec + start FEN + new_game).
- the move corrective error showed a literal `san=...` placeholder the model COPIED -> show a real example.
"""
import chess

from backend.game import Game
from backend.tools import ToolExecutor, validate_call
from llm_dataset.v1.catalog import official_tools


def _exec(call, game=None):
    return ToolExecutor(game or Game(), None).execute(call)


def test_new_game_tool_is_in_the_catalog():
    names = {t["name"] for t in official_tools()}
    assert "new_game" in names


def test_new_game_resets_the_board():
    g = Game()
    g.move("e4"); g.move("e5")
    assert g.san_stack == ["e4", "e5"]
    out = _exec("<tool>new_game</tool>", g)
    assert out.startswith("success:")
    assert g.board.fen() == chess.STARTING_FEN      # back to start
    assert g.san_stack == []                         # history cleared


def test_invalid_fen_error_is_actionable():
    # the malformed FEN from the live transcript (missing move-number fields)
    out = _exec("<tool>load_fen fen=r1b1kbnr/pppp1ppp/2n5/4p3/4P3/2N1N1Q1/PPPP1PPP/RNB1KBNR w KQkq - 17</tool>")
    assert out.startswith("error: invalid_fen")
    assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" in out   # the start FEN, so the model can recover
    assert "new_game" in out                                                   # points at the right tool to reset


def test_move_corrective_error_shows_a_real_example_not_a_placeholder():
    # "spawn a rook" -> the model called `move rook f8` (no san=) -> corrective error.
    msg = validate_call("move", {})
    assert msg is not None
    assert "san=Nf3" in msg          # a real, copyable move...
    assert "san=..." not in msg      # ...NOT the literal placeholder the model pasted verbatim


def test_move_corrective_fires_on_the_spawn_a_rook_shape():
    # `move rook f8` parses to name=move, args={} (no k=v) -> the corrective, not a crash.
    out = _exec("<tool>move rook f8</tool>")
    assert out.startswith("error: tool 'move' needs 'san'")
