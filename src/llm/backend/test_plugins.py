"""The chess-official plugin + its prompt-start hook: pre-loads the coach skill body
and the live board into the system prompt, so the model skips load_skill/board_state.
Anthropic-style additional-context injection, registered per plugin."""
import chess

from backend import plugins
from backend.plugins import chess_official
from backend.game import Game
from backend.inference import build_system_prompt


def test_plugin_bundles_default_tools_and_skills():
    assert chess_official.NAME == "chess-official"
    assert any(t["name"] == "move" for t in chess_official.tools())
    assert any(s.name == "chess-coach" for s in chess_official.skills())


def test_prompt_start_hook_preloads_coach_and_board():
    g = Game()
    g.move("e4")
    text = plugins.prompt_start({"game": g})
    assert "ACTIVE SKILL" in text and "chess-coach" in text
    assert "do NOT call\nload_skill" in text or "do NOT call load_skill" in text
    assert "LIVE BOARD" in text and "last_move=e4" in text and "turn=black" in text
    assert "do NOT call board_state" in text


def test_hook_skips_board_when_no_game():
    text = plugins.prompt_start({})
    assert "ACTIVE SKILL" in text            # coach body still pre-loaded
    assert "LIVE BOARD" not in text          # no game -> no board line


def test_build_system_prompt_injects_hook_with_live_board():
    g = Game()
    sys = build_system_prompt(game=g)
    assert "AVAILABLE TOOLS" in sys          # base manifest still there
    assert "ACTIVE SKILL" in sys             # + the pre-loaded coach body
    assert "LIVE BOARD" in sys and "fen=" in sys
