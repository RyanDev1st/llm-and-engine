"""The chess-official plugin + its prompt-start hook: injects only the plugin's
RUNTIME STATE (the live board) — NOT any skill body. Skills stay progressive-disclosure
(name+desc in the catalog; the model loads bodies on demand). Plug-and-play per plugin."""
import chess

from backend import plugins
from backend.plugins import chess_official
from backend.game import Game
from backend.inference import build_system_prompt


def test_plugin_bundles_default_tools_and_skills():
    assert chess_official.NAME == "chess-official"
    assert any(t["name"] == "move" for t in chess_official.tools())
    assert any(s.name == "chess-coach" for s in chess_official.skills())


def test_prompt_start_hook_injects_live_board_only():
    g = Game()
    g.move("e4")
    text = plugins.prompt_start({"game": g})
    assert "LIVE BOARD" in text and "last_move=e4" in text and "turn=black" in text
    assert "no need to call board_state" in text
    # progressive disclosure preserved: NO skill body pre-loaded
    assert "ACTIVE SKILL" not in text


def test_hook_silent_when_no_game():
    assert plugins.prompt_start({}) == ""     # no runtime state -> nothing injected


def test_build_system_prompt_injects_board_not_skill_body():
    g = Game()
    sys = build_system_prompt(game=g)
    assert "AVAILABLE TOOLS" in sys           # base manifest still there
    assert "AVAILABLE SKILLS" in sys          # catalog (name+desc) for the model to choose
    assert "LIVE BOARD" in sys and "fen=" in sys
    assert "ACTIVE SKILL" not in sys          # but no hard-coded skill body
