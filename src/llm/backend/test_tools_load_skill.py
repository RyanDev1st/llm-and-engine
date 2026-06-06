"""Phase 3 Task 10: the serving executor must run load_skill (return the body),
mirroring the trained contract. Before this, load_skill hit the unknown-tool
path (error: invalid_syntax) — train != serve."""
from backend.game import Game
from backend.tools import ToolExecutor


def _executor():
    # load_skill never touches the engine, so None is fine here.
    return ToolExecutor(Game(), None)


def test_load_skill_returns_the_skill_body():
    out = _executor().execute("<tool>load_skill name=chess-coach</tool>")
    assert not out.startswith("error"), out
    assert "chess-coach" in out
    assert len(out) > 50  # a real SKILL.md body, not a stub


def test_load_skill_unknown_skill_errors():
    out = _executor().execute("<tool>load_skill name=no-such-skill-xyz</tool>")
    assert out == "error: unknown_skill"
