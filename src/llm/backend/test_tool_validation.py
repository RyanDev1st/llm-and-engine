"""Tier-3 serve hardening: name/arg validation in the executor. The validation is
EXECUTOR-ACCURATE, not catalog-display-driven — it must catch genuinely-broken calls
(unknown tool name, missing required arg, out-of-range silent-default enum) WITHOUT
over-rejecting calls the executor already tolerates (depth defaults, board_state field
supersets, best_move clamping). Start-position tools avoid the engine, so this is fast."""
from backend.game import Game
from backend.skills import load_skills
from backend.tools import ToolExecutor, validate_call, _condense_skill_body


def test_load_skill_returns_condensed_body():
    # The served skill body must be condensed: NO yaml frontmatter (it's already in the
    # system-prompt catalog) and meaningfully shorter than the raw SKILL.md — this is both the
    # train==serve shape (training fed a terse body) and the skill-load latency fix.
    full = next(s.content for s in load_skills() if s.name == "chess-coach")
    body = ToolExecutor(Game(), None).execute("<tool>load_skill name=chess-coach</tool>")
    assert not body.lstrip().startswith("---")          # frontmatter stripped
    assert "name: chess-coach" not in body              # the yaml keys are gone
    assert len(body) < len(full)                        # smaller than the raw file
    # the rules the model actually needs survive the condense
    assert "board_state" in body and "best_move" in body and "MUST continue" in body


def test_condense_caps_a_pathological_body():
    huge = "---\nname: x\ndescription: y\n---\n" + ("step line\n" * 600)   # ~6k chars
    out = _condense_skill_body(huge)
    assert not out.startswith("---") and len(out) <= 2900 and out.endswith("…")


def _ex():
    return ToolExecutor(Game(), None)  # engine unused by these tools


# --- catches genuinely-broken calls (model gets a precise, self-correctable signal) ---

def test_unknown_tool_name_is_named_not_generic_syntax():
    out = _ex().execute("<tool>teleport_piece foo=bar</tool>")
    assert out == "error: unknown_tool 'teleport_piece'"   # not the generic invalid_syntax


def test_move_without_san_names_the_missing_arg():
    out = _ex().execute("<tool>move</tool>")
    assert "needs 'san=...'" in out and out.startswith("error: tool 'move'")


def test_bad_enum_value_is_rejected_with_the_allowed_set():
    # list_pieces silently defaults a bad color to side-to-move -> wrong data; validate it.
    out = _ex().execute("<tool>list_pieces color=blue</tool>")
    assert out.startswith("error: arg 'color' for 'list_pieces'") and "white/black/mine" in out
    # random_position silently falls back to a puzzle on a bad kind.
    out2 = _ex().execute("<tool>random_position kind=blitz</tool>")
    assert out2.startswith("error: arg 'kind' for 'random_position'") and "puzzle/scramble/open" in out2


# --- never over-rejects calls the executor already handles (no false positives) ---

def test_valid_and_defaulted_calls_still_pass():
    ex = _ex()
    # eval with NO depth: catalog marks depth "required", executor defaults via clamp -> valid.
    assert ex.execute("<tool>eval depth=12</tool>").startswith("score:")
    # board_state field OUTSIDE the catalog enum (turn) is a real executor field -> valid.
    assert "turn=white" in ex.execute("<tool>board_state fields=turn</tool>")
    # in-range enum + required arg present -> valid.
    assert ex.execute("<tool>list_pieces color=white</tool>").startswith("pieces:")
    assert ex.execute("<tool>move san=e4</tool>").startswith("success:")
    # a skill called as a tool still routes to the corrective skill-verb error, not unknown_tool.
    assert "is a skill, not a tool" in ex.execute("<tool>chess-coach</tool>")


def test_validate_call_unit():
    assert validate_call("eval", {}) is None                      # depth defaulted, not required
    assert validate_call("best_move", {"top": "6"}) is None       # executor clamps; not our reject
    assert validate_call("move", {}) is not None                  # missing required san
    assert validate_call("list_pieces", {"color": "mine"}) is None
    assert validate_call("list_pieces", {"color": "green"}) is not None
    assert validate_call("some_plugin_tool", {"x": "y"}) is None  # unknown tool -> dispatch decides
