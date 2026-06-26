"""Tier-3 serve hardening: name/arg validation in the executor. The validation is
EXECUTOR-ACCURATE, not catalog-display-driven — it must catch genuinely-broken calls
(unknown tool name, missing required arg, out-of-range silent-default enum) WITHOUT
over-rejecting calls the executor already tolerates (depth defaults, board_state field
supersets, best_move clamping). Start-position tools avoid the engine, so this is fast."""
import chess.engine

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
    assert "board_state" in body and "best_move" in body and "MUST call" in body


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
    # a REAL example move (san=Nf3), not the literal `san=...` placeholder a small model copies
    assert out.startswith("error: tool 'move'") and "san=Nf3" in out and "san=..." not in out


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


def test_arg_keys_are_case_insensitive():
    # LIVE BUG (the demo screenshot): the model emitted `<tool>move SAN=Rd1#</tool>` (uppercase key)
    # and the parser kept 'SAN', so args.get('san') missed and it bounced with "needs 'san'".
    # Every catalog arg key is lowercase, so keys are case-folded; a clean move now runs.
    ex = _ex()
    out = ex.execute("<tool>move SAN=e4</tool>")
    assert out.startswith("success:")                          # not "error: tool 'move' needs 'san'"
    assert "needs 'san'" not in out
    # uppercase keys across other tools resolve too (no corrective bounce)
    assert ex.execute("<tool>list_pieces COLOR=white</tool>").startswith("pieces:")
    assert "turn=" in ex.execute("<tool>board_state FIELDS=basic</tool>")


def test_move_positional_san_with_mate_suffix_is_coerced():
    # LIVE BUG: after best_move returned Qh4#, the model emitted `<tool>move Qh4#</tool>`.
    # The parser intended to coerce bare SAN into san=..., but a word-boundary regex rejected
    # the trailing '#', so the executor bounced with "needs san". Mate/check suffixes must run.
    g = Game()
    for san in ["f3", "e5", "g4"]:
        assert g.move(san).startswith("success:")
    out = ToolExecutor(g, None).execute("<tool>move Qh4#</tool>")
    assert out.startswith("success: Qh4#")
    assert "needs 'san'" not in out


def test_enum_values_are_case_folded():
    # The model also CAPITALIZES enum values (color=White, kind=Puzzle). The allowed sets are
    # lowercase, so a capitalized value would bounce; the executor folds enum values before validate.
    ex = _ex()
    assert ex.execute("<tool>list_pieces color=White</tool>").startswith("pieces:")
    assert ex.execute("<tool>list_pieces color=BLACK</tool>").startswith("pieces:")
    assert "must be one of" not in ex.execute("<tool>list_pieces color=Mine</tool>")
    # a genuinely invalid value still bounces (not silently defaulted to wrong data)
    assert "must be one of" in ex.execute("<tool>list_pieces color=purple</tool>")


def test_board_state_never_returns_empty_on_junk_fields():
    # LIVE BUG (2026-06-25 Kaggle run): the model emitted `fields=<['all']>` (schema-placeholder +
    # list junk). The old parser matched the junk literally, found no field, and returned a bare
    # "board_state:" — leaving the model ungrounded so it looped and played an extra move. The fix:
    # strip the wrapping chars so 'all' still registers, and NEVER return an empty board_state.
    ex = _ex()
    out = ex.execute("<tool>board_state fields=<['all']></tool>")
    assert out.startswith("board_state: ") and out.strip() != "board_state:"
    assert "turn=white" in out and "fen=" in out               # <['all']> -> the full 'all' set
    # other junk wrappings the model produces all degrade to a useful board, not empty
    assert "turn=white" in ex.execute("<tool>board_state fields=[all]</tool>")
    assert "turn=white" in ex.execute("<tool>board_state fields=basic</tool>")
    # a genuinely unrecognized value falls back to basic rather than empty
    unknown = ex.execute("<tool>board_state fields=xyzzy</tool>")
    assert "turn=white" in unknown and "legal_count=" in unknown


# --- error labeling: a non-engine tool fault must NOT be reported as a Stockfish outage ---

class _Boom(ToolExecutor):
    """Fault injection: make dispatch raise so we exercise execute()'s except wrapper on the
    real call path. `eval` raises a genuine EngineError (Stockfish down); any other tool raises
    a plain Exception (a plugin/sandbox bug)."""
    def _dispatch(self, name, args):
        if name == "eval":
            raise chess.engine.EngineError("stockfish died")
        raise ValueError("plugin handler blew up")


def test_engine_error_still_reports_engine_unavailable():
    assert _Boom(Game(), None).execute("<tool>eval depth=12</tool>") == "error: engine_unavailable"


def test_non_engine_tool_fault_is_named_not_engine_unavailable():
    # A plugin/sandbox handler raising a plain exception must NOT be mislabeled as a Stockfish
    # outage — name the failing tool so the model can re-route and the user isn't told the
    # engine is down for a non-chess bug.
    out = _Boom(Game(), None).execute("<tool>convert_units value=5</tool>")
    assert out == "error: tool_failed 'convert_units'"
    assert "engine_unavailable" not in out


def test_validate_call_unit():
    assert validate_call("eval", {}) is None                      # depth defaulted, not required
    assert validate_call("best_move", {"top": "6"}) is None       # executor clamps; not our reject
    assert validate_call("move", {}) is not None                  # missing required san
    assert validate_call("list_pieces", {"color": "mine"}) is None
    assert validate_call("list_pieces", {"color": "green"}) is not None
    assert validate_call("some_plugin_tool", {"x": "y"}) is None  # unknown tool -> dispatch decides
