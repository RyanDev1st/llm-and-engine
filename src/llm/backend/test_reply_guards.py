"""Reply-shape guards: strip a leading skill/tool ANNOUNCE sentence the small model
sometimes prepends to a real answer ("Loading the chess-coach skill. <answer>"). Training
never puts skill/tool narration in the final reply (finals.py), so we strip it at serve.
Conservative: requires an explicit skill/tool word AND real content after it, so coaching
prose ("Use your rook...") and whole-reply lead-ins are never mangled."""
from backend.inference import _strip_announce_leadin, _is_leadin_only, REPLY_TOKENS


def test_strips_skill_announce_when_answer_follows():
    out = _strip_announce_leadin("Loading the chess-coach skill. The best move is e4.")
    assert out == "The best move is e4."


def test_strips_various_announce_forms():
    cases = {
        "I loaded the chess-coach skill. White is up a pawn.": "White is up a pawn.",
        "Let me load the opening-advisor skill. The Sicilian fights for the center.":
            "The Sicilian fights for the center.",
        "Using the load_skill tool. Here is the plan: develop quickly.":
            "Here is the plan: develop quickly.",
        "I'll use the analysis skill. Your accuracy was 88%.": "Your accuracy was 88%.",
    }
    for src, want in cases.items():
        assert _strip_announce_leadin(src) == want


def test_does_not_touch_coaching_prose():
    # no skill/tool word -> never stripped, even though it starts with an action verb
    for r in ("Use your rook to control the open file.",
              "Loading up pressure on the king is the plan.",
              "Castle early, then push in the center."):
        assert _strip_announce_leadin(r) == r


def test_whole_reply_leadin_left_for_fallback():
    # nothing after the announce -> strip is a no-op; _is_leadin_only routes it to the
    # tool-result fallback narration upstream instead.
    r = "Loading the chess-coach skill."
    assert _strip_announce_leadin(r) == r
    assert _is_leadin_only(r) is True


def test_reply_budget_is_generous_enough_to_finish():
    # regression: the final reply used to be capped at 96 tokens (truncated mid-sentence).
    assert REPLY_TOKENS >= 256
