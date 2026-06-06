"""SFT trains tool use, not tone — final-answer openers must carry no persona."""
from llm_dataset.v1.renderer import tone

BANNED = (
    "no fluff", "cutting to it", "plain take", "straight read", "sure thing",
    "happy to help", "glad you", "direct answer", "here's what", "quick read",
)


def test_openers_have_no_persona_tokens():
    pools = tone.OPENERS_WARM + tone.OPENERS_BLUNT + tone.OPENERS_SOCRATIC
    for opener in pools:
        low = opener.strip().lower()
        assert not any(low.startswith(b) for b in BANNED), f"persona opener: {opener!r}"


def test_pick_still_returns_a_string():
    assert isinstance(tone.pick(1, tone.OPENERS_WARM), str)
