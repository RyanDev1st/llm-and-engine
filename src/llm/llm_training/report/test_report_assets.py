"""CPU tests for the report/PPT assets — the pure logic (no model, no GPU). Covers: the hand-written
chat suites' shape, the timing wrapper's token/second accounting, the showcase markdown (timing +
tok/s present), the cross-model data merge, and the confusion-matrix caption math. PNG rendering is
smoke-tested only when matplotlib is importable (skipped otherwise) so the suite stays GPU-free."""
import pytest

from llm_training.eval_confusion import confusion_caption
from llm_training.report import chart_data as D
from llm_training.report import chat_showcase as CS
from llm_training.report import chat_suites


def test_chat_suites_well_formed():
    chat_suites.validate(chat_suites.PLAIN_CHATS)      # raises on any malformed scenario
    chat_suites.validate(chat_suites.WEB_CHATS)
    assert all(sc.get("fen") for sc in chat_suites.WEB_CHATS), "every WEB scenario needs a board"
    assert not any(sc.get("fen") for sc in chat_suites.PLAIN_CHATS), "PLAIN is board-free"


class _FakeModel:
    """Counts 1 token per word; each generate 'takes' no real time (we add it by hand in the test)."""
    def count_tokens(self, text):
        return len((text or "").split())

    def context_limit(self):
        return 8192

    def generate(self, messages, max_new_tokens, stop):
        return "one two three"


def test_timed_accumulates_and_resets():
    m = CS._Timed(_FakeModel())
    out = m.generate([], 16, [])
    assert out == "one two three"
    assert m.tokens == 3 and m.seconds >= 0.0
    m.generate([], 16, [])
    assert m.tokens == 6                                 # accrues across calls within a turn
    m.reset()
    assert m.tokens == 0 and m.seconds == 0.0


def test_showcase_markdown_has_timing_and_tok_s():
    turns = [{"section": "plain", "scenario": "Vague", "fen": "", "prompt": "yo help me",
              "mode": "auto", "reply": "Sure — here's a plan.", "steps": ["skill `<skill>chess-coach</skill>` -> `ok`"],
              "secs": 3.2, "gen_tokens": 64, "tok_s": 20.0}]
    md = CS.render(turns, turns, "adapter")
    assert "tok/s" in md and "20 tok/s" in md
    assert "| time (s) | tokens | tok/s |" in md          # the per-section timing table
    assert "Section 1 — bare harness" in md and "Section 2 — chess-web sandbox" in md
    assert "yo help me" in md


def test_merge_measured_overlays_without_clobbering_seed():
    measured = {"e4b-q5": {"verb": 0.95, "completed": 0.88, "tok_s": 61},
                "e4b-nf4": {"verb": None, "completed": 0.9},    # None must NOT clobber the 0.964 seed
                "unknown-key": {"verb": 0.5}}                   # ignored
    out = D.merge_measured(D.MODELS, measured)
    by = {m["key"]: m for m in out}
    assert by["e4b-q5"]["verb"] == 0.95 and by["e4b-q5"]["tok_s"] == 61
    assert by["e4b-nf4"]["verb"] == 0.964                      # seed preserved past a None
    assert by["e4b-nf4"]["completed"] == 0.9
    assert "unknown-key" not in by
    assert D.MODELS[2].get("completed") is None                # original list untouched (pure)


def test_model_table_md_marks_missing():
    table = D.model_table_md(D.MODELS)
    assert "routing verb" in table and "tok/s" in table
    assert "—" in table                                        # E2B has no seeded numbers yet


def test_confusion_caption_math_and_exact_name():
    # 3x3 with 90 correct on the diagonal out of 100 -> 90% verb accuracy.
    cm = {"skill": {"skill": 30, "tool": 3, "none": 0},
          "tool": {"skill": 4, "tool": 35, "none": 1},
          "none": {"skill": 1, "tool": 1, "none": 25}}
    cap = confusion_caption(cm, name_hits=50, name_tot=68)
    assert "Verb accuracy 90.0%" in cap                        # (30+35+25)/100
    assert "n=100 val rows" in cap
    assert "50/68" in cap and "skill" in cap and "tool" in cap
    # without the name tally, no exact-name line
    assert "NAME match" not in confusion_caption(cm)


def test_measured_sidecar_merges_across_evals(tmp_path):
    from llm_training.report import measured
    # three different evals, three writes, same model KEY -> one merged file (no clobber)
    measured.update(tmp_path, "e4b-q5", verb=0.94)            # eval_confusion
    measured.update(tmp_path, "e4b-q5", completed=0.88, grounded=0.91)  # eval_completion
    measured.update(tmp_path, "e4b-q5", tok_s=58.0)           # chat_showcase
    measured.update(tmp_path, "e4b-q5", verb=None)            # a None never erases the prior value
    got = measured.collect(tmp_path)
    assert got["e4b-q5"] == {"verb": 0.94, "completed": 0.88, "grounded": 0.91, "tok_s": 58.0}
    # collected sidecars merge onto the seeds for the chart
    models = D.merge_measured(D.MODELS, got)
    q5 = next(m for m in models if m["key"] == "e4b-q5")
    assert q5["verb"] == 0.94 and q5["tok_s"] == 58.0


def test_png_smoke_when_matplotlib_present(tmp_path):
    pytest.importorskip("matplotlib")
    from llm_training.report import ppt_charts
    cm = {"skill": {"skill": 30, "tool": 3, "none": 0}, "tool": {"skill": 4, "tool": 35, "none": 1},
          "none": {"skill": 1, "tool": 1, "none": 25}}
    p1 = ppt_charts.confusion_matrix(cm, ["skill", "tool", "none"], tmp_path / "cm.png",
                                     confusion_caption(cm, 50, 68))
    p2 = ppt_charts.model_lines(D.MODELS, tmp_path / "models.png")
    turns = [{"prompt": "hows my position", "reply": "Even — develop your knight.", "secs": 2.1,
              "gen_tokens": 40, "tok_s": 19.0}]
    p3 = ppt_charts.chat_card("Section 1", turns, tmp_path / "card.png", "real harness")
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 1000          # a real PNG, not an empty file
