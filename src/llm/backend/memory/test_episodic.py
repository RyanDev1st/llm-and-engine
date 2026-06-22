"""Episodic 'how-to-operate' memory: the system LEARNS a tool's correct usage from a turn's
error->fix recovery and RECALLS it for a similar later request — so a frozen model gets a one-shot
call next time instead of repeating the bare-call->error->fix dance. Global (about tools, not the
user), flag-gated (CHESS_EPISODIC), bounded + PII-gated like the profile store. CPU, no model."""
from backend.memory import episodic as ep

PC = {"installed": ["life-skills"], "enabled": ["life-skills"], "marketplace": []}
CHESS = {"installed": ["chess-official"], "enabled": ["chess-official"], "marketplace": []}


def _result(calls, results, reply):
    return {"tool_calls": calls, "tool_results": results, "reply": reply}


# the arg-omission recovery the transcript showed: bare call -> error -> fixed call -> grounded
CORR = _result(
    ["<tool>scale_recipe</tool>", "<tool>scale_recipe from_servings=12 to_servings=30</tool>"],
    ["error: scale_recipe needs numeric from_servings and to_servings",
     "scale_recipe: multiply every ingredient by 2.5x (from 12 to 30 servings)"],
    "Multiply every ingredient by 2.5x.")


def _enable(monkeypatch, tmp_path):
    monkeypatch.setenv("CHESS_EPISODIC", "1")
    monkeypatch.setenv("CHESS_MEMORY_DIR", str(tmp_path))


def test_disabled_by_default(monkeypatch, tmp_path):
    # Default off -> observe/recall are no-ops, so current serve behavior is byte-identical.
    monkeypatch.delenv("CHESS_EPISODIC", raising=False)
    monkeypatch.setenv("CHESS_MEMORY_DIR", str(tmp_path))
    assert ep.observe("scale my recipe 12 to 30", CORR, PC) is False
    assert ep.episodic_block("scale my recipe 12 to 30", PC) == ""


def test_harvest_correction_then_recall_similar(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    assert ep.observe("scale my cookie recipe from 12 up to 30 servings", CORR, PC) is True
    blk = ep.episodic_block("scale my brownie recipe from 6 to 24 servings", PC)
    assert "RECALLED" in blk and "scale_recipe from_servings" in blk   # the working call is surfaced


def test_no_recall_for_a_dissimilar_request(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    ep.observe("scale my cookie recipe from 12 up to 30 servings", CORR, PC)
    assert ep.episodic_block("set a metronome to 120 bpm", PC) == ""


def test_no_recall_when_the_tool_is_not_in_the_live_manifest(monkeypatch, tmp_path):
    # Don't recall a tool the current context can't even call (chess-only serve, recipe lesson).
    _enable(monkeypatch, tmp_path)
    ep.observe("scale my cookie recipe from 12 up to 30 servings", CORR, PC)
    assert ep.episodic_block("scale my recipe from 6 to 24 servings", CHESS) == ""


def test_no_harvest_without_an_error_then_fix(monkeypatch, tmp_path):
    # A clean one-shot turn taught us nothing new -> store nothing (conservative gate).
    _enable(monkeypatch, tmp_path)
    clean = _result(["<tool>scale_recipe from_servings=12 to_servings=30</tool>"],
                    ["scale_recipe: multiply every ingredient by 2.5x (from 12 to 30 servings)"],
                    "Multiply by 2.5x.")
    assert ep.observe("scale my recipe 12 to 30", clean, PC) is False


def test_no_harvest_when_the_turn_never_resolved(monkeypatch, tmp_path):
    # Poison control: a turn that ended in error / produced no answer is NOT a lesson.
    _enable(monkeypatch, tmp_path)
    failed = _result(["<tool>scale_recipe</tool>"], ["error: needs numeric from_servings"], "")
    assert ep.observe("scale my recipe", failed, PC) is False


def test_gate_rejects_pii_in_the_trigger(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    assert ep.observe("email me at bob@x.com about scaling 12 to 30", CORR, PC) is False


def test_store_is_bounded(monkeypatch, tmp_path):
    eps = []
    for i in range(ep._CAP + 10):
        ep.add_episode(eps, f"trigger number {i} alpha beta gamma", f"tool_{i}", f"<tool>tool_{i} x=1</tool>")
    assert len(eps) <= ep._CAP


def test_dedup_refreshes_the_canonical_lesson_per_tool(monkeypatch, tmp_path):
    # One lesson per tool: a second correction for the same tool refreshes it + bumps hits.
    eps = []
    ep.add_episode(eps, "scale 12 to 30", "scale_recipe", "<tool>scale_recipe from_servings=12 to_servings=30</tool>")
    ep.add_episode(eps, "scale 4 to 8", "scale_recipe", "<tool>scale_recipe from_servings=4 to_servings=8</tool>")
    tools = [e["tool"] for e in eps]
    assert tools == ["scale_recipe"] and eps[0]["hits"] == 2          # merged, not duplicated
