"""Guards for the held-out STRESS suite (bench_suites): it must be BIG enough to claim broad
unseen-domain robustness (not an n=20 smoke test), and every routing gold must resolve to a REAL
entry in the live life-skills catalog — so a typo'd gold name (`recipe_scaler` vs `recipe-scaler`)
can't silently score the model wrong. Pure data checks, no model/GPU."""
from backend import plugins
from llm_training.bench_suites import PC, stress_rows
from llm_training.eval_confusion import gold_action

_SKILL_NAMES = {s["name"] for s in plugins.plugin_skills(PC)}
_TOOL_NAMES = {t["name"] for t in plugins.plugin_tools(PC)}


def test_stress_suite_is_large_enough():
    # n=20 has a wide CI at the slice level; expand to a size that supports a robustness claim.
    assert len(stress_rows()) >= 60


def test_every_gold_resolves_to_a_real_catalog_entry():
    # A routing gold must name a skill/tool that actually exists in the catalog the model is
    # shown; a decline gold must carry NO action (the 'none' class).
    for r in stress_rows():
        verb, name = gold_action(r["messages"])
        if verb == "skill":
            assert name in _SKILL_NAMES, f"gold skill '{name}' not in catalog ({r['slice']})"
        elif verb == "tool":
            assert name in _TOOL_NAMES, f"gold tool '{name}' not in catalog ({r['slice']})"
        else:
            assert verb == "none", f"unexpected gold verb {verb} in {r['slice']}"


def test_every_slice_tag_is_populated():
    # Each declared slice must have rows (so per-slice reporting isn't a phantom 0/0).
    slices = {r["slice"] for r in stress_rows()}
    for expected in ("STRESS_ood_skill_clean", "STRESS_ood_skill_messy", "STRESS_ood_tool_clean",
                     "STRESS_ood_tool_messy", "STRESS_decline"):
        assert expected in slices


def test_rows_are_shaped_like_val_rows():
    # eval_benchmark consumes these with no special-casing, so the row schema must match val.
    for r in stress_rows():
        assert {"slice", "reasoning_mode", "skills_index", "tool_manifest",
                "plugin_context", "messages"} <= set(r)
        assert r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant"
