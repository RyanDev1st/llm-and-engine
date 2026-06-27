from llm_dataset.v1.sampler import Scenario, plan_scenarios


def test_plan_scenarios_emits_expected_count_and_axes():
    plan = {"A": 30, "V1_C_dynamic_tool_schema": 10}
    scenarios = plan_scenarios(plan, seed=1)
    assert len(scenarios) == 40
    families = {s.name_family for s in scenarios}
    assert "real" in families and "synthetic" in families
    chess_scenarios = [s for s in scenarios if s.slice == "A"]
    assert all(s.position is not None for s in chess_scenarios)
    universality = [s for s in scenarios if s.slice == "V1_C_dynamic_tool_schema"]
    assert any(s.name_family == "synthetic" for s in universality)


def test_plan_scenarios_deterministic_with_seed():
    plan = {"A": 5}
    a = plan_scenarios(plan, seed=42)
    b = plan_scenarios(plan, seed=42)
    assert [s.intent for s in a] == [s.intent for s in b]


def test_scenarios_use_flat_pure_chess_catalog():
    """v5: every row lists the SAME flat chess catalog (coach + specialists + chat,
    and the core + specialist + python tools) with NO plugin gating or cross-domain
    distractors — the model routes by description/context, matching the served manifest."""
    scenarios = plan_scenarios({"E": 40}, seed=5)
    assert {s.prompt_style for s in scenarios} >= {"casual", "slang", "typo"}
    skill_names = {skill["name"] for s in scenarios for skill in s.skills_index}
    assert {"chess-coach", "game-reviewer", "opening-advisor", "tactical-puzzles",
            "hood-human-chat"} <= skill_names
    assert "cooking-helper" not in skill_names and "code-reviewer" not in skill_names
    tool_names = {t["name"] for s in scenarios for t in s.tool_manifest}
    assert {"best_move", "what_if", "name_opening", "accuracy_report", "find_blunders",
            "python"} <= tool_names
    assert all(not skill.get("plugin") for s in scenarios for skill in s.skills_index)
    assert all(s.plugin_context == {} for s in scenarios)
