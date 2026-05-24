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
