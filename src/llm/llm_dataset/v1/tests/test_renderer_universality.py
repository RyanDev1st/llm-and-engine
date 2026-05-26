from llm_dataset.v1.renderer.universality import render_universality_row
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row

GENERIC_FINAL = "I read the index, picked the right skill, called only declared tools, and answered without XML."


def test_renders_valid_row_for_each_universality_slice():
    for slice_name in (
        "V1_A_skill_index_selection", "V1_B_skill_conflict_and_absence",
        "V1_C_dynamic_tool_schema", "V1_D_tool_unavailable_and_readonly",
        "V1_E_board_grounding", "V1_G_multi_tool_budget", "V1_H_error_recovery",
        "V1_I_eval_language", "V1_J_no_tool_and_mixed_intent",
        "V1_K_adversarial_injection", "V1_M_marketplace_navigation",
    ):
        scenario = plan_scenarios({slice_name: 1}, seed=3)[0]
        row = render_universality_row(scenario)
        assert validate_row(row) == [], (slice_name, validate_row(row))


def test_universality_finals_are_slice_specific_not_boilerplate():
    for slice_name in (
        "V1_A_skill_index_selection", "V1_B_skill_conflict_and_absence",
        "V1_C_dynamic_tool_schema", "V1_E_board_grounding",
        "V1_G_multi_tool_budget", "V1_H_error_recovery",
        "V1_L_rejects_and_audit_fixtures",
    ):
        scenario = plan_scenarios({slice_name: 1}, seed=7)[0]
        row = render_universality_row(scenario)
        assert GENERIC_FINAL not in row["messages"][-1]["content"]


def test_universality_prompts_use_human_styles():
    rows = [
        render_universality_row(scenario)
        for scenario in plan_scenarios({"V1_M_marketplace_navigation": 12}, seed=8)
    ]
    prompts = {row["messages"][0]["content"].lower() for row in rows}
    assert any("am i cooked" in prompt or "gimme" in prompt for prompt in prompts)
    assert any("plz" in prompt or "wat" in prompt for prompt in prompts)
    assert len(prompts) >= 10


def test_marketplace_navigation_refuses_disabled_plugin_tools():
    scenario = plan_scenarios({"V1_M_marketplace_navigation": 1}, seed=8)[0]
    row = render_universality_row(scenario)
    final = row["messages"][-1]["content"].lower()
    assert validate_row(row) == []
    assert "market-tactics" in final
    assert "disabled" in final
    assert "chess-coach" in final
    assert not any("market_" in call for call in row["expected_tool_calls"])
