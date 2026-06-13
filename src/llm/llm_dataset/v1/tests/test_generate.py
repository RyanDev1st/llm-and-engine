import re
from collections import Counter

from llm_dataset.v1.generate import _audit_rejects, plan_for_profile, run
from llm_dataset.v1.profiles import profile
from llm_dataset.v1.renderer.universality import render_universality_row
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row


def test_v1_2_tiny_plan_is_fast_and_uses_new_slice():
    plan = plan_for_profile(profile("v1.2"), tiny=True)
    assert sum(plan.values()) < 100
    assert plan["V1_M_marketplace_navigation"] > 0
    assert plan["V1_N_human_chat_skill_bridge"] > 0

def test_generator_smoke_writes_accepted_and_rejected(tmp_path):
    plan = {"V1_J_no_tool_and_mixed_intent": 3, "V1_K_adversarial_injection": 3}
    ok, bad = run(plan, seed=99, out=tmp_path)
    assert ok >= 5
    assert bad >= 1
    assert (tmp_path / "accepted.jsonl.gz").exists()  # corpus stored gzipped
    assert (tmp_path / "rejected.jsonl.gz").exists()


def test_audit_rejects_have_diverse_reasons():
    rows = [
        render_universality_row(scenario)
        for scenario in plan_scenarios({"V1_M_marketplace_navigation": 16}, seed=12)
    ]
    rejects = _audit_rejects(rows, 16)
    reasons = {row["reject_reason"] for row in rejects}
    assert len(reasons) >= 8
    assert "audit_fixture: disabled_plugin_tool" in reasons
    assert "audit_fixture: false_install_claim" in reasons


def test_audit_rejects_include_skill_generalization_antipatterns():
    rows = [
        render_universality_row(scenario)
        for scenario in plan_scenarios({"V1_N_human_chat_skill_bridge": 16}, seed=12)
    ]
    rejects = _audit_rejects(rows, 16)
    reasons = {row["reject_reason"] for row in rejects}

    assert "audit_fixture: skipped_helper_skill" in reasons
    assert "audit_fixture: helper_tool_before_skill" in reasons
    assert "audit_fixture: irrelevant_skill_selected" in reasons


def test_audit_rejects_all_fail_validation():
    rows = [
        render_universality_row(scenario)
        for scenario in plan_scenarios({"V1_M_marketplace_navigation": 16}, seed=12)
    ]
    rejects = _audit_rejects(rows, 16)
    passed = [row["reject_reason"] for row in rejects if not validate_row(row)]
    assert passed == []


def test_human_chat_skill_bridge_loads_helper_before_chess_skill():
    scenario = plan_scenarios({"V1_N_human_chat_skill_bridge": 1}, seed=12)[0]
    row = render_universality_row(scenario)
    # Skills load via <skill>NAME</skill>; tools via <tool>. Extract the ordered
    # action stream (a short lead-in may precede each), skipping by startswith.
    actions = [
        m
        for msg in row["messages"] if msg["role"] == "assistant"
        for m in re.findall(r"<skill>[^<]*</skill>|<tool>[^<]*</tool>", msg["content"])
    ]
    helper_tool = next(t for t in row["tool_manifest"] if t["name"] == "normalize_human_chat")

    assert any(skill["name"] == "hood-human-chat" for skill in row["skills_index"])
    assert helper_tool["plugin"] == "user-skills"
    assert actions[:3] == [
        "<skill>hood-human-chat</skill>",
        "<tool>normalize_human_chat text=messy_user_chat</tool>",
        "<skill>chess-coach</skill>",
    ]
    assert row["selected_skills"] == ["hood-human-chat", "chess-coach"]
    assert "human-chat helper accepted coverage" in row["acceptance_rules"]
    assert "multi-skill composition accepted coverage" in row["acceptance_rules"]
    assert validate_row(row) == []


def test_human_chat_skill_bridge_uses_style_prompt():
    rows = [
        render_universality_row(scenario)
        for scenario in plan_scenarios({"V1_N_human_chat_skill_bridge": 6}, seed=12)
    ]
    prompts = {row["messages"][0]["content"] for row in rows}

    assert "Use helper skill if this wording is unclear, then route the chess intent." not in prompts
    assert len(prompts) >= 4


    p = profile("v1.2")
    assert p.accepted_target == 75_000
    assert 5_000 <= p.rejected_target <= 10_000
    assert p.gold_dir.as_posix().endswith("data/sft/v1_2")
    assert p.train_path.as_posix().endswith("data/sft/v1_2_train.jsonl")
    assert p.val_path.as_posix().endswith("data/sft/v1_2_val.jsonl")


def test_run_diversifies_repeated_chess_prompts(tmp_path):
    from llm_dataset.v1.jsonl_io import read_rows
    ok, _ = run({"A": 100}, seed=99, out=tmp_path, rejected_target=0)
    prompts = Counter(
        row["messages"][0]["content"]
        for row in read_rows(tmp_path / "accepted.jsonl")
    )
    assert ok == 100
    assert max(prompts.values()) < 20
