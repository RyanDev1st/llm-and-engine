"""Wiring + audit metric for cross-domain skill routing. Stockfish-free: the
routing slice needs no board, so we exercise the plan, the scenario shape, and
the loaded-skill-diversity metric without spinning the engine."""
from llm_dataset.v1.audit import _loaded_skill_diversity
from llm_dataset.v1.domains import pick_domain
from llm_dataset.v1.generate import DEFAULT_PLAN, ROUTING_SLICE, plan_for_profile
from llm_dataset.v1.profiles import V1_2
from llm_dataset.v1.renderer.skill_routing import render_skill_routing_row
from llm_dataset.v1.sampler import plan_scenarios


def test_routing_slice_in_default_plan():
    assert ROUTING_SLICE in DEFAULT_PLAN


def test_plan_meets_accepted_target():
    # Per-slice rounding once landed 10 short of 50k; the plan must clear target.
    plan = plan_for_profile(V1_2)
    assert sum(plan.values()) >= V1_2.accepted_target
    assert all(v >= 60 for v in plan.values())


def test_routing_scenarios_need_no_board():
    scenarios = plan_scenarios({ROUTING_SLICE: 5}, seed=11)
    assert len(scenarios) == 5
    assert all(s.position is None for s in scenarios)
    assert all(s.slice == ROUTING_SLICE for s in scenarios)


def test_dispatch_path_builds_diverse_rows():
    # Mirror generate.run's dispatch for the routing slice across many seeds.
    rows = [
        render_skill_routing_row(pick_domain(s), s, "casual", normalize=s % 4 == 0)
        for s in range(1, 120)
    ]
    assert _loaded_skill_diversity(rows) >= 20
    # chess-coach is a distractor in the index but is never the loaded target.
    targets = {r["selected_skills"][-1] for r in rows}
    assert "chess-coach" not in targets


def test_loaded_skill_diversity_counts_distinct_loads():
    rows = [
        {"messages": [{"role": "assistant", "content": "<skill>math-tutor</skill>"}]},
        {"messages": [{"role": "assistant", "content": "<skill>code-reviewer</skill>"}]},
        {"messages": [{"role": "assistant", "content": "<skill>math-tutor</skill>"}]},
        {"messages": [{"role": "tool", "content": "<skill>ignored</skill>"}]},
    ]
    assert _loaded_skill_diversity(rows) == 2
