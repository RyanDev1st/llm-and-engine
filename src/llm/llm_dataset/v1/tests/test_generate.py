"""Generator wiring for the v5 pure-chess corpus. The cross-domain tests (universality
renderer, skill-routing, marketplace, human-chat bridge, plugin audit fixtures) moved to
legacy with their renderers when the corpus went pure-chess."""
import os
from collections import Counter

import pytest

from llm_dataset.v1.annotator import DEFAULT_SF
from llm_dataset.v1.generate import DEFAULT_PLAN, plan_for_profile, run
from llm_dataset.v1.jsonl_io import read_rows
from llm_dataset.v1.profiles import profile

_needs_sf = pytest.mark.skipif(not os.path.exists(DEFAULT_SF), reason="stockfish binary not available")


def test_default_plan_is_pure_chess():
    # chess core (A-K) + multiturn + the chess-refocused keystones; no cross-domain slices.
    assert set(DEFAULT_PLAN) == set("ABCDEFGHIJK") | {
        "V1_P_multiturn_followup", "V1_R_compute_grounding",
        "V1_S_compound_plan", "V1_T_audited_plan"}
    # grounded-answer slices are up-weighted: concretely ANSWERING is the product.
    assert DEFAULT_PLAN["E"] >= DEFAULT_PLAN["A"]
    assert DEFAULT_PLAN["F"] >= DEFAULT_PLAN["A"]
    assert DEFAULT_PLAN["G"] >= DEFAULT_PLAN["H"]


def test_tiny_plan_is_small_and_covers_every_slice():
    plan = plan_for_profile(profile("v1.2"), tiny=True)
    assert sum(plan.values()) < 100
    assert set(plan) == set(DEFAULT_PLAN)


@_needs_sf
def test_generator_smoke_writes_accepted_and_rejected(tmp_path):
    ok, bad = run({"E": 3, "F": 3}, seed=99, out=tmp_path)
    assert ok >= 4
    assert bad >= 1
    assert (tmp_path / "accepted.jsonl.gz").exists()   # corpus stored gzipped
    assert (tmp_path / "rejected.jsonl.gz").exists()


@_needs_sf
def test_run_diversifies_repeated_chess_prompts(tmp_path):
    ok, _ = run({"A": 100}, seed=99, out=tmp_path, rejected_target=0)
    prompts = Counter(
        row["messages"][0]["content"] for row in read_rows(tmp_path / "accepted.jsonl")
    )
    assert ok == 100
    assert max(prompts.values()) < 20
