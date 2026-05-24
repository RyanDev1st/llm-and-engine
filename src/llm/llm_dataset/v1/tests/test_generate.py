from llm_dataset.v1.generate import run


def test_generator_smoke_writes_accepted_and_rejected(tmp_path):
    plan = {"V1_J_no_tool_and_mixed_intent": 3, "V1_K_adversarial_injection": 3}
    ok, bad = run(plan, seed=99, out=tmp_path)
    assert ok >= 5
    assert (tmp_path / "accepted.jsonl").exists()
