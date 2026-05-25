from llm_dataset.v1.generate import _audit_rejects, run


def test_generator_smoke_writes_accepted_and_rejected(tmp_path):
    plan = {"V1_J_no_tool_and_mixed_intent": 3, "V1_K_adversarial_injection": 3}
    ok, bad = run(plan, seed=99, out=tmp_path)
    assert ok >= 5
    assert bad >= 1
    assert (tmp_path / "accepted.jsonl").exists()
    assert (tmp_path / "rejected.jsonl").exists()


def test_audit_rejects_have_diverse_reasons():
    rows = [
        {
            "id": f"row_{idx}",
            "slice": "V1_A_skill_index_selection",
            "messages": [
                {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
                {"role": "assistant", "content": "specific final"},
            ],
        }
        for idx in range(16)
    ]
    rejects = _audit_rejects(rows, 16)
    reasons = {row["reject_reason"] for row in rejects}
    assert len(reasons) >= 4
