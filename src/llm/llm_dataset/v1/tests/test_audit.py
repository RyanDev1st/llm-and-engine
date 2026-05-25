from collections import Counter

from llm_dataset.v1.audit import _missing


def row(slice_name="V1_A_skill_index_selection", rules=None, final="Specific answer."):
    return {
        "id": f"row_{slice_name}",
        "slice": slice_name,
        "acceptance_rules": rules or ["final_no_xml"],
        "messages": [{"role": "assistant", "content": final}],
        "reject_reason": "audit_fixture: undeclared_tool",
    }


def test_missing_enforces_universality_slice_floor():
    accepted = [row() for _ in range(59)]
    rejected = [row() for _ in range(800)]
    missing = _missing(
        accepted,
        rejected,
        Counter(r["slice"] for r in accepted),
        Counter(r["slice"] for r in rejected),
        Counter(rule for r in accepted for rule in r["acceptance_rules"]),
    )
    assert "V1_A_skill_index_selection accepted < 60" in missing


def test_missing_enforces_rule_coverage_floor():
    accepted = [row(rules=["engine_grounded"]) for _ in range(199)]
    rejected = [row() for _ in range(800)]
    missing = _missing(
        accepted,
        rejected,
        Counter(r["slice"] for r in accepted),
        Counter(r["slice"] for r in rejected),
        Counter(rule for r in accepted for rule in r["acceptance_rules"]),
    )
    assert "rule coverage < 200: engine_grounded" in missing


def test_missing_rejects_monoculture_reject_reasons():
    accepted = [row() for _ in range(4000)]
    rejected = [row() for _ in range(800)]
    missing = _missing(
        accepted,
        rejected,
        Counter(r["slice"] for r in accepted),
        Counter(r["slice"] for r in rejected),
        Counter(rule for r in accepted for rule in r["acceptance_rules"]),
    )
    assert "reject reason diversity < 4" in missing


def test_missing_rejects_generic_final_concentration():
    accepted = [
        row(final="I read the index, picked the right skill, called only declared tools, and answered without XML.")
        for _ in range(4000)
    ]
    rejected = [row() for _ in range(800)]
    missing = _missing(
        accepted,
        rejected,
        Counter(r["slice"] for r in accepted),
        Counter(r["slice"] for r in rejected),
        Counter(rule for r in accepted for rule in r["acceptance_rules"]),
    )
    assert "generic final share > 2%" in missing
