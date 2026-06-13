from collections import Counter

from llm_dataset.v1.audit import _missing
from llm_dataset.v1.profiles import profile


def row(slice_name="V1_A_skill_index_selection", rules=None, final="Specific answer.", user="What should I do?"):
    return {
        "id": f"row_{slice_name}",
        "slice": slice_name,
        "acceptance_rules": rules or ["final_no_xml"],
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": final},
        ],
        "skills_index": [{"name": "chess-coach", "description": "Analyze chess positions.", "plugin": "chess-official", "source": "official_plugin", "enabled": True}],
        "tool_manifest": [{"name": "load_skill", "description": "Load skill.", "args": {"name": "required"}, "applies_when": "always", "plugin": "chess-official", "source": "official_plugin", "enabled": True}],
        "plugin_context": {"installed": ["chess-official"], "enabled": ["chess-official"]},
        "reject_reason": "audit_fixture: undeclared_tool",
    }


def missing_for(accepted, rejected, audit_profile=None):
    return _missing(
        accepted,
        rejected,
        Counter(r["slice"] for r in accepted),
        Counter(r["slice"] for r in rejected),
        Counter(rule for r in accepted for rule in r["acceptance_rules"]),
        audit_profile or profile("v1.2"),
    )


def test_missing_enforces_universality_slice_floor():
    accepted = [row() for _ in range(59)]
    rejected = [row() for _ in range(800)]
    assert "V1_A_skill_index_selection accepted < 60" in missing_for(accepted, rejected)


def test_missing_enforces_rule_coverage_floor():
    accepted = [row(rules=["engine_grounded"]) for _ in range(199)]
    rejected = [row() for _ in range(800)]
    assert "rule coverage < 200: engine_grounded" in missing_for(accepted, rejected)


def test_missing_rejects_monoculture_reject_reasons():
    accepted = [row() for _ in range(4000)]
    rejected = [row() for _ in range(800)]
    assert "reject reason diversity < 4" in missing_for(accepted, rejected)


def test_missing_rejects_generic_final_concentration():
    accepted = [
        row(final="I read the index, picked the right skill, called only declared tools, and answered without XML.")
        for _ in range(4000)
    ]
    rejected = [row() for _ in range(800)]
    assert "generic final share > 2%" in missing_for(accepted, rejected)


def test_v1_2_missing_enforces_modern_scale():
    accepted = [row() for _ in range(74999)]
    rejected = [row() for _ in range(7500)]
    assert "accepted < 75000" in missing_for(accepted, rejected, profile("v1.2"))


def test_v1_2_missing_enforces_prompt_diversity():
    accepted = [row(user="What should I do?") for _ in range(50000)]
    rejected = [row() for _ in range(7500)]
    assert "normalized user prompt concentration too high" in missing_for(accepted, rejected, profile("v1.2"))


def test_v1_2_profile_scales_chess_slice_tolerance():
    accepted = [row(slice_name="A") for _ in range(2585)]  # 180/5222 * 75000 ≈ 2585
    rejected = [row() for _ in range(7500)]
    assert "A outside target tolerance" not in missing_for(accepted, rejected, profile("v1.2"))


def test_v1_2_missing_enforces_plugin_provenance_diversity():
    accepted = [row() for _ in range(50000)]
    rejected = [row() for _ in range(7500)]
    assert "plugin source diversity < 4" in missing_for(accepted, rejected, profile("v1.2"))


def test_v1_2_missing_enforces_real_human_chat_helper_coverage():
    accepted = [row() for _ in range(50000)]
    rejected = [row() for _ in range(7500)]
    missing = missing_for(accepted, rejected, profile("v1.2"))
    assert "human-chat helper accepted coverage < 200" in missing


def test_v1_2_missing_enforces_multi_skill_composition_coverage():
    accepted = [row() for _ in range(50000)]
    rejected = [row() for _ in range(7500)]
    missing = missing_for(accepted, rejected, profile("v1.2"))
    assert "multi-skill composition accepted coverage < 200" in missing
