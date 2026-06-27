"""Cross-domain skill routing: the agent must load the skill whose DESCRIPTION
fits the request — any domain, not just chess-coach. One tool per inference step,
a real multi-line SKILL.md body, the domain tool called after the load, and a
guiding-question final. This is the fix for the loaded-skill bias (was 2 of
~2,737 offered skills ever loaded)."""
import re

from llm_dataset.v1.domains import REAL_DOMAINS, pick_domain, synthetic_domain
from llm_dataset.v1.renderer.skill_routing import render_skill_routing_row
from llm_dataset.v1.validate import validate_row

_TOOL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)", re.DOTALL)


def _loaded(row):
    return [m["content"] for m in row["messages"] if "<skill>" in m["content"]]


def _assistant(row):
    return [m["content"] for m in row["messages"] if m["role"] == "assistant"]


def test_real_domain_row_validates_and_loads_that_skill():
    d = REAL_DOMAINS[0]
    row = render_skill_routing_row(d, seed=1, style="casual", normalize=False)
    assert validate_row(row) == [], validate_row(row)
    assert any(d.skill in c for c in _loaded(row))
    assert all("chess-coach" not in c for c in _loaded(row))
    assert row["selected_skills"] == [d.skill]


def test_skill_body_is_multiline_real_skill_md():
    d = REAL_DOMAINS[0]
    row = render_skill_routing_row(d, seed=2, style="formal", normalize=False)
    idx = next(i for i, m in enumerate(row["messages"])
               if m["role"] == "assistant" and "<skill>" in m["content"])
    body = row["messages"][idx + 1]
    assert body["role"] == "tool"
    assert body["content"].count("\n") >= 3
    assert "when to use" in body["content"].lower()


def test_domain_tool_called_after_skill_load():
    d = REAL_DOMAINS[1]
    row = render_skill_routing_row(d, seed=3, style="casual", normalize=False)
    flat = "".join(_assistant(row))
    assert flat.index(f"<skill>{d.skill}</skill>") < flat.index(d.tool)


def test_one_tool_per_inference_message():
    d = REAL_DOMAINS[2]
    row = render_skill_routing_row(d, seed=4, style="anxious", normalize=False)
    for content in _assistant(row):
        assert len(_TOOL.findall(content)) <= 1, content


def test_final_ends_with_guiding_question():
    d = REAL_DOMAINS[3]
    row = render_skill_routing_row(d, seed=5, style="casual", normalize=False)
    final = row["messages"][-1]["content"]
    assert "<tool>" not in final and "<skill>" not in final
    assert final.rstrip().endswith("?")


def test_lead_in_precedes_each_tool_call():
    d = REAL_DOMAINS[0]
    row = render_skill_routing_row(d, seed=7, style="casual", normalize=False)
    for content in _assistant(row):
        if "<tool>" in content:
            assert not content.startswith("<tool>"), f"missing lead-in: {content}"


def test_normalize_loads_two_skills_across_messages():
    d = REAL_DOMAINS[0]
    row = render_skill_routing_row(d, seed=6, style="slang", normalize=True)
    assert validate_row(row) == [], validate_row(row)
    loaded = _loaded(row)
    assert any("hood-human-chat" in c for c in loaded)
    assert any(d.skill in c for c in loaded)
    assert set(row["selected_skills"]) == {"hood-human-chat", d.skill}
    for content in _assistant(row):
        assert len(_TOOL.findall(content)) <= 1, content


def test_synthetic_domains_are_diverse_and_valid():
    names = {synthetic_domain(s).skill for s in range(1, 300)}
    assert len(names) >= 60, len(names)
    row = render_skill_routing_row(synthetic_domain(42), seed=42, style="casual", normalize=False)
    assert validate_row(row) == [], validate_row(row)


def test_pick_domain_mixes_real_and_synthetic():
    picked = [pick_domain(s).skill for s in range(200)]
    real_names = {d.skill for d in REAL_DOMAINS}
    assert any(n in real_names for n in picked)
    assert any(n not in real_names for n in picked)
