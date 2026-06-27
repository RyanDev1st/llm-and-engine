"""Stage 2 — V1_T_audited_plan (pure-chess v5): the audit slice verifies checkable CHESS
boxes by RUNNING the python tool and reading output (never asserting) and splits
determinism on a positional/semantic box. Honest-partial (disabled-skill abort) was
dropped in v5 — a flat catalog has no disabled-skill trigger."""
from collections import Counter

from llm_dataset.v1.renderer.audited_plan import render_audited_plan_row
from llm_dataset.v1.renderer.tags import tool_calls_of
from llm_dataset.v1.validate import Violation, validate_row


def _shape(row):
    asst = "\n".join(m["content"] for m in row["messages"] if m["role"] == "assistant")
    if not row["selected_skills"]:
        return "honest_partial"
    if "judgment call" in asst or "qualitative" in asst:
        return "semantic_split"
    return "full_audit"


def _py_calls(row):
    return sum(1 for m in row["messages"] if m["role"] == "assistant"
               for tc in tool_calls_of(m) if tc["name"] == "python")


def test_all_rows_validate_clean():
    fails = [(_s, validate_row(render_audited_plan_row(_s))[0]) for _s in range(250)
             if validate_row(render_audited_plan_row(_s))]
    assert not fails, fails[:3]


def test_both_shapes_present_in_expected_proportions():
    shapes = Counter(_shape(render_audited_plan_row(s)) for s in range(400))
    assert shapes["full_audit"] > 250           # the dominant lesson
    assert shapes["semantic_split"] > 50        # split-determinism (soft box)
    assert shapes["honest_partial"] == 0        # dropped in v5 (no disabled-skill trigger)


def test_full_audit_runs_one_python_per_checkable_box():
    # find a full-audit row and confirm a python call closed each python-bound box
    row = next(render_audited_plan_row(s) for s in range(50) if _shape(render_audited_plan_row(s)) == "full_audit")
    assert _py_calls(row) == 2 and "plan-audit" in row["selected_skills"]
    # the final verdict numbers are grounded in tool output (narration_grounded passed above)
    assert not validate_row(row)


def test_semantic_box_is_not_tool_audited():
    row = next(render_audited_plan_row(s) for s in range(50) if _shape(render_audited_plan_row(s)) == "semantic_split")
    final = row["messages"][-1]["content"].lower()
    assert _py_calls(row) == 1                              # only the checkable box audited
    assert "judgment call" in final or "qualitative" in final  # soft box stated, not faked


def test_audit_gate_rejects_assert_without_running():
    # A row that CLAIMS a checkable box but never ran python must fail audit_boxes_grounded.
    row = render_audited_plan_row(2)
    # drop the python tool calls (and their tool results) — keep user + non-python assistant turns
    row["messages"] = [m for m in row["messages"]
                       if m["role"] == "user"
                       or (m["role"] == "assistant"
                           and not any(tc["name"] == "python" for tc in tool_calls_of(m)))]
    rules = {v.rule for v in validate_row(row)}
    assert "audit_boxes_grounded" in rules
