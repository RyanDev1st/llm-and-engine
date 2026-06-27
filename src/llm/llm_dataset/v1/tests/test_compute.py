"""V1_R compute-grounding slice: rows validate (incl. the free-text code= arg), the
model calls the python tool, the final is grounded in the script's real stdout, fast
rows stay think-free, and every script executes through the backend sandbox
(train == serve)."""
import re

from backend.sandbox import run_python
from llm_dataset.v1.renderer.compute import render_compute_row
from llm_dataset.v1.renderer.tags import tool_calls_of
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row

_FACT = re.compile(r"[+-]?\d+\.\d{2}")


def _rows(n=80):
    return [render_compute_row(s)
            for s in plan_scenarios({"V1_R_compute_grounding": n}, seed=20260525)]


def _final(row):
    return [m["content"] for m in row["messages"]
            if m["role"] == "assistant" and not tool_calls_of(m)][-1]


def _tool(row):
    return [m["content"] for m in row["messages"] if m["role"] == "tool"][0]


def _code(row):
    for m in row["messages"]:
        if m["role"] == "assistant":
            for tc in tool_calls_of(m):
                if tc["name"] == "python":
                    return str(tc["arguments"].get("code", ""))
    raise AssertionError("no python tool call in row")


def test_rows_validate_clean():
    # exercises the free-text code= parse: a script with spaces/'=' must not trip
    # args_match_schema with spurious extras.
    for row in _rows():
        assert validate_row(row) == [], (row["id"], validate_row(row))


def test_calls_python_and_final_is_grounded():
    for row in _rows():
        assert row["expected_tool_calls"] == ["python"]
        assert "python" in {t["name"] for t in row["tool_manifest"]}
        assert row["selected_skills"] == []          # no domain skill fits a compute ask
        tool_nums = {f.lstrip("+-") for f in _FACT.findall(_tool(row))}
        final_nums = {f.lstrip("+-") for f in _FACT.findall(_final(row))}
        assert final_nums and final_nums <= tool_nums, (row["id"], final_nums, tool_nums)


def test_fast_rows_have_no_think():
    for row in _rows(160):
        if row["reasoning_mode"] == "fast":
            assert all("<think>" not in m["content"]
                       for m in row["messages"] if m["role"] == "assistant"), row["id"]


def test_script_executes_and_matches_stored_result():
    for row in _rows():
        code = _code(row)
        assert run_python(code) == _tool(row)        # deterministic train == serve
        assert _tool(row).startswith("output: "), code
