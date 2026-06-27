import os

import pytest

from llm_dataset.v1.annotator import DEFAULT_SF, StockfishAnnotator
from llm_dataset.v1.renderer.chess import render_chess_row
from llm_dataset.v1.sampler import plan_scenarios
from llm_dataset.v1.validate import validate_row

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_SF), reason="stockfish binary not available")


def test_renders_valid_grounded_row_for_slice_d():
    plan = {"D": 1}
    scenario = plan_scenarios(plan, seed=11)[0]
    ann = StockfishAnnotator()
    try:                                   # quit or the engine subprocess hangs interpreter exit
        row = render_chess_row(scenario, ann)
    finally:
        ann.quit()
    assert row["slice"] == "D"
    assert row["kind"] == "harness_chess"
    assert "score" in row["messages"][-2]["content"]
    assert validate_row(row) == []
