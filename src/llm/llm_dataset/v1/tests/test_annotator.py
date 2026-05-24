import os
import pytest

import chess

from llm_dataset.v1.annotator import DEFAULT_SF, StockfishAnnotator, AnnotatedPosition

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_SF), reason="stockfish binary not available")


def test_annotator_returns_grounded_truth_for_start_position():
    annotator = StockfishAnnotator()
    fen = chess.STARTING_FEN
    annotated = annotator.annotate(fen, depth=12)
    assert isinstance(annotated, AnnotatedPosition)
    assert annotated.best_san in {"e4", "d4", "Nf3", "c4", "g3"}
    assert -60 <= annotated.score_cp <= 60
    assert annotated.depth == 12
