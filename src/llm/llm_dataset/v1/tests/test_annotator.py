import os
import pytest

import chess

from llm_dataset.v1.annotator import DEFAULT_SF, StockfishAnnotator, AnnotatedPosition

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_SF), reason="stockfish binary not available")


class FakeEngine:
    def __init__(self):
        self.calls = []

    def analyse(self, board, limit):
        self.calls.append(board.fen())
        return {"score": FakeScore(), "pv": []}


class FakeScore:
    def white(self):
        return self

    def is_mate(self):
        return False

    def score(self):
        return 0


def test_threats_skips_start_position_null_move_warning():
    annotator = StockfishAnnotator()
    engine = FakeEngine()
    annotator._engine = engine
    annotator.annotate(chess.STARTING_FEN)
    assert len(engine.calls) == 1


def test_annotator_returns_grounded_truth_for_start_position():
    annotator = StockfishAnnotator()
    fen = chess.STARTING_FEN
    annotated = annotator.annotate(fen, depth=12)
    assert isinstance(annotated, AnnotatedPosition)
    assert annotated.best_san in {"e4", "d4", "Nf3", "c4", "g3"}
    assert -60 <= annotated.score_cp <= 60
    assert annotated.depth == 12
