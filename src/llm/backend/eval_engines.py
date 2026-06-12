"""Selectable eval-bar engine: Stockfish (deep, accurate) or our custom LiquidChess
StaticEvaluator (fast material eval). Only the EVAL BAR switches — the agent's analysis
tools keep using Stockfish for depth. One process-wide choice, toggled from the UI.

Both return white-POV centipawns via eval_white_cp(board) -> (kind, value), matching
state_api.eval_bar's contract: ('mate',(side,n)) or ('cp', white_cp)."""
from __future__ import annotations

import chess

_CHOICE = "stockfish"   # "stockfish" | "custom"; process-wide, set via set_engine()


def available() -> list[str]:
    return ["stockfish", "custom"]


def current() -> str:
    return _CHOICE


def set_engine(name: str) -> str:
    global _CHOICE
    if name in ("stockfish", "custom"):
        _CHOICE = name
    return _CHOICE


def _ensure_engine_on_path() -> None:
    """The custom engine lives at src/chess_engine, but the server runs from src/llm —
    add src/ to sys.path so `chess_engine` imports."""
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[2]   # .../src
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


class CustomEvalAdapter:
    """Wrap LiquidChess StaticEvaluator to the eval_white_cp contract used by the bar."""
    def __init__(self) -> None:
        _ensure_engine_on_path()
        from chess_engine.evaluation.static import StaticEvaluator
        self._ev = StaticEvaluator()

    def eval_white_cp(self, board: chess.Board, depth: int):
        if board.is_checkmate():
            return ("mate", ("black" if board.turn == chess.WHITE else "white", 0))
        cp = self._ev.evaluate_position(board)   # already white-POV centipawns
        return ("cp", int(cp))


_CUSTOM: CustomEvalAdapter | None = None


def bar_engine(stockfish):
    """Return the evaluator the eval bar should use right now: the live Stockfish
    Engine, or a lazily-built custom adapter. Falls back to Stockfish if the custom
    engine can't import."""
    if _CHOICE == "custom":
        global _CUSTOM
        try:
            if _CUSTOM is None:
                _CUSTOM = CustomEvalAdapter()
            return _CUSTOM
        except Exception:
            return stockfish
    return stockfish
