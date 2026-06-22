"""Play-vs-engine opponent — the team's neural value-net selector plays the side to move, so the
website opponent is a REAL engine (~1400 Elo, src/chess_engine) instead of random moves.

Stateless: given the current FEN, return one legal move. Lazily loads the net once (torch + the
verified checkpoint src/chess_engine/weights/nee_latest.pt); degrades gracefully — neural → a random
legal move — so a missing torch/checkpoint never stalls the board. depth 3 alpha-beta keeps web moves
to a few seconds on CPU (CHESS_ENGINE_DEPTH overrides; depth 4+ is much slower)."""
from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from pathlib import Path

import chess

_CKPT = Path(__file__).resolve().parents[1] / "chess_engine" / "weights" / "nee_latest.pt"
_DEPTH = max(1, int(os.environ.get("CHESS_ENGINE_DEPTH", "3")))


def _ensure_engine_on_path() -> None:
    """The engine lives at src/chess_engine but the server runs from src/llm — put src/ on the
    path so `import chess_engine` resolves (mirrors eval_engines._ensure_engine_on_path)."""
    src = Path(__file__).resolve().parents[2]      # .../src
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@lru_cache(maxsize=1)
def _selector():
    """Load the NeuralMoveSelector once. Raises if torch/checkpoint are unavailable — callers
    treat that as 'fall back to a random move'. Cached so the net loads on first use only."""
    _ensure_engine_on_path()
    from chess_engine.battle.selector import NeuralMoveSelector
    return NeuralMoveSelector(str(_CKPT), search_depth=_DEPTH)


def available() -> bool:
    """True if the neural engine can load (torch + checkpoint present)."""
    try:
        _selector()
        return True
    except Exception:
        return False


def choose(fen: str) -> dict:
    """Pick the engine's move for the position. Returns {ok, uci, source} where source is
    'neural' (the trained net) or 'random' (safety net). ok=False for a bad/finished position."""
    try:
        board = chess.Board(str(fen).strip())
    except (ValueError, AttributeError):
        return {"ok": False, "error": "bad_fen"}
    if not board.is_valid():
        return {"ok": False, "error": "bad_fen"}
    if board.is_game_over() or not any(board.legal_moves):
        return {"ok": False, "error": "game_over"}
    try:
        mv = _selector().choose_move(board)
        if mv in board.legal_moves:
            return {"ok": True, "uci": mv.uci(), "source": "neural"}
    except Exception:
        pass                                       # torch/ckpt missing or a selector hiccup
    mv = random.choice(list(board.legal_moves))    # safety net so the board never stalls
    return {"ok": True, "uci": mv.uci(), "source": "random"}
