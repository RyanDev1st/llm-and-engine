"""Board-state helpers shared by the HTTP server: snapshot the live game into a
JSON-friendly dict (FEN, eval bar, legal moves, history) for the frontend."""
from __future__ import annotations

from collections import OrderedDict

import chess

from .engine import Engine
from .game import Game

EVAL_DEPTH = 18

# Eval-bar cache, keyed by (engine-choice, 4-field FEN). snapshot() runs a depth-18 Stockfish eval
# on EVERY call — session-switch, sync, move, /api/state poll — which made switching between games
# slow (seconds each). A position's eval is identical for everyone, so cache it: switching back to a
# game you've already viewed is now instant (cache hit, no engine call). Bounded LRU; position-only
# key (move counters excluded) so transpositions/half-move differences still hit.
_EVAL_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_EVAL_CACHE_MAX = 1024


def _eval_key(who: str, board: chess.Board) -> str:
    return who + "|" + " ".join(board.fen().split()[:4])


def eval_bar(engine: Engine, board: chess.Board) -> dict:
    """White-POV evaluation for the eval bar from the SELECTED engine (Stockfish or our
    custom evaluator). Returns cp, a 0-100 bar %, and which engine produced it. Cached by
    position+engine so repeated views of the same board (notably session switches) are instant."""
    from . import eval_engines
    who = eval_engines.current()
    if board.is_game_over():
        return {"kind": "over", "cp": 0, "bar": 50, "text": _result_text(board), "engine": who}
    if board.fen() == chess.STARTING_FEN:
        return {"kind": "cp", "cp": 0, "bar": 50, "text": "0.00", "engine": who}
    key = _eval_key(who, board)
    cached = _EVAL_CACHE.get(key)
    if cached is not None:
        _EVAL_CACHE.move_to_end(key)
        return dict(cached)
    evaluator = eval_engines.bar_engine(engine)
    kind, val = evaluator.eval_white_cp(board, EVAL_DEPTH)
    if kind == "mate":
        side, n = val
        pct = 100 if side == "white" else 0
        out = {"kind": "mate", "side": side, "n": n, "bar": pct, "text": f"M{n} {side}", "engine": who}
    else:
        cp = int(val)
        bar = 1 / (1 + pow(10, -cp / 400))  # logistic win-prob style mapping
        out = {"kind": "cp", "cp": cp, "bar": round(bar * 100, 1), "text": f"{cp/100:+.2f}", "engine": who}
    _EVAL_CACHE[key] = dict(out)
    while len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
        _EVAL_CACHE.popitem(last=False)
    return out


def _result_text(board: chess.Board) -> str:
    if board.is_checkmate():
        return "0-1 checkmate" if board.turn == chess.WHITE else "1-0 checkmate"
    return "1/2-1/2 draw"


def snapshot(game: Game, engine: Engine) -> dict:
    board = game.board
    last = board.peek().uci() if board.move_stack else None
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "history": list(game.san_stack),
        "last_move": last,
        "in_check": board.is_check(),
        "game_over": board.is_game_over(),
        "legal": [m.uci() for m in board.legal_moves],
        "eval": eval_bar(engine, board),
    }
