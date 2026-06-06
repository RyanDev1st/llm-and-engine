"""Board-state helpers shared by the HTTP server: snapshot the live game into a
JSON-friendly dict (FEN, eval bar, legal moves, history) for the frontend."""
from __future__ import annotations

import chess

from .engine import Engine
from .game import Game

EVAL_DEPTH = 18


def eval_bar(engine: Engine, board: chess.Board) -> dict:
    """White-POV evaluation for the left eval bar. Returns cp and a 0-100 bar %."""
    if board.is_game_over():
        return {"kind": "over", "cp": 0, "bar": 50, "text": _result_text(board)}
    if board.fen() == chess.STARTING_FEN:
        return {"kind": "cp", "cp": 0, "bar": 50, "text": "0.00"}
    kind, val = engine.eval_white_cp(board, EVAL_DEPTH)
    if kind == "mate":
        side, n = val
        pct = 100 if side == "white" else 0
        return {"kind": "mate", "side": side, "n": n, "bar": pct, "text": f"M{n} {side}"}
    cp = int(val)
    bar = 1 / (1 + pow(10, -cp / 400))  # logistic win-prob style mapping
    return {"kind": "cp", "cp": cp, "bar": round(bar * 100, 1), "text": f"{cp/100:+.2f}"}


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
