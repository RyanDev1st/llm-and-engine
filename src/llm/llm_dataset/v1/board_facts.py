"""FEN-grounded helpers so generated chess rows match the REAL backend.

Tool-result strings here mirror src/llm/backend/game.py and tools.py exactly:
- move success -> "success: <san>[, game_over=...]"
- illegal move -> "error: illegal, reason=..."
- board_state 'basic' -> "board_state: turn=..., last_move=..., check=..., legal_count=..."
  (no fen, matching ToolExecutor._board_state)
"""
from __future__ import annotations

import chess


def _board(fen: str) -> chess.Board:
    return chess.Board(fen)


def board_state_line(fen: str, fields: str = "basic") -> str:
    b = _board(fen)
    values = {
        "turn": "white" if b.turn == chess.WHITE else "black",
        "fen": b.fen(),
        "last_move": "none",  # generated positions carry no move history
        "check": "yes" if b.is_check() else "no",
        "legal_count": str(b.legal_moves.count()),
    }
    requested = {p.strip() for p in fields.split(",") if p.strip()}
    if not requested or "basic" in requested:
        requested = {"turn", "last_move", "check", "legal_count"}
    if "all" in requested:
        requested = {"turn", "fen", "last_move", "check", "legal_count"}
    parts = [f"{k}={values[k]}" for k in ("turn", "fen", "last_move", "check", "legal_count") if k in requested]
    return "board_state: " + ", ".join(parts)


def legal_sans(fen: str) -> list[str]:
    b = _board(fen)
    return [b.san(m) for m in b.legal_moves]


def choose_move(fen: str, seed: int, requested: str | None = None) -> str:
    """Return a legal SAN. Honor `requested` iff legal; else a deterministic legal pick."""
    b = _board(fen)
    if requested:
        try:
            b.parse_san(requested)
            return b.san(b.parse_san(requested))
        except ValueError:
            pass
    sans = legal_sans(fen)
    return sans[seed % len(sans)]


def _game_over_suffix(board: chess.Board) -> str:
    if board.is_checkmate():
        return ", game_over=checkmate"
    if board.is_stalemate():
        return ", game_over=stalemate"
    if board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return ", game_over=draw"
    return ""


def move_echo(fen: str, san: str) -> str:
    """Mirror backend/game.py Game.move() result for `san` applied to `fen`."""
    b = _board(fen)
    try:
        mv = b.parse_san(san)
    except chess.AmbiguousMoveError:
        return "error: ambiguous"
    except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError):
        return "error: illegal, reason=that move isn't legal in this position"
    clean = b.san(mv)
    b.push(mv)
    return f"success: {clean}{_game_over_suffix(b)}"
