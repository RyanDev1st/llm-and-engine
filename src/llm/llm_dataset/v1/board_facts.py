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


def legal_moves_for_square(fen: str, seed: int) -> tuple[str, list[str]]:
    """Pick a from-square (of the side to move) that has legal moves, and the
    SANs from it — so a `legal_moves square=<sq>` row is grounded and non-empty."""
    b = _board(fen)
    by_square: dict[str, list[str]] = {}
    for m in b.legal_moves:
        by_square.setdefault(chess.square_name(m.from_square), []).append(b.san(m))
    squares = sorted(by_square)
    sq = squares[seed % len(squares)]
    return sq, by_square[sq]


_PIECE_WORD = {chess.QUEEN: "queen", chess.ROOK: "rook", chess.BISHOP: "bishop", chess.KNIGHT: "knight"}
_COUNT_WORD = {1: "a", 2: "two", 3: "three", 4: "four"}


def _side(board: chess.Board, color: str) -> chess.Color:
    if color == "white":
        return chess.WHITE
    if color == "black":
        return chess.BLACK
    return board.turn   # "mine"/"" -> side to move


def list_pieces_text(fen: str, color: str = "mine") -> str:
    """The `list_pieces` tool-result string (mirrors backend tools.py / chess.py)."""
    b = _board(fen)
    col = _side(b, color)
    majors, pawns = [], []
    for sq, piece in sorted(b.piece_map().items()):
        if piece.color != col:
            continue
        name = chess.square_name(sq)
        (pawns if piece.piece_type == chess.PAWN else majors).append(
            name if piece.piece_type == chess.PAWN else f"{piece.symbol().upper()}={name}")
    parts = majors + ([f"pawns={','.join(pawns)}"] if pawns else [])
    return "pieces: " + ", ".join(parts)


def piece_summary(fen: str, color: str = "mine") -> str:
    """Human prose of the side's material (grounded in the same board as list_pieces) — for a
    final that ANSWERS 'what pieces are left?' with content, not process narration."""
    b = _board(fen)
    col = _side(b, color)
    by_type: dict[int, list[str]] = {}
    pawns = 0
    for sq, piece in b.piece_map().items():
        if piece.color != col:
            continue
        if piece.piece_type == chess.PAWN:
            pawns += 1
        elif piece.piece_type != chess.KING:
            by_type.setdefault(piece.piece_type, []).append(chess.square_name(sq))
    parts: list[str] = []
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        sqs = sorted(by_type.get(pt, []))
        if not sqs:
            continue
        word = _PIECE_WORD[pt]
        n = len(sqs)
        parts.append(f"a {word} on {sqs[0]}" if n == 1
                     else f"{_COUNT_WORD.get(n, str(n))} {word}s on {', '.join(sqs)}")
    parts.append(f"{pawns} pawn" + ("" if pawns == 1 else "s"))
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def king_moves(fen: str) -> tuple[str, list[str]]:
    """The side-to-move king's square and its legal destination SANs (for a legality-check row)."""
    b = _board(fen)
    ksq = b.king(b.turn)
    name = chess.square_name(ksq) if ksq is not None else "?"
    sans = [b.san(m) for m in b.legal_moves if m.from_square == ksq]
    return name, sans


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
