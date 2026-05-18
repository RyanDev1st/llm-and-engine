from __future__ import annotations

from dataclasses import dataclass

from .attack import is_attacked, king_square
from .engine import ChessEngine
from .evaluation import static_evaluation

MATE = 100000


@dataclass(frozen=True)
class SearchResult:
    score: int
    pv: list[str]
    plies: int


def search_position(engine: ChessEngine, depth: int) -> SearchResult:
    plies = _plies(depth)
    return _search(engine, plies, plies, -MATE, MATE)


def _search(engine: ChessEngine, root_depth: int, depth: int, alpha: int, beta: int) -> SearchResult:
    moves = _ordered_moves(engine)
    if depth == 0 or not moves:
        return SearchResult(_terminal_or_static(engine, depth, moves), [], root_depth)
    best_score = -MATE
    best_pv: list[str] = []
    for move in moves:
        engine.move(move)
        child = _search(engine, root_depth, depth - 1, -beta, -alpha)
        engine.undo()
        score = -child.score
        if score > best_score:
            best_score = score
            best_pv = [move, *child.pv]
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return SearchResult(best_score, best_pv, root_depth)


def _ordered_moves(engine: ChessEngine) -> list[str]:
    return sorted(engine.legal_moves(), key=lambda move: _move_priority(engine, move), reverse=True)


def _move_priority(engine: ChessEngine, move: str) -> int:
    target = engine.board.piece_at(move[2:4])
    mover = engine.board.piece_at(move[:2])
    capture = 10 * _value(target) - _value(mover) if target != "." else 0
    promotion = 800 if len(move) > 4 else 0
    mate = 5000 if _is_mate(engine, move) else 0
    check = 50 if mate == 0 and _gives_check(engine, move) else 0
    return capture + promotion + mate + check


def _is_mate(engine: ChessEngine, move: str) -> bool:
    engine.move(move)
    king = king_square(engine.board, engine.board.turn)
    mated = bool(king and is_attacked(engine.board, king, _other(engine.board.turn)) and not engine.legal_moves())
    engine.undo()
    return mated


def _gives_check(engine: ChessEngine, move: str) -> bool:
    engine.move(move)
    king = king_square(engine.board, engine.board.turn)
    checked = bool(king and is_attacked(engine.board, king, _other(engine.board.turn)))
    engine.undo()
    return checked


def _terminal_or_static(engine: ChessEngine, depth: int, moves: list[str]) -> int:
    if moves:
        return _white_pov(engine) * static_evaluation(engine)
    king = king_square(engine.board, engine.board.turn)
    if king and is_attacked(engine.board, king, _other(engine.board.turn)):
        return -MATE + depth
    return 0


def _plies(depth: int) -> int:
    return min(3, max(1, depth))


def _white_pov(engine: ChessEngine) -> int:
    return 1 if engine.board.turn == "w" else -1


def _other(turn: str) -> str:
    return "b" if turn == "w" else "w"


def _value(piece: str) -> int:
    return {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}.get(piece.upper(), 0)
