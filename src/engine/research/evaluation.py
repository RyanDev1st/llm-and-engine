from __future__ import annotations

from .board import BoardState
from .engine import ChessEngine

FILES = "abcdefgh"


def static_evaluation(engine: ChessEngine) -> int:
    return engine.evaluate_material() + pawn_structure(engine.board) + piece_activity(engine.board)


def pawn_structure(board: BoardState) -> int:
    white = pawn_files(board, "P")
    black = pawn_files(board, "p")
    return side_pawn_score(white, black, True) - side_pawn_score(black, white, False)


def pawn_files(board: BoardState, pawn: str) -> dict[int, list[int]]:
    files: dict[int, list[int]] = {index: [] for index in range(8)}
    for row, rank in enumerate(board.squares):
        for col, piece in enumerate(rank):
            if piece == pawn:
                files[col].append(8 - row)
    return files


def side_pawn_score(own: dict[int, list[int]], enemy: dict[int, list[int]], white: bool) -> int:
    score = 0
    for file, ranks in own.items():
        if len(ranks) > 1:
            score -= 15 * (len(ranks) - 1)
        for rank in ranks:
            if isolated(file, own):
                score -= 20
            if passed(file, rank, enemy, white):
                score += 20 + 5 * advancement(rank, white)
    return score


def piece_activity(board: BoardState) -> int:
    if sum(piece != "." for rank in board.squares for piece in rank) > 10:
        return 0
    score = 0
    for row, rank in enumerate(board.squares):
        for col, piece in enumerate(rank):
            if piece.upper() in {"B", "N"}:
                value = center_bonus(row, col)
                score += value if piece.isupper() else -value
    return score


def center_bonus(row: int, col: int) -> int:
    distance = abs(row - 3.5) + abs(col - 3.5)
    return int((7 - distance) * 4)


def isolated(file: int, own: dict[int, list[int]]) -> bool:
    return not any(own.get(adjacent) for adjacent in (file - 1, file + 1))


def passed(file: int, rank: int, enemy: dict[int, list[int]], white: bool) -> bool:
    for adjacent in (file - 1, file, file + 1):
        for enemy_rank in enemy.get(adjacent, []):
            if white and enemy_rank > rank:
                return False
            if not white and enemy_rank < rank:
                return False
    return True


def advancement(rank: int, white: bool) -> int:
    return rank - 2 if white else 7 - rank
