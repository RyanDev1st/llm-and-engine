import chess


PROMOTION_TO_INDEX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def move_label_parts(move: chess.Move, board: chess.Board) -> tuple[int, int, int]:
    return (
        orient_square(move.from_square, board),
        orient_square(move.to_square, board),
        PROMOTION_TO_INDEX.get(move.promotion, 0),
    )


def score_move_from_parts(
    move: chess.Move,
    board: chess.Board,
    from_scores: list[float],
    to_scores: list[float],
    promo_scores: list[float],
) -> float:
    from_index, to_index, promo_index = move_label_parts(move, board)
    return from_scores[from_index] + to_scores[to_index] + promo_scores[promo_index]


def orient_square(square: chess.Square, board: chess.Board) -> chess.Square:
    return chess.square_mirror(square) if board.turn == chess.BLACK else square
