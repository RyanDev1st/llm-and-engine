import chess


_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class StaticEvaluator:
    def evaluate_position(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for piece in board.piece_map().values():
            value = _PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
        return score

    def best_move(self, board: chess.Board) -> chess.Move | None:
        ranked = self.rank_moves(board, max_moves=None)
        return ranked[0][0] if ranked else None

    def rank_moves(
        self,
        board: chess.Board,
        *,
        max_moves: int | None = None,
    ) -> list[tuple[chess.Move, int]]:
        player = board.turn
        ranked = []
        for move in list(board.legal_moves)[:max_moves]:
            before = self.evaluate_position(board)
            board.push(move)
            after = self.evaluate_position(board)
            board.pop()
            delta = after - before if player == chess.WHITE else before - after
            ranked.append((move, delta))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
