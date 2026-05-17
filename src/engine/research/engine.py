from __future__ import annotations

from dataclasses import replace

from .attack import coords, inside, is_attacked, king_square, owns
from .board import BoardState, MoveResult
from .castle import CASTLES, castle_moves, castling_rights_after

VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
DIRECTIONS = {
    "N": ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)),
    "B": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    "R": ((-1, 0), (1, 0), (0, -1), (0, 1)),
    "Q": ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)),
    "K": ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
}
FILES = "abcdefgh"


class ChessEngine:
    def __init__(self, board: BoardState | None = None) -> None:
        self.board = board or BoardState.start()
        self._history: list[tuple[BoardState, str]] = []

    def load_fen(self, fen: str) -> None:
        self.board = BoardState.from_fen(fen)
        self._history = []

    def move(self, uci: str) -> MoveResult:
        if uci not in self.legal_moves():
            return MoveResult(False, f"illegal move: {uci}", self.board.to_fen())
        self._history.append((self.board, uci))
        self.board = self._apply_uci(uci)
        return MoveResult(True, f"played: {uci}", self.board.to_fen())

    def undo(self) -> MoveResult:
        if not self._history:
            return MoveResult(False, "error: no move to undo", self.board.to_fen())
        self.board = self._history.pop()[0]
        return MoveResult(True, "undone", self.board.to_fen())

    def last_move_san(self) -> str | None:
        if not self._history:
            return None
        return _square_from_uci(self._history[-1][1])

    def legal_moves(self) -> list[str]:
        return sorted(move for move in self._pseudo_moves() if self._is_legal(move))

    def next_board(self, uci: str) -> BoardState:
        if uci not in self.legal_moves():
            raise ValueError(f"illegal move: {uci}")
        return self._apply_uci(uci)

    def list_pieces(self) -> list[str]:
        pieces: list[str] = []
        for row in range(8):
            for col in range(8):
                piece = self.board.squares[row][col]
                if piece != ".":
                    pieces.append(f"{piece}@{_square(row, col)}")
        return sorted(pieces, key=lambda item: item.split("@")[1])

    def evaluate_material(self) -> int:
        score = 0
        for row in self.board.squares:
            for piece in row:
                if piece != ".":
                    value = VALUES[piece.upper()]
                    score += value if piece.isupper() else -value
        return score

    def _pseudo_moves(self) -> list[str]:
        moves: list[str] = []
        for row in range(8):
            for col in range(8):
                piece = self.board.squares[row][col]
                if piece != "." and owns(piece, self.board.turn):
                    moves.extend(self._piece_moves(row, col, piece))
        return moves

    def _is_legal(self, uci: str) -> bool:
        color = self.board.turn
        if self.board.piece_at(uci[2:4]).upper() == "K":
            return False
        if uci in CASTLES and self._castle_crosses_attack(uci, color):
            return False
        next_board = self._apply_uci(uci)
        king = king_square(next_board, color)
        return king is not None and not is_attacked(next_board, king, _other(color))

    def _castle_crosses_attack(self, uci: str, color: str) -> bool:
        path = {"e1g1": ("e1", "f1", "g1"), "e1c1": ("e1", "d1", "c1"), "e8g8": ("e8", "f8", "g8"), "e8c8": ("e8", "d8", "c8")}
        return any(is_attacked(self.board, square, _other(color)) for square in path[uci])

    def _apply_uci(self, uci: str) -> BoardState:
        if uci in CASTLES:
            return self._apply_castle(uci)
        source, target = uci[:2], uci[2:4]
        piece = self.board.piece_at(source)
        placed = uci[4].upper() if len(uci) > 4 and piece.isupper() else uci[4] if len(uci) > 4 else piece
        next_board = self.board.with_piece(source, ".").with_piece(target, placed)
        next_board = replace(next_board, castling=castling_rights_after(self.board.castling, source, target))
        if piece.upper() == "P" or self.board.piece_at(target) != ".":
            next_board = replace(next_board, halfmove=-1)
        return next_board.switch_turn()

    def _piece_moves(self, row: int, col: int, piece: str) -> list[str]:
        kind = piece.upper()
        if kind == "P":
            return self._pawn_moves(row, col, piece)
        if kind == "N":
            return self._jump_moves(row, col, piece, DIRECTIONS["N"])
        if kind == "K":
            return self._jump_moves(row, col, piece, DIRECTIONS["K"]) + castle_moves(self.board, piece, _square(row, col))
        return self._slide_moves(row, col, piece, DIRECTIONS[kind])

    def _pawn_moves(self, row: int, col: int, piece: str) -> list[str]:
        step = -1 if piece.isupper() else 1
        start = 6 if piece.isupper() else 1
        promote = 0 if piece.isupper() else 7
        moves: list[str] = []
        if inside(row + step, col) and self.board.squares[row + step][col] == ".":
            moves.extend(_pawn_uci(row, col, row + step, col, row + step == promote))
            if row == start and self.board.squares[row + step * 2][col] == ".":
                moves.append(_uci(row, col, row + step * 2, col))
        for dc in (-1, 1):
            nr, nc = row + step, col + dc
            if inside(nr, nc) and _enemy(piece, self.board.squares[nr][nc]):
                moves.extend(_pawn_uci(row, col, nr, nc, nr == promote))
        return moves

    def _jump_moves(self, row: int, col: int, piece: str, dirs: tuple[tuple[int, int], ...]) -> list[str]:
        return [_uci(row, col, row + dr, col + dc) for dr, dc in dirs if inside(row + dr, col + dc) and not _friend(piece, self.board.squares[row + dr][col + dc])]

    def _slide_moves(self, row: int, col: int, piece: str, dirs: tuple[tuple[int, int], ...]) -> list[str]:
        moves: list[str] = []
        for dr, dc in dirs:
            nr, nc = row + dr, col + dc
            while inside(nr, nc):
                occupant = self.board.squares[nr][nc]
                if _friend(piece, occupant):
                    break
                moves.append(_uci(row, col, nr, nc))
                if _enemy(piece, occupant):
                    break
                nr += dr
                nc += dc
        return moves

    def _apply_castle(self, uci: str) -> BoardState:
        king_from, king_to, rook_from, rook_to = CASTLES[uci]
        next_board = self.board.with_piece(king_from, ".").with_piece(rook_from, ".")
        next_board = next_board.with_piece(king_to, self.board.piece_at(king_from))
        next_board = next_board.with_piece(rook_to, self.board.piece_at(rook_from))
        next_board = replace(next_board, castling=castling_rights_after(self.board.castling, king_from, rook_from))
        return next_board.switch_turn()


def _friend(piece: str, other: str) -> bool:
    return other != "." and piece.isupper() == other.isupper()


def _enemy(piece: str, other: str) -> bool:
    return other != "." and piece.isupper() != other.isupper()


def _other(turn: str) -> str:
    return "b" if turn == "w" else "w"


def _square(row: int, col: int) -> str:
    return f"{FILES[col]}{8 - row}"


def _uci(a: int, b: int, c: int, d: int) -> str:
    return f"{_square(a, b)}{_square(c, d)}"


def _pawn_uci(a: int, b: int, c: int, d: int, promote: bool) -> list[str]:
    move = _uci(a, b, c, d)
    return [f"{move}{piece}" for piece in "qrbn"] if promote else [move]


def _square_from_uci(uci: str) -> str:
    return uci[2:4] if uci[0] == uci[2] else uci
