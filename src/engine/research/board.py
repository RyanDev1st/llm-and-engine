from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefgh"
PIECES = {"K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"}
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(frozen=True)
class MoveResult:
    ok: bool
    message: str
    fen: str


@dataclass(frozen=True)
class BoardState:
    squares: tuple[tuple[str, ...], ...]
    turn: str
    castling: str
    en_passant: str
    halfmove: int
    fullmove: int

    @classmethod
    def start(cls) -> "BoardState":
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "BoardState":
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError("fen must have 6 fields")
        rows = tuple(_parse_rank(rank) for rank in parts[0].split("/"))
        if len(rows) != 8:
            raise ValueError("fen must have 8 ranks")
        turn, castling, en_passant, halfmove, fullmove = parts[1:]
        if turn not in {"w", "b"}:
            raise ValueError("fen turn must be w or b")
        return cls(rows, turn, castling, en_passant, int(halfmove), int(fullmove))

    def to_fen(self) -> str:
        board = "/".join(_format_rank(rank) for rank in self.squares)
        return f"{board} {self.turn} {self.castling} {self.en_passant} {self.halfmove} {self.fullmove}"

    def piece_at(self, square: str) -> str:
        rank, file = _coords(square)
        return self.squares[rank][file]

    def with_piece(self, square: str, piece: str) -> "BoardState":
        rank, file = _coords(square)
        rows = [list(row) for row in self.squares]
        rows[rank][file] = piece
        return BoardState(tuple(tuple(row) for row in rows), self.turn, self.castling, self.en_passant, self.halfmove, self.fullmove)

    def switch_turn(self, en_passant: str = "-") -> "BoardState":
        fullmove = self.fullmove + (1 if self.turn == "b" else 0)
        return BoardState(self.squares, "b" if self.turn == "w" else "w", self.castling, en_passant, self.halfmove + 1, fullmove)


def _parse_rank(rank: str) -> tuple[str, ...]:
    row: list[str] = []
    for char in rank:
        if char.isdigit():
            row.extend("." for _ in range(int(char)))
        elif char in PIECES:
            row.append(char)
        else:
            raise ValueError(f"bad fen piece: {char}")
    if len(row) != 8:
        raise ValueError("fen rank must contain 8 squares")
    return tuple(row)


def _format_rank(rank: tuple[str, ...]) -> str:
    out = ""
    empty = 0
    for piece in rank:
        if piece == ".":
            empty += 1
        else:
            if empty:
                out += str(empty)
                empty = 0
            out += piece
    return out + (str(empty) if empty else "")


def _coords(square: str) -> tuple[int, int]:
    if len(square) != 2 or square[0] not in FILES or square[1] not in "12345678":
        raise ValueError(f"bad square: {square}")
    return 8 - int(square[1]), FILES.index(square[0])
