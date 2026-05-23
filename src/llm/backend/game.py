"""Authoritative board state (python-chess). Owns FEN + move stack; the model
never sees any of it. Methods return spec-exact strings for the state-only
tools (move/undo/legal_moves/list_pieces); analysis tools live in tools.py."""
from __future__ import annotations

import chess


class Game:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.san_stack: list[str] = []

    # ---- move -------------------------------------------------------------
    def move(self, san: str) -> str:
        try:
            mv = self.board.parse_san(san)
        except chess.InvalidMoveError:
            return "error: invalid_syntax"
        except chess.AmbiguousMoveError:
            return f"error: ambiguous, options=[{', '.join(self._options(san))}]"
        except chess.IllegalMoveError:
            return f"error: illegal, reason={self._illegal_reason(san)}"
        clean = self.board.san(mv)
        self.board.push(mv)
        self.san_stack.append(clean)
        return f"success: {clean}{self._game_over_suffix()}"

    def _game_over_suffix(self) -> str:
        if self.board.is_checkmate():
            return ", game_over=checkmate"
        if self.board.is_stalemate():
            return ", game_over=stalemate"
        if self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return ", game_over=draw"
        return ""

    def _options(self, san: str) -> list[str]:
        target = "".join(c for c in san if c not in "+#")[-2:]
        try:
            sq = chess.parse_square(target)
        except ValueError:
            return []
        out = []
        for mv in self.board.legal_moves:
            if mv.to_square == sq and san[0] == self.board.san(mv)[0]:
                out.append(self.board.san(mv))
        return out

    def _illegal_reason(self, san: str) -> str:
        flipped = self.board.copy()
        flipped.push(chess.Move.null())
        try:
            flipped.parse_san(san)
            return "wrong color's turn"
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            pass
        if self.board.is_check():
            return "your king is in check"
        return "that move isn't legal in this position"

    # ---- undo -------------------------------------------------------------
    def undo(self) -> str:
        if not self.board.move_stack:
            return "error: no moves to undo"
        self.board.pop()
        san = self.san_stack.pop() if self.san_stack else "?"
        return f"success: undid {san}"

    # ---- legal_moves ------------------------------------------------------
    def legal_moves(self, square: str | None) -> str:
        moves = list(self.board.legal_moves)
        if square:
            try:
                sq = chess.parse_square(square)
            except ValueError:
                return "legal: none (square empty or not your piece)"
            moves = [m for m in moves if m.from_square == sq]
            if not moves:
                return "legal: none (square empty or not your piece)"
        sans = [self.board.san(m) for m in moves]
        return f"legal: [{', '.join(sans)}]"

    # ---- list_pieces ------------------------------------------------------
    def list_pieces(self, color: str) -> str:
        col = {"white": chess.WHITE, "black": chess.BLACK}.get(color, self.board.turn)
        majors, pawns = [], []
        for sq, piece in sorted(self.board.piece_map().items()):
            if piece.color != col:
                continue
            name = chess.square_name(sq)
            if piece.piece_type == chess.PAWN:
                pawns.append(name)
            else:
                majors.append(f"{piece.symbol().upper()}={name}")
        parts = majors + ([f"pawns={','.join(pawns)}"] if pawns else [])
        return f"pieces: {', '.join(parts)}"
