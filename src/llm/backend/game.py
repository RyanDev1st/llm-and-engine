"""Authoritative board state (python-chess). Owns FEN + move stack; the model
never sees any of it. Methods return spec-exact strings for the state-only
tools (move/undo/legal_moves/list_pieces); analysis tools live in tools.py."""
from __future__ import annotations

import chess


class Game:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.san_stack: list[str] = []

    def load_uci_moves(self, moves: list[str]) -> bool:
        """Reset to the start and replay a UCI move list (from the client-side
        board, which stays authoritative for smooth play). Replaying — rather
        than loading a FEN — preserves move_stack + san_stack so review_move and
        undo keep working. Returns False on the first illegal move."""
        self.board = chess.Board()
        self.san_stack = []
        for uci in moves:
            if not self.move_uci(str(uci)):
                return False
        return True

    def load_fen(self, fen: str) -> bool:
        """Set the board to an arbitrary position (puzzle setup, FEN paste). A FEN
        is a snapshot, so move history starts fresh from here. Returns False on an
        invalid/illegal FEN without disturbing the current board."""
        try:
            board = chess.Board(str(fen).strip())
        except ValueError:
            return False
        if not board.is_valid():
            return False
        self.board = board
        self.san_stack = []
        return True

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

    def over_status(self) -> str:
        """'checkmate' / 'stalemate' / 'draw' if the game has ended, else ''.
        Used by the routing layer to stop the model calling analysis tools on a
        finished game (it should state the result instead)."""
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return "draw"
        return ""

    def _game_over_suffix(self) -> str:
        status = self.over_status()
        return f", game_over={status}" if status else ""

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

    def move_uci(self, uci: str) -> bool:
        """Apply a drag-and-drop move given in UCI (e.g. e2e4, e7e8q). Returns
        True if legal and played. Board stays authoritative for the frontend."""
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return False
        if mv not in self.board.legal_moves:
            return False
        self.san_stack.append(self.board.san(mv))
        self.board.push(mv)
        return True

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
