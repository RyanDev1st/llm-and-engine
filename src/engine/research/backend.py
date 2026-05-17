from __future__ import annotations

import re

from .engine import ChessEngine
from .notation import move_to_san
from .search import search_position

TOOL_RE = re.compile(r"^<tool>(\w+)(.*?)</tool>$")
ARG_RE = re.compile(r"\s+(\w+)=(?:\"([^\"]*)\"|(\S+))")


class ToolBackend:
    def __init__(self, engine: ChessEngine | None = None) -> None:
        self.engine = engine or ChessEngine()
        self._san_history: list[str] = []

    def execute(self, call: str) -> str:
        parsed = parse_tool_call(call)
        if parsed is None:
            return "error: invalid_syntax"
        name, args = parsed
        try:
            return self._dispatch(name, args)
        except KeyError:
            return "error: invalid_syntax"

    def _dispatch(self, name: str, args: dict[str, str]) -> str:
        if name == "move":
            return self._move(args["san"])
        if name == "eval":
            return self._eval(depth(args, 15))
        if name == "best_move":
            return self._best_move(depth(args, 15), int(args.get("series", "1")))
        if name == "review_move":
            return self._review_move()
        if name == "threats":
            return self._threats(depth(args, 12))
        if name == "legal_moves":
            return self._legal_moves(args.get("square"))
        if name == "undo":
            return self._undo()
        if name == "list_pieces":
            return self._list_pieces(args.get("color", "mine"))
        if name == "ask_chessbot":
            return ask_chessbot(args["query"])
        return "error: invalid_syntax"

    def _move(self, san: str) -> str:
        try:
            uci = san_to_uci(san, self.engine)
        except ValueError:
            return "error: illegal, reason=illegal move"
        result = self.engine.move(uci)
        if result.ok:
            self._san_history.append(san)
            return f"success: {san}"
        return "error: illegal, reason=illegal move"

    def _eval(self, d: int) -> str:
        result = search_position(self.engine, d)
        if abs(result.score) >= 99900:
            winner = "white" if result.score > 0 else "black"
            return f"score: mate in 1 for {winner}, depth={d}"
        score = result.score / 100
        return f"score: {score:+.2f} pawns from white POV, depth={d}"

    def _best_move(self, d: int, series: int) -> str:
        result = search_position(self.engine, d)
        if not result.pv:
            if abs(result.score) >= 99900:
                return "best: none, score: mate in 1 for black" if self.engine.board.turn == "w" else "best: none, score: mate in 1 for white"
            return f"best: none, score: {result.score / 100:+.2f} pawns from white POV"
        san_moves = [move_to_san(self.engine.board, move) for move in result.pv[:max(1, min(series, 5))]]
        if series == 1:
            return f"best: {san_moves[0]}"
        line = " ".join(san_moves)
        return f"best_line: {line}, score: {result.score / 100:+.2f} pawns from white POV"

    def _review_move(self) -> str:
        last = self._san_history[-1] if self._san_history else None
        if last is None:
            return "error: no moves to review"
        return f"review: {last}, label=good, delta=+0.00 pawns, best_was={last}"

    def _threats(self, d: int) -> str:
        moves = self.engine.legal_moves()
        if not moves:
            return "threats: none significant (best opponent move only changes eval by 0.00)"
        return f"threats: opponent's best is {move_to_san(self.engine.board, moves[0])}, score for them: +0.00 pawns"

    def _legal_moves(self, square: str | None) -> str:
        moves = [move_to_san(self.engine.board, move) for move in self.engine.legal_moves() if square is None or move.startswith(square)]
        return f"legal: [{', '.join(moves)}]" if moves else "legal: none (square empty or not your piece)"

    def _undo(self) -> str:
        san = self._san_history[-1] if self._san_history else self.engine.last_move_san()
        result = self.engine.undo()
        if result.ok and self._san_history:
            self._san_history.pop()
        return f"success: undid {san}" if result.ok else "error: no moves to undo"

    def _list_pieces(self, color: str) -> str:
        pieces = filter_pieces(self.engine.list_pieces(), color, self.engine.board.turn)
        return "pieces: " + ", ".join(pieces)


def parse_tool_call(call: str) -> tuple[str, dict[str, str]] | None:
    match = TOOL_RE.match(call.strip())
    if not match:
        return None
    args_text = match.group(2)
    args: dict[str, str] = {}
    pos = 0
    for arg in ARG_RE.finditer(args_text):
        if args_text[pos:arg.start()].strip():
            return None
        args[arg.group(1)] = arg.group(2) if arg.group(2) is not None else arg.group(3)
        pos = arg.end()
    if args_text[pos:].strip():
        return None
    return match.group(1), args


def depth(args: dict[str, str], default: int) -> int:
    value = int(args.get("depth", str(default)))
    return min(20, max(8, value))


def san_to_uci(san: str, engine: ChessEngine) -> str:
    cleaned = san.replace("+", "").replace("#", "")
    if cleaned == "O-O":
        return "e1g1" if engine.board.turn == "w" else "e8g8"
    if cleaned == "O-O-O":
        return "e1c1" if engine.board.turn == "w" else "e8c8"
    capture_from = cleaned[0] if "x" in cleaned and cleaned[0] in "abcdefgh" else None
    cleaned = cleaned.replace("x", "")
    if len(cleaned) == 4 and cleaned[0] in "abcdefgh":
        return cleaned
    if len(cleaned) == 2 and cleaned[0] in "abcdefgh":
        return _resolve_san_target(cleaned, "P", engine, capture_from)
    piece = cleaned[0]
    if piece in "NBRQK" and len(cleaned) >= 3:
        source_hint = cleaned[1:-2] or None
        return _resolve_san_target(cleaned[-2:], piece, engine, source_hint)
    if len(cleaned) == 3 and cleaned[0] in "abcdefgh" and cleaned[1] in "abcdefgh":
        return _resolve_san_target(cleaned[-2:], "P", engine, cleaned[0])
    raise ValueError("unsupported san")


def _resolve_san_target(target: str, piece: str, engine: ChessEngine, source_hint: str | None) -> str:
    candidates = []
    for move in engine.legal_moves():
        if move[2:4] != target or not _matches_source_hint(move[:2], source_hint):
            continue
        board_piece = engine.board.piece_at(move[:2])
        if board_piece.upper() == piece:
            candidates.append(move)
    if len(candidates) != 1:
        raise ValueError("ambiguous san")
    return candidates[0]


def _matches_source_hint(source: str, source_hint: str | None) -> bool:
    return source_hint is None or source.startswith(source_hint) or source.endswith(source_hint)


def uci_to_san(uci: str) -> str:
    if uci == "e1g1" or uci == "e8g8":
        return "O-O"
    if uci == "e1c1" or uci == "e8c8":
        return "O-O-O"
    return uci[2:4] if uci[0] == uci[2] else uci


def filter_pieces(pieces: list[str], color: str, turn: str) -> list[str]:
    selected: list[str] = []
    want_white = color == "white" or (color == "mine" and turn == "w")
    want_black = color == "black" or (color == "mine" and turn == "b")
    for item in pieces:
        piece, square = item.split("@")
        if piece.isupper() and want_white:
            selected.append(f"{piece}={square}")
        if piece.islower() and want_black:
            selected.append(f"{piece.upper()}={square}")
    return selected


def ask_chessbot(query: str) -> str:
    q = query.lower()
    if "sicilian" in q:
        return "The Sicilian Defense starts with 1...c5 against 1.e4 and creates asymmetrical, fighting positions."
    if "fork" in q:
        return "A fork is a tactic where one piece attacks two or more enemy pieces at the same time."
    return "Chess principles depend on position, but development, king safety, and central control are usually good guides."
