from __future__ import annotations

import re

from .engine import ChessEngine
from .notation import move_to_san, san_line
from .review import review_last_move
from .search import MATE, SearchResult, search_position

TOOL_RE = re.compile(r"^<tool>(\w+)(.*?)</tool>$")
ARG_RE = re.compile(r"\s+(\w+)=(?:\"([^\"]*)\"|(\S+))")


class ToolBackend:
    def __init__(self, engine: ChessEngine | None = None) -> None:
        self.engine = engine or ChessEngine()
        self._move_history: list[tuple[str, str]] = []

    def execute(self, call: str) -> str:
        parsed = parse_tool_call(call)
        if parsed is None:
            return "error: invalid_syntax"
        try:
            return self._dispatch(*parsed)
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
            self._move_history.append((san, uci))
            return f"success: {san}"
        return "error: illegal, reason=illegal move"

    def _eval(self, d: int) -> str:
        result = search_position(self.engine, d)
        depth_note = f"requested_depth={d}, searched_plies={result.plies}"
        if abs(result.score) >= 99900:
            return f"score: {mate_label(result)}, {depth_note}"
        score = result.score / 100
        return f"score: {score:+.2f} pawns from white POV, {depth_note}"

    def _best_move(self, d: int, series: int) -> str:
        result = search_position(self.engine, d)
        score_note = f"requested_depth={d}, searched_plies={result.plies}"
        if not result.pv:
            return no_best(result, score_note)
        san_moves = san_line(self.engine, result.pv[:max(1, min(series, 5))])
        if series == 1:
            return f"best: {san_moves[0]}, {score_note}"
        line = " ".join(san_moves)
        return f"best_line: {line}, score: {result.score / 100:+.2f} pawns from white POV, {score_note}"

    def _review_move(self) -> str:
        if not self._move_history:
            return "error: no moves to review"
        san, uci = self._move_history[-1]
        return review_last_move(self.engine, san, uci, 15)

    def _threats(self, d: int) -> str:
        result = search_position(self.engine, d)
        if not result.pv:
            return "threats: none significant (no legal opponent moves)"
        san = move_to_san(self.engine.board, result.pv[0])
        score = (1 if self.engine.board.turn == "w" else -1) * result.score / 100
        return f"threats: best reply is {san}, score for side to move: {score:+.2f} pawns"

    def _legal_moves(self, square: str | None) -> str:
        moves = [move_to_san(self.engine.board, move) for move in self.engine.legal_moves() if square is None or move.startswith(square)]
        return f"legal: [{', '.join(moves)}]" if moves else "legal: none (square empty or not your piece)"

    def _undo(self) -> str:
        san = self._move_history[-1][0] if self._move_history else self.engine.last_move_san()
        result = self.engine.undo()
        if result.ok and self._move_history:
            self._move_history.pop()
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


def mate_label(result: SearchResult) -> str:
    winner = "white" if result.score > 0 else "black"
    distance = MATE - abs(result.score)
    if distance > 0:
        return f"mate in {distance} for {winner}"
    return f"mate for {winner}"


def no_best(result: SearchResult, score_note: str) -> str:
    score = mate_label(result) if abs(result.score) >= 99900 else f"{result.score / 100:+.2f} pawns from white POV"
    return f"best: none, score: {score}, {score_note}"


def depth(args: dict[str, str], default: int) -> int:
    value = int(args.get("depth", str(default)))
    return min(20, max(8, value))


def san_to_uci(san: str, engine: ChessEngine) -> str:
    cleaned = san.replace("+", "").replace("#", "")
    if cleaned == "O-O":
        return f"e{'1' if engine.board.turn == 'w' else '8'}g{'1' if engine.board.turn == 'w' else '8'}"
    if cleaned == "O-O-O":
        return f"e{'1' if engine.board.turn == 'w' else '8'}c{'1' if engine.board.turn == 'w' else '8'}"
    promotion = ""
    if "=" in cleaned:
        cleaned, promote = cleaned.split("=", 1); promotion = promote.lower()
    capture_from = cleaned[0] if "x" in cleaned and cleaned[0] in "abcdefgh" else None
    cleaned = cleaned.replace("x", "")
    if len(cleaned) == 4 and cleaned[0] in "abcdefgh":
        return cleaned + promotion
    if len(cleaned) == 2 and cleaned[0] in "abcdefgh":
        return _resolve_san_target(cleaned, "P", engine, capture_from, promotion)
    if len(cleaned) == 3 and cleaned[0] in "abcdefgh" and cleaned[1] in "abcdefgh":
        return _resolve_san_target(cleaned[-2:], "P", engine, cleaned[0], promotion)
    if cleaned[0] in "NBRQK" and len(cleaned) >= 3:
        return _resolve_san_target(cleaned[-2:], cleaned[0], engine, cleaned[1:-2] or None, promotion)
    raise ValueError("unsupported san")


def _resolve_san_target(target: str, piece: str, engine: ChessEngine, source_hint: str | None, promotion: str = "") -> str:
    candidates = []
    for move in engine.legal_moves():
        if move[2:4] != target or not _matches_source_hint(move[:2], source_hint) or move[4:] != promotion:
            continue
        board_piece = engine.board.piece_at(move[:2])
        if board_piece.upper() == piece:
            candidates.append(move)
    if len(candidates) != 1: raise ValueError("ambiguous san")
    return candidates[0]


def _matches_source_hint(source: str, source_hint: str | None) -> bool:
    return source_hint is None or source.startswith(source_hint) or source.endswith(source_hint)


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
    return "A fork is a tactic where one piece attacks two or more enemy pieces at the same time." if "fork" in q else "Chess principles depend on position, but development, king safety, and central control are usually good guides."
