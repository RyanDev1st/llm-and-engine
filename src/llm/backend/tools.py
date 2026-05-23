"""The 9-tool executor. Maps a `<tool>NAME args</tool>` call against the live
python-chess board + Stockfish to the exact return strings of spec section 2.

review_move and threats are computed here because they compose board + engine."""
from __future__ import annotations

import chess
import chess.engine

from . import ask_kb
from .engine import Engine
from .game import Game
from .toolfmt import clamp_depth, fmt_white_score, parse_call

LABELS = (  # (max cp loss, label); >last -> blunder
    (10, "excellent"), (50, "good"), (100, "inaccuracy"), (200, "mistake"),
)


class ToolExecutor:
    def __init__(self, game: Game, engine: Engine) -> None:
        self.game = game
        self.engine = engine

    def execute(self, tool_call: str) -> str:
        name, args = parse_call(tool_call)
        if not name:
            return "error: invalid_syntax"
        try:
            return self._dispatch(name, args)
        except chess.engine.EngineError:
            return "error: engine_unavailable"
        except Exception:
            return "error: engine_unavailable"

    def _dispatch(self, name: str, args: dict[str, str]) -> str:
        if name == "move":
            return self.game.move(args.get("san", ""))
        if name == "undo":
            return self.game.undo()
        if name == "legal_moves":
            return self.game.legal_moves(args.get("square"))
        if name == "list_pieces":
            return self.game.list_pieces(args.get("color", "mine"))
        if name == "ask_chessbot":
            return ask_kb.answer(args.get("query", ""))
        if name == "eval":
            kind, val = self.engine.eval_white_cp(self.game.board, clamp_depth(args, 15))
            return fmt_white_score(kind, val, clamp_depth(args, 15))
        if name == "best_move":
            return self._best_move(args)
        if name == "review_move":
            return self._review(clamp_depth(args, 15))
        if name == "threats":
            return self._threats(clamp_depth(args, 12))
        return "error: invalid_syntax"

    def _best_move(self, args: dict[str, str]) -> str:
        depth = clamp_depth(args, 15)
        try:
            series = max(1, min(5, int(args.get("series", 1))))
        except ValueError:
            series = 1
        sans, (kind, val) = self.engine.best_line(self.game.board, depth, series)
        if not sans:
            return "best: none (no legal moves)"
        if series == 1:
            return f"best: {sans[0]}"
        score = fmt_white_score(kind, val, depth).removeprefix("score: ").split(", depth")[0]
        return f"best_line: {' '.join(sans)}, score: {score}"

    def _review(self, depth: int) -> str:
        board = self.game.board
        if not board.move_stack:
            return "error: no moves to review"
        played = self.game.san_stack[-1] if self.game.san_stack else board.peek().uci()
        mv = board.pop()
        mover = board.turn
        info_before = self.engine.analyse(board, depth)
        best = info_before.get("pv", [None])[0]
        best_san = board.san(best) if best else "?"
        before = info_before["score"].pov(mover).score(mate_score=100000)
        board.push(mv)
        after = self.engine.analyse(board, depth)["score"].pov(mover).score(mate_score=100000)
        loss = before - after
        label = next((lab for cap, lab in LABELS if loss <= cap), "blunder")
        return f"review: {played}, label={label}, delta={(after - before) / 100:+.2f} pawns, best_was={best_san}"

    def _threats(self, depth: int) -> str:
        board = self.game.board
        opp = not board.turn
        base = self.engine.analyse(board, depth)["score"].pov(opp).score(mate_score=100000)
        nb = board.copy()
        nb.push(chess.Move.null())
        best_san, (kind, val) = self.engine.best_for_side_to_move(nb, depth)
        if kind == "mate":
            return f"threats: opponent's best is {best_san}, score for them: mate in {abs(val)}"
        swing = val - base
        if swing < 50:
            return f"threats: none significant (best opponent move only changes eval by {swing / 100:.2f})"
        return f"threats: opponent's best is {best_san}, score for them: {val / 100:+.2f} pawns"
