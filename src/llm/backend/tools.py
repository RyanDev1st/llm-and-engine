"""The tool executor. Maps a `<tool>NAME args</tool>` call against the live
python-chess board + Stockfish to the exact return strings of spec section 2,
plus load_skill (progressive disclosure — returns a skill's SKILL.md body).

review_move and threats are computed here because they compose board + engine."""
from __future__ import annotations

import chess
import chess.engine

from . import ask_kb
from .engine import Engine
from .game import Game
from .skills import load_skills
from .toolfmt import clamp_depth, fmt_white_score, parse_call

LABELS = (  # (max cp loss, label); >last -> blunder
    (10, "excellent"), (50, "good"), (100, "inaccuracy"), (200, "mistake"),
)
DEFAULT_EVAL_DEPTH = 18


def parse_range(value: str | None, default: int, high: int) -> int:
    try:
        n = int(value) if value is not None else default
    except ValueError:
        n = default
    return max(1, min(high, n))


def format_score(kind: str, val) -> str:
    if kind == "mate":
        side, n = val
        return f"M{n} for {side}"
    return f"{int(val) / 100:+.2f}"


class ToolExecutor:
    def __init__(self, game: Game, engine: Engine, plugin_context: dict | None = None) -> None:
        self.game = game
        self.engine = engine
        # Which plugin bundles are enabled — gates which plugin tools dispatch + which
        # plugin skill bodies load_skill can fetch. Set by the loop/web_app.
        self.plugin_context = plugin_context

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
        if name == "load_skill":
            return self._load_skill(args.get("name", ""))
        if name == "board_state":
            return self._board_state(args.get("fields", "basic"))
        if name == "move":
            return self.game.move(args.get("san", ""))
        if name == "load_fen":
            fen = args.get("fen", "")
            if self.game.load_fen(fen):
                return self._board_state("all")
            return "error: invalid_fen"
        if name == "random_position":
            from .positions import random_position
            return random_position(self.game, args.get("kind", "puzzle"), engine=self.engine)
        if name == "fetch_puzzle":
            from .online_positions import fetch_puzzle
            return fetch_puzzle(self.game)
        if name == "undo":
            return self.game.undo()
        if name == "legal_moves":
            return self.game.legal_moves(args.get("square"))
        if name == "list_pieces":
            return self.game.list_pieces(args.get("color", "mine"))
        if name == "ask_chessbot":
            return ask_kb.answer(args.get("query", ""))
        if name == "eval":
            depth = clamp_depth(args, DEFAULT_EVAL_DEPTH)
            if self.game.board.fen() == chess.STARTING_FEN:
                return f"score: 0.00 pawns from white POV, depth={depth} (starting position is equal)"
            kind, val = self.engine.eval_white_cp(self.game.board, depth)
            return fmt_white_score(kind, val, depth)
        if name == "best_move":
            return self._best_move(args)
        if name == "review_move":
            return self._review(clamp_depth(args, 15))
        if name == "threats":
            return self._threats(clamp_depth(args, 12))
        # Enabled plugins' tools (openings/analysis/...) — the registry routes to the
        # plugin that owns the name. This is the surface that tests cross-bundle routing.
        from . import plugins
        plugin_res = plugins.dispatch(name, args, self, self.plugin_context)
        if plugin_res is not None:
            return plugin_res
        # A skill is NOT a tool. The harness contract is load_skill name=<skill>.
        # If the model calls a known skill BY NAME as a tool
        # (<tool>chess-coach</tool>), do NOT silently accept it — that would mask
        # the protocol violation. Return a corrective error naming the right
        # call so the loop retries with load_skill (self-correction, enforced).
        plugin_skill_names = {s["name"] for s in plugins.plugin_skills(self.plugin_context)}
        if name in {s.name for s in load_skills()} | plugin_skill_names:
            return f"error: '{name}' is a skill, not a tool — call load_skill name={name}"
        return "error: invalid_syntax"

    def _load_skill(self, name: str) -> str:
        """Progressive disclosure: return the named skill's full SKILL.md body so
        the model can follow it. Checks the skills dir first, then enabled plugins'
        bundled skills (openings/analysis/...)."""
        bodies = {skill.name: skill.content for skill in load_skills()}
        if name in bodies:
            return bodies[name]
        from . import plugins
        body = plugins.skill_body(name, self.plugin_context)
        return body if body is not None else "error: unknown_skill"

    def _board_state(self, fields: str) -> str:
        board = self.game.board
        requested = {part.strip() for part in fields.split(",") if part.strip()}
        if not requested or "basic" in requested:
            requested = {"turn", "last_move", "check", "legal_count"}
        if "all" in requested:
            requested = {"turn", "fen", "last_move", "check", "legal_count", "history"}
        values = {
            "turn": "white" if board.turn == chess.WHITE else "black",
            "fen": board.fen(),
            "last_move": self.game.san_stack[-1] if self.game.san_stack else "none",
            "check": "yes" if board.is_check() else "no",
            "legal_count": str(board.legal_moves.count()),
            "history": " ".join(self.game.san_stack) or "none",
        }
        parts = [f"{key}={values[key]}" for key in values if key in requested]
        return "board_state: " + ", ".join(parts)

    def _best_move(self, args: dict[str, str]) -> str:
        depth = clamp_depth(args, DEFAULT_EVAL_DEPTH)
        top = parse_range(args.get("top"), 1, 5)
        series = parse_range(args.get("series"), 1, 5)
        if "top" in args:
            moves = self.engine.best_moves(self.game.board, depth, top)
            if not moves:
                return "best: none (no legal moves)"
            if series > 1:
                sans, (kind, val) = self.engine.best_line(self.game.board, depth, series)
                score = fmt_white_score(kind, val, depth).removeprefix("score: ").split(", depth")[0]
                head = "; ".join(f"{i}. {san} ({format_score(kind, val)})" for i, (san, (kind, val)) in enumerate(moves, 1))
                return f"best_moves: {head}; best_line: {' '.join(sans)}, score: {score}"
            return "best_moves: " + "; ".join(
                f"{i}. {san} ({format_score(kind, val)})" for i, (san, (kind, val)) in enumerate(moves, 1))
        sans, (kind, val) = self.engine.best_line(self.game.board, depth, series)
        if not sans:
            return "best: none (no legal moves)"
        score = fmt_white_score(kind, val, depth).removeprefix("score: ").split(", depth")[0]
        if series == 1:
            # Always attach the score, in the same ", score: ... pawns from white POV"
            # idiom as best_line/best_moves. The bare "best: <move>" form carried no
            # number, and the model (trained only on scored best_* results) then
            # invented one from opening priors. Grounding it kills that fabrication.
            return f"best: {sans[0]}, score: {score}"
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
