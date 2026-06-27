from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SF = (
    ROOT
    / "src/llm/runtime/stockfish/stockfish"
    / "stockfish-windows-x86-64-avx2.exe"
)


@dataclass(frozen=True)
class AnnotatedPosition:
    fen: str
    depth: int
    score_cp: int
    score_kind: str
    best_san: str
    best_line_sans: tuple[str, ...]
    threats_san: str | None
    # top-k root moves (san, white-POV centipawns) for the best_move top=N format
    top_moves: tuple[tuple[str, int], ...] = ()


class StockfishAnnotator:
    def __init__(self, path: Path = DEFAULT_SF, timeout: float = 5.0):
        self.path = str(path)
        self.timeout = timeout
        self._engine: chess.engine.SimpleEngine | None = None
        self._cache: dict[tuple[str, int], "AnnotatedPosition"] = {}

    def _ensure(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self._engine

    def annotate(self, fen: str, depth: int = 12) -> AnnotatedPosition:
        # Memoize by (fen, depth): the generator reuses a small pool of seed FENs
        # across ~50k rows, so this turns the regen from hours into minutes.
        key = (fen, depth)
        if key in self._cache:
            return self._cache[key]
        try:
            result = self._annotate_once(fen, depth)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, BrokenPipeError):
            self._restart()
            result = self._annotate_once(fen, depth)
        self._cache[key] = result
        return result

    def _annotate_once(self, fen: str, depth: int) -> AnnotatedPosition:
        board = chess.Board(fen)
        infos = self._ensure().analyse(
            board, chess.engine.Limit(depth=depth, time=self.timeout), multipv=3
        )
        info = infos[0]
        score = info["score"].white()
        pv = info.get("pv", [])
        best_san = board.san(pv[0]) if pv else ""
        line_sans: list[str] = []
        b = board.copy()
        for move in pv[:5]:
            line_sans.append(b.san(move))
            b.push(move)
        # top-k candidates: each multipv root move scored from White's POV
        top_moves: list[tuple[str, int]] = []
        for inf in infos:
            ipv = inf.get("pv", [])
            if not ipv:
                continue
            top_moves.append((board.san(ipv[0]), int(inf["score"].white().score(mate_score=100000))))
        # Threats via a null move are well-defined for ANY position, not just ones with a
        # move_stack — our positions are FEN-loaded (empty stack), so gating on move_stack
        # left every threats row blank. Skip only the start position (a null move there is
        # meaningless and warns); _threats itself guards check/terminal.
        threats = None if board.fen() == chess.STARTING_FEN else self._threats(board, depth)
        kind, cp = ("mate", score.mate()) if score.is_mate() else ("cp", int(score.score()))
        return AnnotatedPosition(
            fen, depth, cp, kind, best_san, tuple(line_sans), threats, tuple(top_moves),
        )

    def _restart(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
        self._engine = None

    def _threats(self, board: chess.Board, depth: int) -> str | None:
        if board.is_game_over() or board.is_check():   # can't pass the move while in check
            return None
        nb = board.copy()
        nb.push(chess.Move.null())
        info = self._ensure().analyse(
            nb, chess.engine.Limit(depth=depth, time=self.timeout)
        )
        pv = info.get("pv", [])
        return nb.san(pv[0]) if pv else None

    def quit(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            finally:
                self._engine = None
