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


class StockfishAnnotator:
    def __init__(self, path: Path = DEFAULT_SF, timeout: float = 5.0):
        self.path = str(path)
        self.timeout = timeout
        self._engine: chess.engine.SimpleEngine | None = None

    def _ensure(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self._engine

    def annotate(self, fen: str, depth: int = 12) -> AnnotatedPosition:
        try:
            return self._annotate_once(fen, depth)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, BrokenPipeError):
            self._restart()
            return self._annotate_once(fen, depth)

    def _annotate_once(self, fen: str, depth: int) -> AnnotatedPosition:
        board = chess.Board(fen)
        info = self._ensure().analyse(
            board, chess.engine.Limit(depth=depth, time=self.timeout)
        )
        score = info["score"].white()
        pv = info.get("pv", [])
        best_san = board.san(pv[0]) if pv else ""
        line_sans: list[str] = []
        b = board.copy()
        for move in pv[:5]:
            line_sans.append(b.san(move))
            b.push(move)
        threats = self._threats(board, depth)
        if score.is_mate():
            return AnnotatedPosition(
                fen, depth, score.mate(), "mate",
                best_san, tuple(line_sans), threats,
            )
        return AnnotatedPosition(
            fen, depth, int(score.score()), "cp",
            best_san, tuple(line_sans), threats,
        )

    def _restart(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
        self._engine = None

    def _threats(self, board: chess.Board, depth: int) -> str | None:
        if board.is_game_over():
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
                self._engine.quit()
            finally:
                self._engine = None
