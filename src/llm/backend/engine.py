"""Stockfish UCI wrapper. Owns the engine process; all chess analysis goes
through here. Returns python-chess primitives; spec-string formatting lives in
tools.py."""
from __future__ import annotations

from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SF = ROOT / "src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"


class Engine:
    def __init__(self, path: str | Path = DEFAULT_SF, timeout: float = 5.0) -> None:
        self.path = str(path)
        self.timeout = timeout
        self._eng: chess.engine.SimpleEngine | None = None

    def _ensure(self) -> chess.engine.SimpleEngine:
        if self._eng is None:
            self._eng = chess.engine.SimpleEngine.popen_uci(self.path)
        return self._eng

    def analyse(self, board: chess.Board, depth: int):
        return self._ensure().analyse(board, chess.engine.Limit(depth=depth, time=self.timeout))

    def eval_white_cp(self, board: chess.Board, depth: int):
        """Return (kind, value): ('mate',(side,n)) or ('cp', white_pov_centipawns)."""
        score = self.analyse(board, depth)["score"].white()
        if score.is_mate():
            m = score.mate()
            return ("mate", ("white" if m > 0 else "black", abs(m)))
        return ("cp", score.score())

    def best_line(self, board: chess.Board, depth: int, series: int):
        """Return (san_moves, (kind, value)) for the principal variation."""
        info = self.analyse(board, depth)
        pv = info.get("pv", [])[: max(1, series)]
        san = board.variation_san(pv) if pv else ""
        # strip move numbers from variation_san for the compact spec form
        sans = _plain_sans(board, pv)
        score = info["score"].white()
        if score.is_mate():
            m = score.mate()
            return sans, ("mate", ("white" if m > 0 else "black", abs(m)))
        return sans, ("cp", score.score())

    def best_for_side_to_move(self, board: chess.Board, depth: int):
        """Best move + score from the POV of the side to move (used by threats)."""
        info = self.analyse(board, depth)
        pv = info.get("pv", [])
        san = board.san(pv[0]) if pv else None
        pov = info["score"].pov(board.turn)
        if pov.is_mate():
            return san, ("mate", pov.mate())
        return san, ("cp", pov.score())

    def quit(self) -> None:
        if self._eng is not None:
            try:
                self._eng.quit()
            finally:
                self._eng = None


def _plain_sans(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    out, b = [], board.copy()
    for mv in moves:
        out.append(b.san(mv))
        b.push(mv)
    return out
