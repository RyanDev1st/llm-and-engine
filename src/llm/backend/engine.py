"""Stockfish UCI wrapper. Owns the engine process; all chess analysis goes
through here. Returns python-chess primitives; spec-string formatting lives in
tools.py."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[3]
_BUNDLED_SF = ROOT / "src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"


def _resolve_sf() -> str:
    """Find a runnable Stockfish: explicit CHESS_SF env > the bundled Windows exe (if it
    exists) > a `stockfish` on PATH (Linux/Colab/Kaggle apt install) > common locations.
    Returns a path string; popen errors surface later if truly absent."""
    env = os.environ.get("CHESS_SF")
    if env and Path(env).exists():
        return env
    if _BUNDLED_SF.exists():
        return str(_BUNDLED_SF)
    found = shutil.which("stockfish")
    if found:
        return found
    for p in ("/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"):
        if Path(p).exists():
            return p
    return str(_BUNDLED_SF)   # fall back to the documented default; popen_uci will error clearly


DEFAULT_SF = _resolve_sf()


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
        sans = _plain_sans(board, pv)
        score = info["score"].white()
        if score.is_mate():
            m = score.mate()
            return sans, ("mate", ("white" if m > 0 else "black", abs(m)))
        return sans, ("cp", score.score())

    def best_moves(self, board: chess.Board, depth: int, top: int):
        """Return MultiPV candidates as (san, (kind, value)) from White POV."""
        infos = self._ensure().analyse(
            board, chess.engine.Limit(depth=depth, time=self.timeout), multipv=max(1, top))
        out = []
        for info in infos[: max(1, top)]:
            pv = info.get("pv", [])
            if not pv:
                continue
            score = info["score"].white()
            if score.is_mate():
                m = score.mate()
                value = ("mate", ("white" if m > 0 else "black", abs(m)))
            else:
                value = ("cp", score.score())
            out.append((board.san(pv[0]), value))
        return out

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
