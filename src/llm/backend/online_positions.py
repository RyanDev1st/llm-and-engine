"""Real position source: fetch a rated tactical puzzle from Lichess (public API, no
key) and set it on the board. This is the "search a real FEN board online" capability —
unlike the local curated bank, the FEN, themes, rating, and SOLUTION are real and
verifiable, so the coach narrates a grounded tactic instead of a hand-labeled guess.

Robust by construction: short timeout, /next (rotating) then /daily fallback, and on
ANY network/parse failure we fall back to the local bank so the tool never hangs the
loop or errors out. stdlib urllib only — no new dependency."""
from __future__ import annotations

import json
import urllib.request

import chess

_UA = {"User-Agent": "chess-coach/0.1 (local training demo)"}
_ENDPOINTS = ("https://lichess.org/api/puzzle/next", "https://lichess.org/api/puzzle/daily")


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (fixed https host)
        return json.load(r)


def _fetch_puzzle_json(timeout: float) -> dict | None:
    """One rotating puzzle; fall back to the daily if /next is unavailable."""
    for url in _ENDPOINTS:
        try:
            return _get_json(url, timeout)
        except Exception:  # noqa: BLE001 — try the next endpoint, else local fallback
            continue
    return None


def fetch_puzzle(game, timeout: float = 6.0) -> str:
    """Set `game` to a real Lichess puzzle and return a grounded description. The FEN's
    side-to-move is the solver; solution[0] is the answer (rendered as SAN so the coach
    can reveal it if asked). Falls back to the local curated bank if offline."""
    data = _fetch_puzzle_json(timeout)
    if not data:
        from .positions import random_position
        return "note: online puzzle source unavailable, using a local puzzle. " \
               + random_position(game, "puzzle")
    pz = data.get("puzzle", {}) or {}
    fen = pz.get("fen", "")
    if not fen or not game.load_fen(fen):
        from .positions import random_position
        return "note: online puzzle FEN unusable, using a local puzzle. " \
               + random_position(game, "puzzle")
    side = "white" if game.board.turn == chess.WHITE else "black"
    themes = ", ".join(pz.get("themes", []) or []) or "tactics"
    rating = pz.get("rating", "?")
    pid = pz.get("id", "?")
    answer = _solution_san(fen, pz.get("solution", []) or [])
    ans = f" answer={answer}" if answer else ""
    return (f"position: lichess puzzle {pid} (rating {rating}, themes: {themes}). "
            f"{side} to move and find the tactic. fen={fen}{ans}")


def _solution_san(fen: str, solution: list[str]) -> str:
    """The first solution move (UCI from Lichess) as SAN, for grounded narration. ''
    if it can't be parsed — never raise into the tool path."""
    if not solution:
        return ""
    try:
        board = chess.Board(fen)
        return board.san(chess.Move.from_uci(solution[0]))
    except Exception:  # noqa: BLE001
        return ""
