"""Real position source: fetch a rated tactical puzzle from Lichess (public API, no
key) and set it on the board. This is the "search a real FEN board online" capability —
unlike the local curated bank, the FEN, themes, and rating are real and verifiable.
The setup result does not expose the solution; the coach calls best_move later to reveal it.

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
    side-to-move is the solver. Falls back to the local curated bank if offline."""
    data = _fetch_puzzle_json(timeout)
    if not data:
        from .positions import random_position
        return "note: online puzzle source unavailable, using a local puzzle. " \
               + random_position(game, "puzzle")
    pz = data.get("puzzle", {}) or {}
    fen = _puzzle_fen(data)
    if not fen or not game.load_fen(fen):
        from .positions import random_position
        return "note: online puzzle FEN unusable, using a local puzzle. " \
               + random_position(game, "puzzle")
    side = "white" if game.board.turn == chess.WHITE else "black"
    themes = ", ".join(pz.get("themes", []) or []) or "tactics"
    rating = pz.get("rating", "?")
    pid = pz.get("id", "?")
    return (f"position: lichess puzzle {pid} (rating {rating}, themes: {themes}). "
            f"{side} to move and find the tactic. fen={fen}")


def _puzzle_fen(data: dict) -> str:
    """The puzzle's starting FEN. /daily ships `puzzle.fen` directly; /next ships only
    the game PGN + initialPly, so replay the PGN's SAN moves to that ply to reconstruct
    the position (where the solver is to move). '' if neither path yields a FEN."""
    pz = data.get("puzzle", {}) or {}
    fen = pz.get("fen", "")
    if fen:
        return fen
    pgn = (data.get("game", {}) or {}).get("pgn", "")
    ply = pz.get("initialPly")
    if not pgn or ply is None:
        return ""
    try:
        board = chess.Board()
        for i, san in enumerate(pgn.split()):   # Lichess game.pgn = space-separated SAN
            board.push_san(san)
            if i == ply:
                return board.fen()
    except Exception:  # noqa: BLE001 — malformed PGN -> let the caller fall back
        return ""
    return ""


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
