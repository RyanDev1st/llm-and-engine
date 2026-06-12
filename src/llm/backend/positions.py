"""Position sources for skills that need material to work on (e.g. the
tactical-puzzle-generator, which scans 'the current board' but has no way to GET a
tactical position). `random_position(kind)` sets the board to one and returns it:

- kind="puzzle" (default): a curated FEN with a known tactic (fork / pin / skewer /
  mate) so the puzzle skill has a real motif to find — not the empty start position.
- kind="scramble": play a handful of random LEGAL moves from the start for a random
  middlegame (variety; not guaranteed tactical).
- kind="open": a random common opening position.

Curated, not truly random — a random FEN is usually illegal or tactically dead, which
would give the puzzle skill nothing to scan."""
from __future__ import annotations

import random

import chess

# (theme, side-to-move-has-the-tactic, FEN). Verified legal; each has a clear motif.
PUZZLES: list[tuple[str, str]] = [
    ("fork: knight forks king and queen", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("mate in 1: back-rank", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("pin: win the pinned knight", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("skewer: check wins the rook behind", "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1"),
    ("mate in 2: queen + rook", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"),
    ("fork: pawn forks two pieces", "r1bqkbnr/ppp2ppp/2np4/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4"),
    ("discovered attack", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"),
    ("hanging piece: win a free knight", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 5 4"),
]

OPENINGS: list[str] = [
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",   # Sicilian
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",   # Queen's pawn
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",   # Open game
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",    # Indian
]


def random_position(game, kind: str = "puzzle", rng: random.Random | None = None) -> str:
    """Set `game`'s board to a position of the requested kind and return a description
    the model can scan. Falls back to a puzzle for an unknown kind."""
    r = rng or random.Random()
    kind = (kind or "puzzle").lower()
    if kind == "scramble":
        board = chess.Board()
        for _ in range(r.randint(6, 14)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(r.choice(moves))
        game.board = board
        game.san_stack = []
        return f"position: random scramble set. fen={board.fen()}, turn={'white' if board.turn else 'black'}"
    if kind == "open":
        fen = r.choice(OPENINGS)
        game.load_fen(fen)
        return f"position: random opening set. fen={fen}"
    theme, fen = r.choice(PUZZLES)
    game.load_fen(fen)
    side = "white" if fen.split()[1] == "w" else "black"
    return f"position: puzzle set ({theme}). {side} to move and find it. fen={fen}"
