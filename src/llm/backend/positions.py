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

# (theme, FEN). Side-to-move has the tactic. Each FEN + theme was VERIFIED with our own
# Stockfish (depth 14): the labelled motif IS the engine's top line, decisive (mate or
# >=+1.9). The old bank had 5/8 mislabeled/dead positions that the model then narrated as
# fabricated tactics — replaced. The setup result intentionally does NOT expose the answer:
# the model should call best_move later when the user asks for a hint/solution.
PUZZLES: list[tuple[str, str]] = [
    ("mate in 1 (back-rank): the rook delivers mate", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("mate in 1 (back-rank): the rook delivers mate", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"),
    ("white to move and win a whole rook", "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1"),
    ("mate in 1 for Black (back-rank): the rook delivers mate", "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"),
    ("mate in 1 for Black (fool's mate): the queen ends it", "rnbqkbnr/ppp2ppp/8/3pp3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3"),
    ("black to move and win material in the centre", "r1bqkbnr/pppp1ppp/2n5/1B2N3/4P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 3"),
    ("mate in 2: two-rook ladder", "7k/8/8/8/8/8/R7/1R5K w - - 0 1"),
]

OPENINGS: list[str] = [
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",   # Sicilian
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",   # Queen's pawn
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",   # Open game
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",    # Indian
]


def random_position(game, kind: str = "puzzle", rng: random.Random | None = None,
                    engine=None) -> str:
    """Set `game`'s board to a position of the requested kind and return a description
    the model can scan. Falls back to a puzzle for an unknown kind. When `engine` is
    given (the live Stockfish), it is ignored for setup: hiding the solution prevents the
    first puzzle-presenting reply from leaking the answer."""
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
