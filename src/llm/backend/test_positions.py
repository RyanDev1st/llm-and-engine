"""random_position tool: supplies a fresh position (tactical puzzle / scramble / open)
so skills like the tactical-puzzle-generator have material to work on. Every curated
FEN must be legal; the tool must actually set the board."""
import random

import chess

from backend.positions import PUZZLES, OPENINGS, random_position
from backend.game import Game
from backend.tools import ToolExecutor
from backend.tool_hints import matched_tools


def test_all_curated_fens_are_legal():
    for theme, fen in PUZZLES:
        assert chess.Board(fen).is_valid(), theme
    for fen in OPENINGS:
        assert chess.Board(fen).is_valid(), fen


def test_random_position_sets_the_board():
    g = Game()
    out = random_position(g, "puzzle", random.Random(1))
    assert out.startswith("position: puzzle set") and "fen=" in out
    assert g.board.fen() != chess.STARTING_FEN            # board actually changed
    # scramble produces a legal, non-start board
    g2 = Game()
    random_position(g2, "scramble", random.Random(2))
    assert g2.board.is_valid() and g2.board.fen() != chess.STARTING_FEN
    # unknown kind falls back to a puzzle
    g3 = Game()
    assert "puzzle set" in random_position(g3, "wat", random.Random(3))


def test_executor_dispatches_random_position():
    g = Game()
    out = ToolExecutor(g, None).execute("<tool>random_position kind=puzzle</tool>")
    assert out.startswith("position:") and g.board.fen() != chess.STARTING_FEN


def test_puzzle_intent_routes_to_random_position():
    for msg in ["give me a puzzle", "make me a tactical puzzle", "scramble the board",
                "set up a random position", "generate a puzzle",
                "give me a chess puzzle", "puzzle me", "i want a new puzzle"]:
        assert "random_position" in matched_tools(msg), msg


def test_random_position_kind_follows_words():
    from backend.tool_hints import matched_calls
    assert matched_calls("give me a chess puzzle")["random_position"] == \
        "<tool>random_position kind=puzzle</tool>"
    assert matched_calls("randomize the fen")["random_position"] == \
        "<tool>random_position kind=scramble</tool>"
    assert matched_calls("scramble the board")["random_position"] == \
        "<tool>random_position kind=scramble</tool>"
    assert matched_calls("set up a random opening")["random_position"] == \
        "<tool>random_position kind=open</tool>"


def test_real_puzzle_routes_to_fetch_puzzle_not_local():
    from backend.tool_hints import matched_tools as mt
    for msg in ["give me a real puzzle", "a puzzle from lichess", "fetch an online puzzle",
                "today's daily puzzle", "get a rated puzzle"]:
        t = mt(msg)
        assert "fetch_puzzle" in t, msg
        assert "random_position" not in t, msg   # mutually exclusive — don't do both
