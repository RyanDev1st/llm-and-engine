from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import chess

SEED_DIR = Path(__file__).resolve().parent / "seeds"

CATEGORIES = ("opening", "middlegame", "endgame", "tactics", "terminal")

FILES = {
    "opening": "positions_openings.fen",
    "middlegame": "positions_middlegame.fen",
    "endgame": "positions_endgame.fen",
    "tactics": "positions_tactics.fen",
    "terminal": "positions_terminal.fen",
}


@dataclass(frozen=True)
class Position:
    fen: str
    category: str


class PositionBank:
    def __init__(self, by_category: dict[str, list[Position]]):
        self._by_category = by_category

    def count(self, category: str) -> int:
        return len(self._by_category.get(category, ()))

    def all(self, category: str) -> list[Position]:
        return list(self._by_category.get(category, ()))


def load_default_bank(root: Path = SEED_DIR) -> PositionBank:
    by_category: dict[str, list[Position]] = {}
    for category, filename in FILES.items():
        text = (root / filename).read_text(encoding="utf-8")
        fens = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        positions: list[Position] = []
        for fen in fens:
            chess.Board(fen)
            positions.append(Position(fen=fen, category=category))
        by_category[category] = positions
    return PositionBank(by_category)


def sample_position(bank: PositionBank, category: str, seed: int) -> Position:
    pool = bank.all(category)
    if not pool:
        raise ValueError(f"position category empty: {category}")
    rng = random.Random(seed)
    return pool[rng.randrange(len(pool))]
