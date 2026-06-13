"""Topic-keyed chess knowledge for slices I and K, so the user's QUESTION drives
the answer (and, for I, the ask_chessbot query + tool result) instead of a single
hardcoded reply. Before this, every I question got the Sicilian answer and every K
question got the knight-value answer — the model would learn confidently wrong
answers. Finals are STATEMENTS (no trailing question) to match the slice contract.

I = looked up via the ask_chessbot KB tool (grounded in a tool result).
K = answered directly from general principle (no tool; no fabricated cp number)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KBItem:
    prompts: tuple[str, ...]
    query: str    # ask_chessbot arg for slice I; "" for K (no tool)
    result: str   # tool result for slice I; "" for K
    answer: str    # final reply (statement, no trailing '?')


KB: dict[str, tuple[KBItem, ...]] = {
    "I": (
        KBItem(("what is the sicilian?", "explain the sicilian", "tell me about the sicilian defense"),
               "sicilian",
               "Sicilian: Black meets 1.e4 with 1...c5, fighting for the d4 square asymmetrically.",
               "It's Black's sharpest reply to 1.e4 — striking at the centre from the side instead of meeting it head-on."),
        KBItem(("what is a fork?", "explain a fork", "what does forking mean"),
               "fork",
               "Fork: one piece attacks two or more enemy targets at once; the knight is the classic forker.",
               "A fork is a single piece attacking two targets at once — the knight is the classic forker because it hits squares nothing else covers."),
        KBItem(("why castle?", "what is castling?", "should I castle"),
               "castling",
               "Castling: a king-safety move that tucks the king toward a corner and connects the rooks; never through or into check.",
               "Castling gets your king to safety and links your rooks in one move — just remember you can't castle through or into check."),
        KBItem(("who is capablanca?", "tell me about capablanca", "what is capablanca known for"),
               "capablanca",
               "Capablanca: Cuban world champion 1921-27, renowned for clean, almost effortless endgame technique.",
               "Capablanca was world champion in the 1920s, famous for endgames so clean they looked effortless."),
    ),
    "K": (
        KBItem(("how much is a knight worth?", "what's a knight worth", "value of a knight"),
               "", "",
               "A knight is worth about three pawns, the same as a bishop — though its real value depends on how active it is, not the raw count."),
        KBItem(("is the queen the strongest piece?", "what's the strongest piece", "how strong is the queen"),
               "", "",
               "Yes — the queen is the most powerful piece, moving like a rook and bishop combined; only the king matters more, since losing it ends the game."),
        KBItem(("how many points is a rook?", "what's a rook worth", "value of a rook"),
               "", "",
               "A rook is worth about five pawns — stronger than a knight or bishop, which is why trading a rook for a minor piece is usually a bad deal."),
        KBItem(("what is a passed pawn?", "explain a passed pawn", "why are passed pawns good"),
               "", "",
               "A passed pawn has no enemy pawns blocking or guarding its path to promotion, so it ties down the defender and gets dangerous in the endgame."),
    ),
}


def pick_kb(slice_name: str, seed: int) -> KBItem:
    items = KB[slice_name]
    return items[random.Random(seed * 73 + 5).randrange(len(items))]
