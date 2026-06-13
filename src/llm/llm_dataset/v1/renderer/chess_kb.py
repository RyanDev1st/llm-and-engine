"""Topic-keyed chess knowledge for slices I and K, so the user's QUESTION drives
the answer (and, for I, the ask_chessbot query + tool result) instead of a single
hardcoded reply. Before this, every I question got the Sicilian answer and every K
question got the knight-value answer — the model would learn confidently wrong
answers. Finals are STATEMENTS (no trailing question) to match the slice contract.

Each topic carries SEVERAL answer paraphrases (same fact, different wording),
picked by seed, so distinct finals scale past the topic count without the model
memorising one sentence per topic — while every paraphrase stays on-topic.

I = looked up via the ask_chessbot KB tool (grounded in a tool result).
K = answered directly from general principle (no tool; no fabricated cp number)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KBItem:
    prompts: tuple[str, ...]
    query: str             # ask_chessbot arg for slice I; "" for K (no tool)
    result: str            # tool result for slice I; "" for K
    answers: tuple[str, ...]  # final-reply paraphrases (statements, no trailing '?')


KB: dict[str, tuple[KBItem, ...]] = {
    "I": (
        KBItem(("what is the sicilian?", "explain the sicilian", "tell me about the sicilian defense"),
               "sicilian",
               "Sicilian: Black meets 1.e4 with 1...c5, fighting for the d4 square asymmetrically.",
               ("It's Black's sharpest reply to 1.e4 — striking at the centre from the side instead of meeting it head-on.",
                "The Sicilian answers 1.e4 with 1...c5, fighting for the centre asymmetrically rather than mirroring White.",
                "It's the 1...c5 defence: Black contests d4 from the wing, which is why it leads to such sharp, unbalanced games.")),
        KBItem(("what is a fork?", "explain a fork", "what does forking mean"),
               "fork",
               "Fork: one piece attacks two or more enemy targets at once; the knight is the classic forker.",
               ("A fork is a single piece attacking two targets at once — the knight is the classic forker because it hits squares nothing else covers.",
                "It's one piece hitting two or more enemy pieces simultaneously; knights fork best since their jump can't be blocked.",
                "Forking means attacking two things with one move, so the opponent can only save one — knights are the textbook example.")),
        KBItem(("why castle?", "what is castling?", "should I castle"),
               "castling",
               "Castling: a king-safety move that tucks the king toward a corner and connects the rooks; never through or into check.",
               ("Castling gets your king to safety and links your rooks in one move — just remember you can't castle through or into check.",
                "It's the one move that both shelters your king in a corner and connects the rooks; it's illegal through or into check.",
                "You castle to tuck the king away and activate a rook at once — but not while in check, through check, or into it.")),
        KBItem(("who is capablanca?", "tell me about capablanca", "what is capablanca known for"),
               "capablanca",
               "Capablanca: Cuban world champion 1921-27, renowned for clean, almost effortless endgame technique.",
               ("Capablanca was world champion in the 1920s, famous for endgames so clean they looked effortless.",
                "He was the Cuban world champion of the early 1920s, legendary for simple, precise endgame technique.",
                "Capablanca held the title from 1921-27 and is remembered for an almost machine-like clarity in the endgame.")),
    ),
    "K": (
        KBItem(("how much is a knight worth?", "what's a knight worth", "value of a knight"),
               "", "",
               ("A knight is worth about three pawns, the same as a bishop — though its real value depends on how active it is, not the raw count.",
                "Roughly three points, level with a bishop; but a knight in a strong outpost is worth more than the number suggests.",
                "About three pawns — equal to a bishop on paper, though activity and pawn structure swing its true worth.")),
        KBItem(("is the queen the strongest piece?", "what's the strongest piece", "how strong is the queen"),
               "", "",
               ("Yes — the queen is the most powerful piece, moving like a rook and bishop combined; only the king matters more, since losing it ends the game.",
                "The queen is the strongest, since it moves as a rook and bishop together; the king is more important only because losing it loses the game.",
                "By power, the queen tops every piece — rook plus bishop in one; the king is merely more valuable because it can't be lost.")),
        KBItem(("how many points is a rook?", "what's a rook worth", "value of a rook"),
               "", "",
               ("A rook is worth about five pawns — stronger than a knight or bishop, which is why trading a rook for a minor piece is usually a bad deal.",
                "Around five points, more than a minor piece; giving up a rook for a knight or bishop (losing the exchange) usually hurts.",
                "Roughly five pawns — clearly above a knight or bishop, so swapping one for a minor is normally a concession.")),
        KBItem(("what is a passed pawn?", "explain a passed pawn", "why are passed pawns good"),
               "", "",
               ("A passed pawn has no enemy pawns blocking or guarding its path to promotion, so it ties down the defender and gets dangerous in the endgame.",
                "It's a pawn with a clear lane to promotion — no enemy pawns ahead on its file or the ones beside it — which makes it a real endgame weapon.",
                "A passed pawn faces no opposing pawns on its way to queening, so it forces the defender to spend pieces stopping it.")),
    ),
}


def pick_kb(slice_name: str, seed: int) -> KBItem:
    items = KB[slice_name]
    return items[random.Random(seed * 73 + 5).randrange(len(items))]


def pick_answer(item: KBItem, seed: int) -> str:
    """One of the topic's answer paraphrases, seeded — same fact, varied wording."""
    return random.Random(seed * 79 + 13).choice(item.answers)
