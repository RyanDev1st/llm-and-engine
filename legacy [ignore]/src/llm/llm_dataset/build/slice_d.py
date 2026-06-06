"""Author slice D (implicit eval). Spec 6.1-D.

User asks an evaluative question -> assistant emits <tool>eval depth=15</tool>,
backend returns a real Stockfish score, assistant narrates warmly (positive =
white better). ~15% mate scores, ~15% timeout/engine_unavailable stress.

All non-error scores are computed by real Stockfish so the set is replay-valid
by construction.
"""
from __future__ import annotations

import random
from typing import Callable

from llm_training.system_prompt import SYSTEM_PROMPT

USER_PHRASINGS = [
    "how's my position looking?", "who's winning here?", "am I cooked?", "is this lost for me?",
    "rate my position", "how am I doing?", "score this position for me", "what does the engine think?",
    "be honest, how bad is it?", "are we winning?", "give me the eval", "where do I stand right now?",
    "is this position any good?", "how's it going on the board?", "am I better or worse here?",
    "what's the verdict on this position?", "should I be worried?", "how do things look?",
    "evaluate this for me please", "who's got the edge?", "am I ahead?", "is it close?",
    "how's my game?", "do I have a winning position?", "tell me the score", "what's the assessment?",
    "am I crushing or getting crushed?", "how strong is my position?", "is this drawish?",
    "what's the engine eval here?", "gimme the read on this position", "how lopsided is this?",
    "honestly how am I doing here", "is my position holding up?", "what's my standing?",
]

# Candidate forced-mate positions; engine confirms mate at runtime (others
# skipped). Mix of back-rank, KR/KQ-vs-K and ladder mates for both colours.
MATE_FENS = [
    "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
    "7k/5ppp/8/8/8/8/8/5R1K w - - 0 1",
    "6k1/4Rppp/8/8/8/8/8/7K w - - 0 1",
    "k7/8/1K6/8/8/8/8/7R w - - 0 1",
    "7k/8/6K1/8/8/8/8/7R w - - 0 1",
    "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1",
    "7k/8/5QK1/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/5k2/6q1/7K b - - 0 1",
    "r6k/6PP/8/8/8/8/ppp5/K6R b - - 0 1",
    "6k1/8/6K1/8/8/8/8/r7 b - - 0 1",
    "5rk1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "1k6/8/1K6/8/8/8/8/7R w - - 0 1",
    "7k/5K2/8/8/8/8/8/7R w - - 0 1",
    "8/8/8/8/8/2k5/1q6/K7 b - - 0 1",
    "6k1/5p1p/8/8/8/8/8/R5K1 w - - 0 1",
]


def _bucket_narration(cp: int, rng: random.Random) -> str:
    if cp >= 300:
        opts = ["White is winning here - a commanding advantage on the board!",
                "This is clearly White's game now, a winning edge.",
                "White is way ahead - close to decisive."]
    elif cp >= 150:
        opts = ["White holds a clear edge here - definitely the more comfortable side.",
                "White's a fair bit better; nice pull in this position.",
                "Things favour White noticeably right now."]
    elif cp >= 50:
        opts = ["White's a touch better, but it's still very much a game.",
                "Slight edge to White - nothing decided yet.",
                "White is marginally on top here."]
    elif cp > -50:
        opts = ["It's about level - a balanced fight!",
                "Roughly equal here, anyone's game.",
                "Dead even, so it's all to play for."]
    elif cp > -150:
        opts = ["Black's a touch better, but you're well in it.",
                "Slight edge to Black - still plenty to fight for.",
                "Black is marginally ahead right now."]
    elif cp > -300:
        opts = ["Black holds a clear edge here - time to dig in.",
                "Black's a fair bit better; the position is under pressure.",
                "Things favour Black noticeably at the moment."]
    else:
        opts = ["Black is winning here - a tough spot to defend.",
                "This is clearly Black's game now; an uphill battle.",
                "Black is way ahead - close to lost, but never give up!"]
    return rng.choice(opts)


def _mate_narration(side: str, n: int, rng: random.Random) -> str:
    if side == "white":
        return rng.choice([f"Mate is on the board - White forces it in {n}! Decisive.",
                           f"White has a forced mate in {n}. Game over, basically!"])
    return rng.choice([f"Careful - Black has a forced mate in {n}. Decisive against White.",
                      f"Tough news: Black mates in {n} here."])


def _convo(uid: str, user: str, tool_result: str, narration: str) -> dict:
    return {
        "id": uid, "slice": "D",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "<tool>eval depth=15</tool>"},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": narration},
        ],
        "validated": True, "notes": "slice D implicit eval (real stockfish score)",
    }


def _error_convo(uid: str, user: str, kind: str, rng: random.Random) -> dict:
    narr = rng.choice([
        "Sorry, the engine took too long to answer - want me to try that again?",
        "Hmm, I couldn't reach the engine just then. Shall I give it another go?",
        "The engine isn't responding at the moment - happy to retry if you'd like!",
    ])
    return _convo(uid, user, f"error: {kind}", narr)


def generate(score_fn: Callable[[str], tuple], count: int = 315, seed: int = 7) -> list[dict]:
    """score_fn(fen)->(kind, value): kind in {cp,mate}; value=cp int or (side,n)."""
    rng = random.Random(seed)
    out: list[dict] = []
    n_err = round(count * 0.15)
    n_mate_target = round(count * 0.15)

    # 1) mate examples (only those the engine confirms as mate)
    mi = 0
    for fen in MATE_FENS:
        if mi >= n_mate_target:
            break
        kind, val = score_fn(fen)
        if kind != "mate":
            continue
        side, nmate = val
        tr = f"score: mate in {nmate} for {side}, depth=15"
        out.append(_convo(f"ex_D_m{mi:03d}", rng.choice(USER_PHRASINGS), tr, _mate_narration(side, nmate, rng)))
        mi += 1

    # 2) timeout / engine_unavailable stress
    for k in range(n_err):
        kind = "timeout" if k % 2 == 0 else "engine_unavailable"
        out.append(_error_convo(f"ex_D_e{k:03d}", rng.choice(USER_PHRASINGS), kind, rng))

    # 3) real-cp evals fill the remainder up to `count`
    i = 0
    guard = 0
    while len(out) < count and guard < count * 10:
        guard += 1
        kind, val = score_fn(_random_fen(rng))
        if kind != "cp":
            continue
        cp = int(val)
        tr = f"score: {cp/100:+.2f} pawns from white POV, depth=15"
        out.append(_convo(f"ex_D_{i:04d}", rng.choice(USER_PHRASINGS), tr, _bucket_narration(cp, rng)))
        i += 1
    return out


def _random_fen(rng: random.Random) -> str:
    """A non-terminal position reached by random legal play from the start."""
    import chess
    for _ in range(20):
        board = chess.Board()
        for _ in range(rng.randint(2, 36)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and any(board.legal_moves):
            return board.fen()
    return chess.Board().fen()
