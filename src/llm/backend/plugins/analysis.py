"""analysis plugin: post-game review. Replays the game and uses Stockfish to score
each move — real analysis (Chess.com/Lichess 'game review' style). Tests routing:
"how accurate was I?" / "where did I blunder?" must go here, NOT to eval (which only
reads the CURRENT position) or review_move (only the LAST move)."""
from __future__ import annotations

import chess

NAME = "analysis"

TOOLS = [
    {"name": "accuracy_report", "description": "Score how accurately the whole game was played so far (per-side accuracy).",
     "args": {"depth": "required"}, "applies_when": "has_history"},
    {"name": "find_blunders", "description": "List the blunders made so far in the game with the better move.",
     "args": {"depth": "required"}, "applies_when": "has_history"},
]

SKILLS = [{
    "name": "game-reviewer",
    "description": "Use when the user asks how they played overall, their accuracy, or to find blunders across the game.",
    "body": ("---\nname: game-reviewer\ndescription: Whole-game review.\n---\n\n"
             "# game-reviewer\n\nFor 'how did I play', 'my accuracy', 'where did I go wrong': call "
             "`accuracy_report` for the per-side accuracy, and `find_blunders` to point out the worst "
             "moves with the better option. Summarise plainly; only state numbers the tools returned."),
}]

_BLUNDER_CP = 200   # an eval drop of ≥2 pawns (vs the best move) is a blunder


def _per_move(executor, depth: int):
    """Yield (ply_index, mover, played_san, cp_loss, best_san) by replaying the game and
    comparing each move to the engine's best at that position. cp_loss is from the mover's
    POV (positive = how much worse than best)."""
    sans = list(executor.game.san_stack)
    if not sans:
        return
    board = chess.Board()
    for i, san in enumerate(sans):
        try:
            mv = board.parse_san(san)
        except ValueError:
            break
        info = executor.engine.analyse(board, depth)
        best = info.get("pv", [None])[0]
        best_san = board.san(best) if best else "?"
        before = info["score"].pov(board.turn).score(mate_score=100000)
        mover = "white" if board.turn == chess.WHITE else "black"
        board.push(mv)
        after = executor.engine.analyse(board, depth)["score"].pov(not board.turn).score(mate_score=100000)
        yield i, mover, san, max(0, before - after), best_san


def _accuracy(cp_loss_list: list[int]) -> int:
    """A simple, monotone accuracy %: 100 at 0 avg loss, decaying with average cp loss."""
    if not cp_loss_list:
        return 100
    avg = sum(cp_loss_list) / len(cp_loss_list)
    return max(0, round(100 - avg / 5))   # ~20cp avg loss -> 96%, 100cp -> 80%


def handle(name: str, args: dict, executor) -> str | None:
    if name not in ("accuracy_report", "find_blunders"):
        return None
    if not executor.game.san_stack:
        return "error: no moves to review"
    try:
        depth = max(8, min(16, int(args.get("depth", 12))))
    except ValueError:
        depth = 12
    rows = list(_per_move(executor, depth))
    if not rows:
        return "error: no moves to review"
    if name == "accuracy_report":
        w = [c for _, m, _, c, _ in rows if m == "white"]
        b = [c for _, m, _, c, _ in rows if m == "black"]
        return f"accuracy: white={_accuracy(w)}%, black={_accuracy(b)}%, moves={len(rows)}"
    blunders = [(i, m, s, c, bs) for i, m, s, c, bs in rows if c >= _BLUNDER_CP]
    if not blunders:
        return "blunders: none — no move lost ≥2 pawns vs the engine's best."
    parts = [f"move {i+1} ({m}) {s} lost {c/100:.1f} (better: {bs})" for i, m, s, c, bs in blunders[:5]]
    return "blunders: " + "; ".join(parts)
