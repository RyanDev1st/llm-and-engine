"""Coaching analysis tools that compose the board + engine but are kept OUT of
tools.py (already at the file-size cap). The dispatcher (tools._dispatch) routes
to these by name.

what_if is the one capability the existing surface lacks: eval (current position),
best_move (the engine's pick), and review_move (the LAST move played) never weigh a
move the user is *considering*. A coach constantly needs "should I play X?" /
"is X or Y better?" — and answering it by play→eval→undo mutates the live board and
costs three steps. what_if does it read-only on a copy and compares to the engine's
best, so the model can GROUND a move-comparison ("why is my idea worse") instead of
asserting one — directly serving the grounded-'why' the corpus must teach."""
from __future__ import annotations

import chess

from .toolfmt import clamp_depth

# cp loss (vs the engine's best, from the mover's POV) -> coach label. Mirrors
# tools.LABELS but phrased for a move the user is *weighing*, not one already played.
_VERDICT = (
    (15, "the engine's top choice"),
    (50, "about as good as the best move"),
    (120, "a slight inaccuracy"),
    (250, "a clear mistake"),
)


def _verdict(loss_cp: int) -> str:
    for cap, text in _VERDICT:
        if loss_cp <= cap:
            return text
    return "a blunder"


def _fmt(score: chess.engine.Score) -> str:
    """White-POV display, matching eval/best_move language."""
    if score.is_mate():
        m = score.mate()
        return f"M{abs(m)} for {'white' if m > 0 else 'black'}"
    return f"{score.score() / 100:+.2f}"


def what_if(game, engine, args: dict[str, str]) -> str:
    """Score the position AFTER a candidate SAN move and compare it to the engine's
    best move from the current position. Read-only (works on a board copy); white-POV
    numbers so the wording matches eval/best_move."""
    san = (args.get("san") or "").strip()
    if not san:
        return ("error: what_if needs the move you're weighing — "
                "e.g. <tool>what_if san=Nf3</tool>")
    board = game.board
    if board.is_game_over():
        return "what_if: the game is already over — there's no move to weigh."
    try:
        mv = board.parse_san(san)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        return f"error: '{san}' isn't a legal move in this position."
    depth = clamp_depth(args, 18)
    mover = board.turn
    info = engine.analyse(board, depth)
    pv = info.get("pv", [])
    best_san = board.san(pv[0]) if pv else "?"
    before = info["score"].pov(mover).score(mate_score=100000)
    w_best = info["score"].white()
    clean = board.san(mv)
    nb = board.copy()
    nb.push(mv)
    nb_info = engine.analyse(nb, depth)
    after = nb_info["score"].pov(mover).score(mate_score=100000)
    w_cand = nb_info["score"].white()
    loss = max(0, before - after)
    if clean == best_san or loss <= 15:
        return f"what_if: {clean} is {_verdict(loss)} here, reaching {_fmt(w_cand)} (white POV)."
    return (f"what_if: {clean} -> {_fmt(w_cand)} (white POV); best is {best_san} -> {_fmt(w_best)}; "
            f"{clean} gives up {loss / 100:.2f} vs best ({_verdict(loss)}).")
