from __future__ import annotations

from typing import Any

from ..annotator import AnnotatedPosition, StockfishAnnotator
from ..board_facts import (board_state_line, choose_move, king_moves, legal_moves_for_square,
                           list_pieces_text, move_echo)
from ..sampler import Scenario
from . import tone
from .chess_kb import KBItem, pick_answer, pick_kb
from .chess_envelope import build_chess_envelope
from .finals import e_top_form, final_narration, wants_number
from .review import ReviewFacts, delta_str, review_for_played
from .tags import skill_call_msg, tool_call_msg, tool_result_msg
from .text import best_move_score, score_pawns, score_text
from .thinking import pick_mode

SLICE_USER_TEMPLATES = {
    "A": ("play {san}", "let's go {san}", "{san} for me", "push {san}"),
    "B": ("should I move the knight or bishop?", "what plan should I choose?", "which capture is best?", "help me decide"),
    "C": ("can my king move to e2?", "is it legal for my king to step to e2?",
          "can the king go to e2?", "where can my king move?"),
    "D": ("who is winning?", "rate this position", "is this lost for me?", "how is it?",
          "what's the exact eval?", "give me the score in pawns", "how many pawns am I up?",
          "what's the centipawn eval?"),
    "E": ("what should I play?", "best move?", "give me the line", "what's my best option here?",
          "best move and the eval?"),
    "F": ("how was that move?", "did I blunder?", "rate my last move", "was that ok?"),
    "G": ("any threats?", "what is the opponent up to?", "watch out for what?"),
    "H": ("what pieces are left?", "what do I have on the board?", "list my material",
          "which pieces do I still have?"),
    "I": ("what is the sicilian?", "why castle?", "what is a fork?", "who is capablanca?"),
    "J": ("hey there", "thanks!", "what can you do?", "feeling good"),
    "K": ("how much is a knight worth?", "is the queen the strongest piece?", "checkmate, that's a deal"),
}

INTERNAL_LESSON = "Use board tools before claims. Ground evaluation in Stockfish output."


def _best_moves_result(top_moves: tuple) -> str:
    # mirrors the live backend best_moves format exactly (tools._best_move)
    return "best_moves: " + "; ".join(f"{i}. {san} ({cp / 100:+.2f})" for i, (san, cp) in enumerate(top_moves, 1))


def _emit_best_move(messages: list[dict[str, Any]], scenario: Scenario, annotated: AnnotatedPosition) -> None:
    """Emit the engine best_move call+result (top-N or best_line form). Shared by slice E and now
    slice B (survey legal moves -> ask the engine -> grounded recommendation)."""
    if e_top_form(scenario, annotated):
        messages.append(tool_call_msg("best_move", {"depth": 15, "top": 3}))
        res = _best_moves_result(annotated.top_moves)
        if annotated.score_kind == "mate":   # ground the 'mate in N' final (top form carries no score field)
            res += f", score: {best_move_score(annotated)}"
        messages.append(tool_result_msg("best_move", res))
    else:
        messages.append(tool_call_msg("best_move", {"depth": 15, "series": 3}))
        line = " ".join(annotated.best_line_sans)
        messages.append(tool_result_msg("best_move", f"best_line: {line}, score: {best_move_score(annotated)}"))


def _played_move(annotated: AnnotatedPosition | None, slice_name: str, seed: int) -> str | None:
    """The move to execute. F alternates the engine's best (-> a grounded 'good move'
    review) and an arbitrary legal move (-> a grounded critique); A just plays one."""
    if not annotated or slice_name not in {"A", "F"}:
        return None
    if slice_name == "F" and seed % 2 == 0 and annotated.best_san:
        return annotated.best_san
    return choose_move(annotated.fen, seed)


def render_chess_row(scenario: Scenario, annotator: StockfishAnnotator) -> dict[str, Any]:
    annotated = annotator.annotate(scenario.position.fen, depth=12) if scenario.position else None
    seed = scenario.seed
    # A legal move to execute for the move-playing slices (A plays a requested move, F
    # plays then reviews). See _played_move: F alternates a genuinely good move and a weak
    # one so the review teaches BOTH praise and correction, not a rubber-stamp.
    move = _played_move(annotated, scenario.slice, seed)
    # Measure that move honestly (real label + centipawn loss) for the F review — replaces
    # the old hardcoded "label=good, delta=+0.05" that fabricated every review.
    review = review_for_played(annotator, annotated, move) if (annotated and scenario.slice == "F" and move) else None
    # Knowledge slices (I/K) are topic-keyed so the user's QUESTION drives the
    # answer (and I's ask_chessbot query+result), not a single hardcoded reply.
    kb = pick_kb(scenario.slice, seed) if scenario.slice in ("I", "K") else None
    user = _style_prompt(tone.pick(seed, kb.prompts), scenario) if kb else _user_message(scenario, move)
    mode = pick_mode(seed)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    if scenario.slice != "J":
        _emit_skill_load(messages)
    if scenario.slice == "F" and annotated:
        messages.append(tool_call_msg("move", {"san": move}))
        messages.append(tool_result_msg("move", move_echo(annotated.fen, move)))
    if scenario.slice in {"A", "B", "C", "D", "E", "F", "G", "H"}:
        messages.append(tool_call_msg("board_state", {"fields": "basic"}))
        messages.append(tool_result_msg("board_state", _board_state_text(annotated)))
    _emit_slice_tool(messages, scenario, annotated, move, kb, review)
    body = final_narration(scenario, annotated, move, wants_number(user),
                           pick_answer(kb, seed) if kb else None, review=review)
    messages.append({"role": "assistant", "content": body})
    return build_chess_envelope(scenario, messages, annotated, mode)


def _user_message(scenario: Scenario, move: str | None) -> str:
    templates = SLICE_USER_TEMPLATES.get(scenario.slice, ("explain the position",))
    base = tone.pick(scenario.seed, templates)
    if "{san}" in base:
        base = base.replace("{san}", move or "e4")
    elif "{square}" in base:
        base = base.replace("{square}", "e2")
    return _style_prompt(base, scenario)


def _style_prompt(base: str, scenario: Scenario) -> str:
    if scenario.prompt_style == "formal":
        return f"Please {base}."
    if scenario.prompt_style == "casual":
        return base
    if scenario.prompt_style == "slang":
        return f"yo, {base}"
    if scenario.prompt_style == "typo":
        return f"{base} pls"
    if scenario.prompt_style == "anxious":
        return f"I'm worried here - {base}"
    return f"I'm new to chess; {base}"


def _emit_skill_load(messages: list[dict[str, Any]]) -> None:
    messages.append(skill_call_msg("chess-coach"))
    messages.append(tool_result_msg("load_skill", INTERNAL_LESSON))


def _board_state_text(annotated: AnnotatedPosition | None) -> str:
    if annotated is None:
        return "board_state: turn=white, last_move=none, check=no, legal_count=20"
    return board_state_line(annotated.fen)


def _emit_slice_tool(
    messages: list[dict[str, Any]], scenario: Scenario, annotated: AnnotatedPosition | None,
    move: str | None, kb: KBItem | None = None, review: ReviewFacts | None = None,
) -> None:
    if scenario.slice == "A" and annotated:
        messages.append(tool_call_msg("move", {"san": move}))
        messages.append(tool_result_msg("move", move_echo(annotated.fen, move)))
    elif scenario.slice == "B" and annotated:
        # Survey the legal options FIRST, then ask the engine — so the final is a grounded
        # recommendation (was: legal_moves only -> the finals just narrated "I checked the moves").
        sq, sans = legal_moves_for_square(annotated.fen, scenario.seed)
        messages.append(tool_call_msg("legal_moves", {"square": sq}))
        messages.append(tool_result_msg("legal_moves", f"legal: [{', '.join(sans)}]"))
        _emit_best_move(messages, scenario, annotated)
    elif scenario.slice == "C" and annotated:
        # Legality check: read the king's actual legal squares, so the final ANSWERS the
        # "can my king move to X?" question instead of refusing to guess.
        ksq, ksans = king_moves(annotated.fen)
        messages.append(tool_call_msg("legal_moves", {"square": ksq}))
        messages.append(tool_result_msg("legal_moves",
            f"legal: [{', '.join(ksans)}]" if ksans else "legal: none (the king has no legal moves)"))
    elif scenario.slice == "D" and annotated:
        messages.append(tool_call_msg("eval", {"depth": 15}))
        messages.append(tool_result_msg("eval", score_text(annotated)))
    elif scenario.slice == "E" and annotated:
        _emit_best_move(messages, scenario, annotated)
    elif scenario.slice == "F" and annotated and review:
        messages.append(tool_call_msg("review_move", {"depth": 12}))
        # REAL review: measured label + centipawn swing, not a hardcoded "good, +0.05".
        messages.append(tool_result_msg("review_move", f"review: {review.played}, label={review.label}, delta={delta_str(review)}, best_was={review.best}"))
    elif scenario.slice == "G" and annotated:
        threat = annotated.threats_san or "none"
        messages.append(tool_call_msg("threats", {"depth": 12}))
        messages.append(tool_result_msg("threats", f"threats: opponent's best is {threat}, score for them: {score_pawns(annotated)}"))
    elif scenario.slice == "H" and annotated:
        messages.append(tool_call_msg("list_pieces", {"color": "mine"}))
        messages.append(tool_result_msg("list_pieces", list_pieces_text(annotated.fen)))
    elif scenario.slice == "I" and kb:
        messages.append(tool_call_msg("ask_chessbot", {"query": kb.query}))
        messages.append(tool_result_msg("ask_chessbot", kb.result))

