from __future__ import annotations

import re
from typing import Any

from ..annotator import AnnotatedPosition, StockfishAnnotator
from ..board_facts import board_state_line, choose_move, legal_moves_for_square, move_echo
from ..sampler import Scenario
from . import tone
from .finals import e_top_form, final_narration, wants_number
from .leadins import lead
from .text import score_pawns, score_text
from .thinking import think, think_answer

_TOOL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")

# Short, generic GOAL per slice for the <think> reasoning (intent, never facts).
SLICE_GOAL = {
    "A": "play the move they named", "B": "decide between the options",
    "C": "handle that move request", "D": "judge who stands better",
    "E": "find the best move", "F": "review the move just played",
    "G": "check the opponent's threats", "H": "read the board for them",
    "I": "explain that chess idea", "J": "chat with them",
    "K": "answer a general chess question",
}


def _step(seed: int, action: str, i: int, call: str, goal: str, have: str) -> str:
    """One assistant tool-step: <think> (decide) + lead-in + the tool call."""
    return f"{think(seed, action, i, goal=goal, have=have)}\n{lead(seed, action, i)}\n{call}"

SLICE_USER_TEMPLATES = {
    "A": ("play {san}", "let's go {san}", "{san} for me", "push {san}"),
    "B": ("should I move the knight or bishop?", "what plan should I choose?", "which capture is best?", "help me decide"),
    "C": ("play e5 for me", "can my king move to e2?", "castle through check", "make the illegal capture"),
    "D": ("who is winning?", "rate this position", "is this lost for me?", "how is it?",
          "what's the exact eval?", "give me the score in pawns", "how many pawns am I up?",
          "what's the centipawn eval?"),
    "E": ("what should I play?", "best move?", "give me the line", "show me a plan",
          "best move and the eval?"),
    "F": ("how was that move?", "did I blunder?", "rate my last move", "was that ok?"),
    "G": ("any threats?", "what is the opponent up to?", "watch out for what?"),
    "H": ("legal moves on {square}?", "undo that", "what pieces are left?"),
    "I": ("what is the sicilian?", "why castle?", "what is a fork?", "who is capablanca?"),
    "J": ("hey there", "thanks!", "what can you do?", "feeling good"),
    "K": ("how much is a knight worth?", "is the queen the strongest piece?", "checkmate, that's a deal"),
}

INTERNAL_LESSON = "Use board tools before claims. Ground evaluation in Stockfish output."


def _best_moves_result(top_moves: tuple) -> str:
    # mirrors the live backend best_moves format exactly (tools._best_move)
    return "best_moves: " + "; ".join(f"{i}. {san} ({cp / 100:+.2f})" for i, (san, cp) in enumerate(top_moves, 1))


def render_chess_row(scenario: Scenario, annotator: StockfishAnnotator) -> dict[str, Any]:
    annotated = annotator.annotate(scenario.position.fen, depth=12) if scenario.position else None
    # A legal move to execute for the move-playing slices (A plays a requested
    # move, F plays then reviews). Chosen from the real position so it is legal.
    move = choose_move(annotated.fen, scenario.seed) if (annotated and scenario.slice in {"A", "F"}) else None
    user = _user_message(scenario, move)
    goal = SLICE_GOAL.get(scenario.slice, "help with the position")
    seed = scenario.seed
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]
    _emit_skill_load(messages, scenario, goal)
    if scenario.slice == "F" and annotated:
        messages.append({"role": "assistant",
                         "content": _step(seed, "move", 1, f"<tool>move san={move}</tool>", goal, "skill")})
        messages.append({"role": "tool", "content": move_echo(annotated.fen, move)})
    if scenario.slice in {"A", "B", "C", "D", "E", "F", "G", "H"}:
        messages.append({"role": "assistant",
                         "content": _step(seed, "board_state", 2, "<tool>board_state fields=basic</tool>", goal, "skill")})
        messages.append({"role": "tool", "content": _board_state_text(annotated)})
    _emit_slice_tool(messages, scenario, annotated, move, goal)
    messages.append({"role": "assistant",
                     "content": f"{think_answer(seed, goal)}\n"
                                + final_narration(scenario, annotated, move, wants_number(user))})
    return _envelope(scenario, messages, annotated)


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


def _emit_skill_load(messages: list[dict[str, str]], scenario: Scenario, goal: str = "") -> None:
    messages.append({"role": "assistant",
                     "content": _step(scenario.seed, "load_skill", 0,
                                      "<tool>load_skill name=chess-coach</tool>", goal, "")})
    messages.append({"role": "tool", "content": INTERNAL_LESSON})


def _board_state_text(annotated: AnnotatedPosition | None) -> str:
    if annotated is None:
        return "board_state: turn=white, last_move=none, check=no, legal_count=20"
    return board_state_line(annotated.fen)


def _emit_slice_tool(
    messages: list[dict[str, str]], scenario: Scenario, annotated: AnnotatedPosition | None,
    move: str | None, goal: str = ""
) -> None:
    seed = scenario.seed
    if scenario.slice == "A" and annotated:
        messages.append({"role": "assistant", "content": _step(seed, "move", 3, f"<tool>move san={move}</tool>", goal, "board")})
        messages.append({"role": "tool", "content": move_echo(annotated.fen, move)})
    elif scenario.slice == "B" and annotated:
        sq, sans = legal_moves_for_square(annotated.fen, scenario.seed)
        messages.append({"role": "assistant", "content": _step(seed, "legal_moves", 3, f"<tool>legal_moves square={sq}</tool>", goal, "board")})
        messages.append({"role": "tool", "content": f"legal: [{', '.join(sans)}]"})
    elif scenario.slice == "D" and annotated:
        messages.append({"role": "assistant", "content": _step(seed, "eval", 3, "<tool>eval depth=15</tool>", goal, "board")})
        messages.append({"role": "tool", "content": score_text(annotated)})
    elif scenario.slice == "E" and annotated:
        if e_top_form(scenario, annotated):
            messages.append({"role": "assistant", "content": _step(seed, "best_move", 3, "<tool>best_move depth=15 top=3</tool>", goal, "board")})
            messages.append({"role": "tool", "content": _best_moves_result(annotated.top_moves)})
        else:
            messages.append({"role": "assistant", "content": _step(seed, "best_move", 3, "<tool>best_move depth=15 series=3</tool>", goal, "board")})
            line = " ".join(annotated.best_line_sans)
            messages.append({"role": "tool", "content": f"best_line: {line}, score: {score_pawns(annotated)}"})
    elif scenario.slice == "F" and annotated:
        messages.append({"role": "assistant", "content": _step(seed, "review_move", 3, "<tool>review_move depth=12</tool>", goal, "board")})
        messages.append({"role": "tool", "content": f"review: {move}, label=good, delta=+0.05 pawns, best_was={annotated.best_san}"})
    elif scenario.slice == "G" and annotated:
        threat = annotated.threats_san or "none"
        messages.append({"role": "assistant", "content": _step(seed, "threats", 3, "<tool>threats depth=12</tool>", goal, "board")})
        messages.append({"role": "tool", "content": f"threats: opponent's best is {threat}, score for them: {score_pawns(annotated)}"})
    elif scenario.slice == "H" and annotated:
        messages.append({"role": "assistant", "content": _step(seed, "list_pieces", 3, "<tool>list_pieces color=mine</tool>", goal, "board")})
        messages.append({"role": "tool", "content": _list_pieces_text(annotated.fen)})
    elif scenario.slice == "I":
        messages.append({"role": "assistant", "content": _step(seed, "ask_chessbot", 3, "<tool>ask_chessbot query=sicilian</tool>", goal, "skill")})
        messages.append({"role": "tool", "content": "Sicilian: Black answers 1.e4 with 1...c5 to fight for d4 asymmetrically."})


def _list_pieces_text(fen: str) -> str:
    import chess
    b = chess.Board(fen)
    col = b.turn
    majors, pawns = [], []
    for sq, piece in sorted(b.piece_map().items()):
        if piece.color != col:
            continue
        name = chess.square_name(sq)
        if piece.piece_type == chess.PAWN:
            pawns.append(name)
        else:
            majors.append(f"{piece.symbol().upper()}={name}")
    parts = majors + ([f"pawns={','.join(pawns)}"] if pawns else [])
    return "pieces: " + ", ".join(parts)


def _envelope(
    scenario: Scenario, messages: list[dict[str, str]], annotated: AnnotatedPosition | None
) -> dict[str, Any]:
    expected = [
        name
        for m in messages
        if m["role"] == "assistant"
        for name in _TOOL.findall(m["content"])
    ]
    return {
        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
        "slice": scenario.slice,
        "kind": "harness_chess",
        "intent": scenario.intent,
        "plugin_context": scenario.plugin_context,
        "skills_index": [dict(s) for s in scenario.skills_index],
        "selected_skills": ["chess-coach"],
        "tool_manifest": list(scenario.tool_manifest),
        "expected_tool_calls": expected,
        "grounding_sources": ["board_state"] if scenario.slice in {"A", "B", "C", "D", "E", "F", "G", "H"} else [],
        "messages": messages,
        "acceptance_rules": [
            "final_no_xml", "known_tool_only", "args_match_schema",
            "selected_skill_exists", "skill_index_only_before_load", "skill_body_strict", "engine_grounded",
        ] + (["narration_grounded"] if scenario.slice in {"D", "E", "F", "G"} else []),
        "position_fen": scenario.position.fen if scenario.position else None,
        "stockfish_truth": (
            {"score_cp": annotated.score_cp, "best_san": annotated.best_san, "depth": annotated.depth}
            if annotated else None
        ),
    }
