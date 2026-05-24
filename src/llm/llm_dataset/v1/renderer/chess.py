from __future__ import annotations

from typing import Any

from ..annotator import AnnotatedPosition, StockfishAnnotator
from ..sampler import Scenario
from . import tone
from .text import eval_language, score_pawns, score_text

SLICE_USER_TEMPLATES = {
    "A": ("play {san}", "let's go {san}", "{san} for me", "push {san}"),
    "D": ("who is winning?", "rate this position", "is this lost for me?", "how is it?"),
    "E": ("what should I play?", "best move?", "give me the line", "show me a plan"),
    "F": ("how was that move?", "did I blunder?", "rate my last move", "was that ok?"),
    "G": ("any threats?", "what is the opponent up to?", "watch out for what?"),
    "H": ("legal moves on {square}?", "undo that", "what pieces are left?"),
    "I": ("what is the sicilian?", "why castle?", "what is a fork?", "who is capablanca?"),
    "J": ("hey there", "thanks!", "what can you do?", "feeling good"),
    "K": ("how much is a knight worth?", "is the queen the strongest piece?", "checkmate, that's a deal"),
}

INTERNAL_LESSON = "Use board tools before claims. Ground evaluation in Stockfish output."


def render_chess_row(scenario: Scenario, annotator: StockfishAnnotator) -> dict[str, Any]:
    annotated = annotator.annotate(scenario.position.fen, depth=15) if scenario.position else None
    user = _user_message(scenario)
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]
    _emit_skill_load(messages, scenario)
    if scenario.slice in {"A", "D", "E", "F", "G", "H"}:
        messages.append({"role": "assistant", "content": "<tool>board_state fields=basic</tool>"})
        messages.append({"role": "tool", "content": _board_state_text(annotated)})
    _emit_slice_tool(messages, scenario, annotated)
    messages.append({"role": "assistant", "content": _final(scenario, annotated)})
    return _envelope(scenario, messages, annotated)


def _user_message(scenario: Scenario) -> str:
    templates = SLICE_USER_TEMPLATES.get(scenario.slice, ("explain the position",))
    base = tone.pick(scenario.seed, templates)
    if "{san}" in base and scenario.position:
        return base.replace("{san}", "e4")
    if "{square}" in base:
        return base.replace("{square}", "e2")
    return base


def _emit_skill_load(messages: list[dict[str, str]], scenario: Scenario) -> None:
    messages.append({"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"})
    messages.append({"role": "tool", "content": INTERNAL_LESSON})


def _board_state_text(annotated: AnnotatedPosition | None) -> str:
    if annotated is None:
        return "board_state: turn=white, check=no, legal_count=20"
    return f"board_state: turn=white, fen={annotated.fen}, check=no, legal_count=20"


def _emit_slice_tool(
    messages: list[dict[str, str]], scenario: Scenario, annotated: AnnotatedPosition | None
) -> None:
    if scenario.slice == "A":
        messages.append({"role": "assistant", "content": "<tool>move san=e4</tool>"})
        messages.append({"role": "tool", "content": "success: e4"})
    elif scenario.slice == "D" and annotated:
        messages.append({"role": "assistant", "content": "<tool>eval depth=15</tool>"})
        messages.append({"role": "tool", "content": score_text(annotated)})
    elif scenario.slice == "E" and annotated:
        messages.append({"role": "assistant", "content": "<tool>best_move depth=15 series=3</tool>"})
        line = " ".join(annotated.best_line_sans)
        messages.append({"role": "tool", "content": f"best_line: {line}, score: {score_pawns(annotated)}"})
    elif scenario.slice == "F":
        messages.append({"role": "assistant", "content": "<tool>review_move depth=12</tool>"})
        messages.append({"role": "tool", "content": "review: e4, label=good, delta=+0.05 pawns, best_was=e4"})
    elif scenario.slice == "G" and annotated:
        threat = annotated.threats_san or "none"
        messages.append({"role": "assistant", "content": "<tool>threats depth=12</tool>"})
        messages.append({"role": "tool", "content": f"threats: opponent's best is {threat}, score for them: {score_pawns(annotated)}"})
    elif scenario.slice == "H":
        messages.append({"role": "assistant", "content": "<tool>list_pieces color=mine</tool>"})
        messages.append({"role": "tool", "content": "pieces: K=e1, Q=d1, R=a1, R=h1, B=c1, B=f1, N=b1, N=g1"})
    elif scenario.slice == "I":
        messages.append({"role": "assistant", "content": "<tool>ask_chessbot query=sicilian</tool>"})
        messages.append({"role": "tool", "content": "Sicilian: Black answers 1.e4 with 1...c5 to fight for d4 asymmetrically."})


def _final(scenario: Scenario, annotated: AnnotatedPosition | None) -> str:
    if scenario.tone == "warm":
        opener = tone.pick(scenario.seed, tone.OPENERS_WARM)
    elif scenario.tone == "blunt":
        opener = tone.pick(scenario.seed, tone.OPENERS_BLUNT)
    else:
        opener = tone.pick(scenario.seed, tone.OPENERS_SOCRATIC)
    if scenario.slice == "A":
        return f"{opener} Played e4 — central, opens lines for the bishop and queen."
    if scenario.slice == "D" and annotated:
        return f"{opener} {eval_language(annotated)}"
    if scenario.slice == "E" and annotated:
        return f"{opener} Engine likes {annotated.best_san}; the plan continues {' '.join(annotated.best_line_sans[1:3])}."
    if scenario.slice == "F":
        return f"{opener} That last move was solid. Engine still likes the same continuation."
    if scenario.slice == "G" and annotated:
        threat = annotated.threats_san or "nothing forcing"
        return f"{opener} Watch for {threat} from your opponent."
    if scenario.slice == "H":
        return f"{opener} You still have full material on the back rank — eight pieces ready."
    if scenario.slice == "I":
        return f"{opener} It's a sharp counter to 1.e4 that fights for the centre asymmetrically."
    if scenario.slice == "J":
        return f"{opener} Glad you're here. Ask me anything about the position or how you played."
    if scenario.slice == "K":
        return f"{opener} A knight is worth about three pawns in most positions, but context matters more than the number."
    return f"{opener} I read the position and the tools, then answered without inventing facts."


def _envelope(
    scenario: Scenario, messages: list[dict[str, str]], annotated: AnnotatedPosition | None
) -> dict[str, Any]:
    expected = [
        m["content"].split()[0].removeprefix("<tool>")
        for m in messages
        if m["role"] == "assistant" and m["content"].startswith("<tool>")
    ]
    return {
        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
        "slice": scenario.slice,
        "kind": "harness_chess",
        "intent": scenario.intent,
        "plugin_context": scenario.plugin_context,
        "skills_index": [{"name": s["name"], "description": s["description"]} for s in scenario.skills_index],
        "selected_skills": ["chess-coach"],
        "tool_manifest": list(scenario.tool_manifest),
        "expected_tool_calls": expected,
        "grounding_sources": ["board_state"] if scenario.slice in {"A", "D", "E", "F", "G", "H"} else [],
        "messages": messages,
        "acceptance_rules": [
            "final_no_xml", "known_tool_only", "args_match_schema",
            "selected_skill_exists", "skill_index_only_before_load", "engine_grounded",
        ],
        "position_fen": scenario.position.fen if scenario.position else None,
        "stockfish_truth": (
            {"score_cp": annotated.score_cp, "best_san": annotated.best_san, "depth": annotated.depth}
            if annotated else None
        ),
    }
