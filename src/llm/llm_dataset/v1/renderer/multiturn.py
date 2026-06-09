"""Multi-turn follow-up rows: teach the model to CONTINUE a conversation rather
than restart it each turn.

Shape mirrors EPHEMERAL serving exactly: turn 1 is just (user question, coach
answer) — its tool scratchpad is NOT in the row, and the turn-1 answer is marked
`train: False` so it is context-only (no loss), matching what the served model
sees at turn 2. Only turn 2 is trained.

Two archetypes (seed-split ~50/50):
- A (reference-only): a "why?"-style follow-up answered from the established
  context with NO tool and NO invented specifics — references the prior turn and
  offers to dig deeper. Teaches "don't re-dump / re-call when you already said it".
- B (needs new info): a "what should I play / exact eval" follow-up that DOES
  re-run the right tool (correct — new info), and the answer connects back to the
  prior turn. Teaches "re-ground when the question genuinely needs it".
"""
from __future__ import annotations

import re
from typing import Any

from ..annotator import AnnotatedPosition, StockfishAnnotator
from ..board_facts import board_state_line
from ..sampler import Scenario
from . import tone
from .chess import INTERNAL_LESSON, _style_prompt
from .leadins import ask, lead
from .text import eval_magnitude, score_pawns, score_phrase, score_text

_TOOL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")

TURN1_QS = ("who's winning?", "how am I doing?", "rate this position", "how's it looking?")
TURN2_WHY = ("why?", "why do you say that?", "explain that", "what makes you say so?", "how come?")
TURN2_TOOL = ("what should I play then?", "ok what's the best move?",
              "so what's the exact eval?", "give me the line then")
BACKREF_A = ("Same read as before —", "Like I said,", "As I noted,")
BACKREF_B = ("Since you asked,", "Right —", "Building on that,", "Okay —")


def _is_b(scenario: Scenario) -> bool:
    return scenario.seed % 2 == 0


def _side(cp: int) -> str:
    return "White" if cp > 0 else "Black"


def _final_a(annotated: AnnotatedPosition, seed: int) -> str:
    cp = annotated.score_cp
    standing = "it's still about level" if abs(cp) < 40 else f"{_side(cp)} is still on top"
    backref = tone.pick(seed, BACKREF_A)
    return ask(f"{backref} {standing} — that's the read straight off the position, not a guess.", seed, 4)


def _turn2_tool(messages: list[dict], scenario: Scenario, annotated: AnnotatedPosition, user2: str) -> str:
    seed = scenario.seed
    backref = tone.pick(seed, BACKREF_B)
    messages.append({"role": "assistant", "content": f"{lead(seed, 'load_skill', 0)}\n<tool>load_skill name=chess-coach</tool>"})
    messages.append({"role": "tool", "content": INTERNAL_LESSON})
    messages.append({"role": "assistant", "content": f"{lead(seed, 'board_state', 2)}\n<tool>board_state fields=basic</tool>"})
    messages.append({"role": "tool", "content": board_state_line(annotated.fen)})
    if "eval" in user2.lower() or "exact" in user2.lower():
        messages.append({"role": "assistant", "content": f"{lead(seed, 'eval', 3)}\n<tool>eval depth=15</tool>"})
        messages.append({"role": "tool", "content": score_text(annotated)})
        return ask(f"{backref} {score_phrase(annotated)}", seed, 4)
    messages.append({"role": "assistant", "content": f"{lead(seed, 'best_move', 3)}\n<tool>best_move depth=15 series=3</tool>"})
    line = " ".join(annotated.best_line_sans)
    messages.append({"role": "tool", "content": f"best_line: {line}, score: {score_pawns(annotated)}"})
    nxt = " ".join(annotated.best_line_sans[1:3])
    return ask(f"{backref} {annotated.best_san} is the move; the line runs {nxt}.", seed, 4)


def render_multiturn_row(scenario: Scenario, annotator: StockfishAnnotator) -> dict[str, Any]:
    annotated = annotator.annotate(scenario.position.fen, depth=12)
    seed = scenario.seed
    user1 = _style_prompt(tone.pick(seed, TURN1_QS), scenario)
    final1 = ask(eval_magnitude(annotated, seed), seed, 1)
    # turn 1 is context only: tool scratchpad omitted, answer masked from loss.
    messages: list[dict[str, str]] = [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": final1, "train": False},
    ]
    if _is_b(scenario):
        user2 = _style_prompt(tone.pick(seed, TURN2_TOOL), scenario)
        messages.append({"role": "user", "content": user2})
        final2 = _turn2_tool(messages, scenario, annotated, user2)
        selected = ["chess-coach"]
        rules = ["final_no_xml", "known_tool_only", "args_match_schema",
                 "selected_skill_exists", "engine_grounded", "narration_grounded"]
    else:
        user2 = _style_prompt(tone.pick(seed, TURN2_WHY), scenario)
        messages.append({"role": "user", "content": user2})
        final2 = _final_a(annotated, seed)
        selected = []
        rules = ["final_no_xml", "known_tool_only", "args_match_schema"]
    messages.append({"role": "assistant", "content": final2})
    return _envelope(scenario, messages, annotated, selected, rules)


def _envelope(scenario, messages, annotated, selected, rules) -> dict[str, Any]:
    expected = [name for m in messages if m["role"] == "assistant"
                for name in _TOOL.findall(m["content"])]
    return {
        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
        "slice": scenario.slice,
        "kind": "harness_chess",
        "intent": scenario.intent,
        "plugin_context": scenario.plugin_context,
        "skills_index": [dict(s) for s in scenario.skills_index],
        "selected_skills": selected,
        "tool_manifest": list(scenario.tool_manifest),
        "expected_tool_calls": expected,
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": rules,
        "position_fen": scenario.position.fen if scenario.position else None,
        "stockfish_truth": {"score_cp": annotated.score_cp, "best_san": annotated.best_san,
                            "depth": annotated.depth},
    }
