"""Multi-turn follow-up rows: teach the model to CONTINUE a conversation rather
than restart it each turn, and to TRACK dialogue state instead of re-dumping or
fabricating.

Shape mirrors EPHEMERAL serving exactly: turn 1 is just (user question, coach
answer) — its tool scratchpad is NOT in the row, and the turn-1 answer is marked
`train: False` so it is context-only (no loss), matching what the served model
sees at turn 2. Only turn 2 is trained. Turn 1 carries NO reasoning channel: it
mirrors the served VISIBLE reply (thinking is stripped before display).

The turn-2 archetypes live in `multiturn_followups.py` (renderer/). Core lesson:
a follow-up that ASSERTS a position fact (a standing, a move, a number) RE-GROUNDS
it with a fresh tool call — the model never answers "same read as before" from
memory. Each grounded archetype also FULFILS one of the offer-closers single-turn
finals dangle (the plan, threats, the line, the alternatives, the best move, the
exact eval), closing the offer→fulfil gap. Only `clarify` stays tool-free: it ASKS
rather than asserts, so there is no fact to ground.
"""
from __future__ import annotations

from typing import Any

from ..annotator import AnnotatedPosition, StockfishAnnotator
from ..sampler import Scenario
from . import tone
from .chess import _style_prompt
from .multiturn_followups import ARCHETYPES
from .tags import tool_calls_of
from .text import eval_magnitude
from .leadins import ask
from .thinking import pick_mode

TURN1_QS = ("who's winning?", "how am I doing?", "rate this position", "how's it looking?",
            "what's the read here?", "am I better or worse?", "where do I stand?", "how's my position?")


def _archetype(seed: int) -> int:
    return seed % len(ARCHETYPES)


def render_multiturn_row(scenario: Scenario, annotator: StockfishAnnotator) -> dict[str, Any]:
    annotated = annotator.annotate(scenario.position.fen, depth=12)
    seed = scenario.seed
    mode = pick_mode(seed)
    user1 = _style_prompt(tone.pick(seed, TURN1_QS), scenario)
    final1 = ask(eval_magnitude(annotated, seed), seed, 1)
    # turn 1 is context only: tool scratchpad omitted, answer masked from loss, and
    # NO reasoning (it mirrors the served VISIBLE reply, which strips thinking).
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": final1, "train": False},
    ]
    final2, selected, rules = ARCHETYPES[_archetype(seed)](messages, scenario, annotated)
    messages.append({"role": "assistant", "content": final2})
    return _envelope(scenario, messages, annotated, selected, rules, mode)


def _envelope(scenario, messages, annotated, selected, rules, mode="think") -> dict[str, Any]:
    expected = [tc["name"] for m in messages if m["role"] == "assistant"
                for tc in tool_calls_of(m) if tc["name"] != "load_skill"]
    return {
        "id": f"v1_{scenario.slice.lower()}_{scenario.seed:09d}",
        "slice": scenario.slice,
        "kind": "harness_chess",
        "reasoning_mode": mode,
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
