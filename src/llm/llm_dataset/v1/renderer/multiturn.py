"""Multi-turn follow-up rows: teach the model to CONTINUE a conversation rather
than restart it each turn, and to TRACK dialogue state (follow-up / clarify /
stuck / recover) instead of re-dumping or fabricating.

Shape mirrors EPHEMERAL serving exactly: turn 1 is just (user question, coach
answer) — its tool scratchpad is NOT in the row, and the turn-1 answer is marked
`train: False` so it is context-only (no loss), matching what the served model
sees at turn 2. Only turn 2 is trained. Turn 1 carries NO <think> on purpose: it
mirrors the served VISIBLE reply (think is stripped before display); turn 2 — the
trained step — carries the inline <think> decision trace.

Five archetypes (seed-split ~even):
- reference: a "why?"-style follow-up answered from the established context with
  NO tool and NO invented specifics — references the prior turn. Teaches "don't
  re-dump / re-call when you already said it".
- tool: a "what should I play / exact eval" follow-up that DOES re-run the right
  tool (correct — new info), answer connects back. Teaches "re-ground when needed".
- clarify: an AMBIGUOUS follow-up — the coach asks ONE clarifying question rather
  than guessing or calling a tool. Teaches "ask when the ask is unclear".
- stuck: the user is stuck ("no idea", "I'm lost") — the coach gives a grounded
  next-step NUDGE (engine's move via the tool), it does NOT restart/re-roll.
- self_correct: a tool call errors mid-dialogue; the coach diagnoses and retries
  instead of giving up or fabricating. Teaches recovery inside a conversation.
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
from .thinking import think, think_answer, think_fix

_TOOL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")

TURN1_QS = ("who's winning?", "how am I doing?", "rate this position", "how's it looking?")
TURN2_WHY = ("why?", "why do you say that?", "explain that", "what makes you say so?", "how come?")
TURN2_TOOL = ("what should I play then?", "ok what's the best move?",
              "so what's the exact eval?", "give me the line then")
TURN2_EVAL = ("so what's the exact eval?", "give me the precise number", "exact centipawns?")
TURN2_CLARIFY = ("can you help me here?", "what about the other side?", "what now?",
                 "not sure what to do", "where do I go from here?")
TURN2_STUCK = ("I'm stuck", "no idea what to do", "I don't know", "I'm totally lost", "I give up")
BACKREF_A = ("Same read as before —", "Like I said,", "As I noted,")
BACKREF_B = ("Since you asked,", "Right —", "Building on that,", "Okay —")

GOAL = "follow up on where the game stands"
# turn-2 tool follow-ups carry the same grounded rule-set the chess slices use.
_TOOL_RULES = ["final_no_xml", "known_tool_only", "args_match_schema",
               "selected_skill_exists", "engine_grounded", "narration_grounded"]
_REF_RULES = ["final_no_xml", "known_tool_only", "args_match_schema"]


def _archetype(seed: int) -> int:
    return seed % 5


def _act(seed: int, name: str, step: int, call: str, have: str) -> str:
    """Assistant tool-step in the trained turn: <think> + lead-in + the call."""
    return f"{think(seed, name, step, goal=GOAL, have=have)}\n{lead(seed, name, step)}\n{call}"


def _side(cp: int) -> str:
    return "White" if cp > 0 else "Black"


def _final_a(annotated: AnnotatedPosition, seed: int) -> str:
    cp = annotated.score_cp
    standing = "it's still about level" if abs(cp) < 40 else f"{_side(cp)} is still on top"
    backref = tone.pick(seed, BACKREF_A)
    return ask(f"{backref} {standing} — that's the read straight off the position, not a guess.", seed, 4)


def _reload_and_read(messages: list[dict], seed: int, annotated: AnnotatedPosition) -> None:
    """Common turn-2 grounding prefix: reload the skill, then read the board."""
    messages.append({"role": "assistant", "content": _act(seed, "load_skill", 0, "<tool>load_skill name=chess-coach</tool>", "")})
    messages.append({"role": "tool", "content": INTERNAL_LESSON})
    messages.append({"role": "assistant", "content": _act(seed, "board_state", 2, "<tool>board_state fields=basic</tool>", "skill")})
    messages.append({"role": "tool", "content": board_state_line(annotated.fen)})


def _turn2_tool(messages: list[dict], scenario: Scenario, annotated: AnnotatedPosition, user2: str) -> str:
    seed = scenario.seed
    backref = tone.pick(seed, BACKREF_B)
    _reload_and_read(messages, seed, annotated)
    if "eval" in user2.lower() or "exact" in user2.lower():
        messages.append({"role": "assistant", "content": _act(seed, "eval", 3, "<tool>eval depth=15</tool>", "board")})
        messages.append({"role": "tool", "content": score_text(annotated)})
        return ask(f"{backref} {score_phrase(annotated)}", seed, 4)
    messages.append({"role": "assistant", "content": _act(seed, "best_move", 3, "<tool>best_move depth=15 series=3</tool>", "board")})
    line = " ".join(annotated.best_line_sans)
    messages.append({"role": "tool", "content": f"best_line: {line}, score: {score_pawns(annotated)}"})
    nxt = " ".join(annotated.best_line_sans[1:3])
    return ask(f"{backref} {annotated.best_san} is the move; the line runs {nxt}.", seed, 4)


def _arch_reference(messages, scenario, annotated):
    seed = scenario.seed
    messages.append({"role": "user", "content": _style_prompt(tone.pick(seed, TURN2_WHY), scenario)})
    return f"{think_answer(seed, GOAL, enough=False)}\n{_final_a(annotated, seed)}", [], _REF_RULES


def _arch_clarify(messages, scenario, annotated):
    seed = scenario.seed
    messages.append({"role": "user", "content": _style_prompt(tone.pick(seed, TURN2_CLARIFY), scenario)})
    backref = tone.pick(seed, BACKREF_A)
    answer = ask(f"{backref} happy to take it further — do you want the attacking plan, "
                 f"or to shore up your defense first?", seed, 4)
    return f"{think_answer(seed, GOAL, enough=False)}\n{answer}", [], _REF_RULES


def _arch_tool(messages, scenario, annotated):
    seed = scenario.seed
    user2 = _style_prompt(tone.pick(seed, TURN2_TOOL), scenario)
    messages.append({"role": "user", "content": user2})
    answer = _turn2_tool(messages, scenario, annotated, user2)
    return f"{think_answer(seed, GOAL)}\n{answer}", ["chess-coach"], list(_TOOL_RULES)


def _arch_stuck(messages, scenario, annotated):
    seed = scenario.seed
    messages.append({"role": "user", "content": _style_prompt(tone.pick(seed, TURN2_STUCK), scenario)})
    _reload_and_read(messages, seed, annotated)
    messages.append({"role": "assistant", "content": _act(seed, "best_move", 3, "<tool>best_move depth=15 series=3</tool>", "board")})
    line = " ".join(annotated.best_line_sans)
    messages.append({"role": "tool", "content": f"best_line: {line}, score: {score_pawns(annotated)}"})
    backref = tone.pick(seed, BACKREF_B)
    answer = ask(f"{backref} no need to restart — a grounded try here is {annotated.best_san}; "
                 f"want me to walk the idea, or see your other options?", seed, 4)
    return f"{think_answer(seed, GOAL)}\n{answer}", ["chess-coach"], list(_TOOL_RULES)


def _arch_self_correct(messages, scenario, annotated):
    seed = scenario.seed
    messages.append({"role": "user", "content": _style_prompt(tone.pick(seed, TURN2_EVAL), scenario)})
    _reload_and_read(messages, seed, annotated)
    messages.append({"role": "assistant", "content": _act(seed, "eval", 3, "<tool>eval depth=99</tool>", "board")})
    messages.append({"role": "tool", "content": "error: invalid_syntax"})
    messages.append({"role": "assistant", "content": f"{think_fix(seed, 'eval')}\n{lead(seed, 'eval', 4)}\n<tool>eval depth=15</tool>"})
    messages.append({"role": "tool", "content": score_text(annotated)})
    backref = tone.pick(seed, BACKREF_B)
    answer = ask(f"{backref} {score_phrase(annotated)}", seed, 4)
    return f"{think_answer(seed, GOAL)}\n{answer}", ["chess-coach"], list(_TOOL_RULES)


_ARCHETYPES = (_arch_tool, _arch_reference, _arch_clarify, _arch_stuck, _arch_self_correct)


def render_multiturn_row(scenario: Scenario, annotator: StockfishAnnotator) -> dict[str, Any]:
    annotated = annotator.annotate(scenario.position.fen, depth=12)
    seed = scenario.seed
    user1 = _style_prompt(tone.pick(seed, TURN1_QS), scenario)
    final1 = ask(eval_magnitude(annotated, seed), seed, 1)
    # turn 1 is context only: tool scratchpad omitted, answer masked from loss,
    # and NO <think> (it mirrors the served VISIBLE reply, which strips think).
    messages: list[dict[str, str]] = [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": final1, "train": False},
    ]
    final2, selected, rules = _ARCHETYPES[_archetype(seed)](messages, scenario, annotated)
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
