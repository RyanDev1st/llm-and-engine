"""Turn-2 follow-up archetypes for the multi-turn slice.

The product lesson (Ryan's v5 review): a follow-up that ASSERTS a position fact must
RE-GROUND it with a fresh tool call — never answer "same read as before" from memory.
The old `reference` archetype did exactly that on "why?/explain" questions, training the
confabulation we fought across v1-v4. So every archetype here that states a standing,
move, or number first calls the relevant tool and reads it; only `clarify` stays
tool-free, and it ASKS rather than asserts (no fact to ground).

Each archetype also fulfils one of the offer-closers the single-turn finals dangle
(leadins.GUIDING): the plan, the threats, the deeper line, the alternatives, the best
move, the exact eval. That closes the offer→fulfil gap (offers were trained ~9x more
than grounded fulfilments).
"""
from __future__ import annotations

import chess

from ..annotator import AnnotatedPosition
from ..board_facts import board_state_line
from ..move_facts import move_facts
from ..sampler import Scenario
from . import tone
from .chess import INTERNAL_LESSON, _best_moves_result, _style_prompt
from .finals import _threat_body
from .grounded import move_reason
from .leadins import ask
from .tags import skill_call_msg, tool_call_msg, tool_result_msg
from .text import eval_magnitude, score_pawns, score_phrase, score_text

# Turn-2 user asks, grouped by the offer they accept.
TURN2_WHY = ("why?", "why do you say that?", "explain that", "what makes you say so?",
             "how come?", "explain the idea behind it")
TURN2_TOOL = ("what should I play then?", "ok what's the best move?", "what's the strongest move?")
TURN2_EVAL = ("so what's the exact eval?", "give me the precise number", "exact centipawns?")
TURN2_CLARIFY = ("can you help me here?", "what about the other side?", "what now?",
                 "not sure what to do", "where do I go from here?")
TURN2_STUCK = ("I'm stuck", "no idea what to do", "I don't know", "I'm totally lost", "I give up")
TURN2_PLAN = ("map the plan to convert it", "yes, the plan", "how do I convert this?",
              "what's the plan here?", "ok keep going from here")
TURN2_THREATS = ("check their threats", "what's their best reply?", "what are they threatening?",
                 "what does my opponent have?", "their best reply first")
TURN2_LINE = ("show the next few moves", "go deeper on the main line", "give me the line then",
              "what's the continuation?")
TURN2_ALTS = ("what are the alternatives?", "any other options?", "what else could I play?",
              "look at the alternatives")
BACKREF = ("Since you asked,", "Right —", "Building on that,", "Okay —", "Sure —")

# turn-2 grounded follow-ups carry the same grounded rule-set the chess slices use.
_TOOL_RULES = ["final_no_xml", "known_tool_only", "args_match_schema",
               "selected_skill_exists", "engine_grounded", "narration_grounded"]
_REF_RULES = ["final_no_xml", "known_tool_only", "args_match_schema"]


def _user(scenario: Scenario, text: str) -> dict:
    return {"role": "user", "content": _style_prompt(text, scenario)}


def _mover_cp(annotated: AnnotatedPosition) -> int:
    white = chess.Board(annotated.fen).turn == chess.WHITE
    raw = (10000 if annotated.score_kind == "mate" and annotated.score_cp > 0
           else -10000 if annotated.score_kind == "mate" else annotated.score_cp)
    return raw if white else -raw


def _reason(annotated: AnnotatedPosition, seed: int) -> str:
    mf = move_facts(annotated.fen, annotated.best_san)
    return move_reason(mf, _mover_cp(annotated), seed) if mf else "it's the engine's top choice"


def _reload_and_read(messages: list[dict], annotated: AnnotatedPosition) -> None:
    """Common turn-2 grounding prefix: reload the skill, then read the board."""
    messages.append(skill_call_msg("chess-coach"))
    messages.append(tool_result_msg("load_skill", INTERNAL_LESSON))
    messages.append(tool_call_msg("board_state", {"fields": "basic"}))
    messages.append(tool_result_msg("board_state", board_state_line(annotated.fen)))


def _emit_best_move(messages: list[dict], annotated: AnnotatedPosition, *, series: int = 3) -> None:
    messages.append(tool_call_msg("best_move", {"depth": 15, "series": series}))
    line = " ".join(annotated.best_line_sans)
    messages.append(tool_result_msg("best_move", f"best_line: {line}, score: {score_pawns(annotated)}"))


def _emit_top_moves(messages: list[dict], annotated: AnnotatedPosition, *, top: int = 3) -> None:
    messages.append(tool_call_msg("best_move", {"depth": 15, "top": top}))
    messages.append(tool_result_msg("best_move", _best_moves_result(annotated.top_moves[:top])))


def _emit_eval(messages: list[dict], annotated: AnnotatedPosition) -> None:
    messages.append(tool_call_msg("eval", {"depth": 15}))
    messages.append(tool_result_msg("eval", score_text(annotated)))


def _emit_threats(messages: list[dict], annotated: AnnotatedPosition) -> None:
    messages.append(tool_call_msg("threats", {"depth": 15}))
    threat = annotated.threats_san
    body = (f"threat: {threat}, score: {score_pawns(annotated)}" if threat
            else f"threats: none, score: {score_pawns(annotated)}")
    messages.append(tool_result_msg("threats", body))


# ---- composers (all SAN/number cited is present in the tool result above) ----
def _why_final(annotated: AnnotatedPosition, seed: int) -> str:
    reason = _reason(annotated, seed)
    cap = reason[0].upper() + reason[1:]
    nxt = " ".join(annotated.best_line_sans[:3])
    tail = f" — the engine's line {nxt} bears it out" if nxt else ""
    return ask(f"{tone.pick(seed, BACKREF)} {cap}{tail}. {eval_magnitude(annotated, seed)}", seed, 4)


def _plan_final(annotated: AnnotatedPosition, seed: int) -> str:
    nxt = " ".join(annotated.best_line_sans[1:3])
    cont = f", then {nxt}" if nxt else ""
    return ask(f"{tone.pick(seed, BACKREF)} here's how you convert it: {annotated.best_san}{cont} — "
               f"{_reason(annotated, seed)}. {eval_magnitude(annotated, seed)}", seed, 4)


def _line_final(annotated: AnnotatedPosition, seed: int) -> str:
    full = " ".join(annotated.best_line_sans[:5]) or annotated.best_san
    return ask(f"{tone.pick(seed, BACKREF)} going deeper, the main line runs {full} — "
               f"{_reason(annotated, seed)}. {eval_magnitude(annotated, seed)}", seed, 4)


def _alts_final(annotated: AnnotatedPosition, seed: int) -> str:
    standing, reason = eval_magnitude(annotated, seed), _reason(annotated, seed)
    if len(annotated.top_moves) >= 2:
        alts = ", ".join(san for san, _ in annotated.top_moves[1:3])
        return ask(f"{tone.pick(seed, BACKREF)} {annotated.best_san} is still best — {reason}. "
                   f"{alts} are the other tries. {standing}", seed, 4)
    return ask(f"{tone.pick(seed, BACKREF)} {annotated.best_san} is the cleanest — {reason}. {standing}", seed, 4)


# ---- archetypes: (final, selected_skills, rules) ----
def _arch_why(messages, scenario, annotated):
    messages.append(_user(scenario, tone.pick(scenario.seed, TURN2_WHY)))
    _reload_and_read(messages, annotated)
    _emit_best_move(messages, annotated, series=3)
    return _why_final(annotated, scenario.seed), ["chess-coach"], list(_TOOL_RULES)


def _arch_tool(messages, scenario, annotated):
    user2 = _style_prompt(tone.pick(scenario.seed, TURN2_TOOL), scenario)
    messages.append({"role": "user", "content": user2})
    _reload_and_read(messages, annotated)
    _emit_best_move(messages, annotated, series=3)
    nxt = " ".join(annotated.best_line_sans[1:3])
    cont = f"; the line runs {nxt}" if nxt else ""
    final = ask(f"{tone.pick(scenario.seed, BACKREF)} {annotated.best_san} is the move{cont}.", scenario.seed, 4)
    return final, ["chess-coach"], list(_TOOL_RULES)


def _arch_eval(messages, scenario, annotated):
    messages.append(_user(scenario, tone.pick(scenario.seed, TURN2_EVAL)))
    _reload_and_read(messages, annotated)
    _emit_eval(messages, annotated)
    return ask(f"{tone.pick(scenario.seed, BACKREF)} {score_phrase(annotated)}", scenario.seed, 4), \
        ["chess-coach"], list(_TOOL_RULES)


def _arch_plan(messages, scenario, annotated):
    messages.append(_user(scenario, tone.pick(scenario.seed, TURN2_PLAN)))
    _reload_and_read(messages, annotated)
    _emit_best_move(messages, annotated, series=3)
    return _plan_final(annotated, scenario.seed), ["chess-coach"], list(_TOOL_RULES)


def _arch_threats(messages, scenario, annotated):
    messages.append(_user(scenario, tone.pick(scenario.seed, TURN2_THREATS)))
    _reload_and_read(messages, annotated)
    _emit_threats(messages, annotated)
    return ask(f"{tone.pick(scenario.seed, BACKREF)} {_threat_body(annotated, False, scenario.seed)}",
               scenario.seed, 4), ["chess-coach"], list(_TOOL_RULES)


def _arch_line(messages, scenario, annotated):
    messages.append(_user(scenario, tone.pick(scenario.seed, TURN2_LINE)))
    _reload_and_read(messages, annotated)
    _emit_best_move(messages, annotated, series=5)
    return _line_final(annotated, scenario.seed), ["chess-coach"], list(_TOOL_RULES)


def _arch_alts(messages, scenario, annotated):
    messages.append(_user(scenario, tone.pick(scenario.seed, TURN2_ALTS)))
    _reload_and_read(messages, annotated)
    _emit_top_moves(messages, annotated, top=3)
    return _alts_final(annotated, scenario.seed), ["chess-coach"], list(_TOOL_RULES)


def _arch_stuck(messages, scenario, annotated):
    seed = scenario.seed
    messages.append(_user(scenario, tone.pick(seed, TURN2_STUCK)))
    _reload_and_read(messages, annotated)
    _emit_best_move(messages, annotated, series=3)
    final = ask(f"{tone.pick(seed, BACKREF)} no need to restart — a grounded try here is "
                f"{annotated.best_san}; want me to walk the idea, or see your other options?", seed, 4)
    return final, ["chess-coach"], list(_TOOL_RULES)


# Clarify: AMBIGUOUS follow-up answered with ONE clarifying question — asserts no fact,
# so it stays tool-free (the only archetype that does).
_CLARIFY_OFFERS = (
    "happy to take it further — do you want the attacking plan, or to shore up your defense first?",
    "glad to keep going — should we build the attacking plan, or firm up the defense first?",
    "we can dig deeper — want to press for the attack, or stabilize the position first?",
    "plenty more here — go on the offensive, or solidify what you've got first?",
    "let's keep at it — chase the initiative, or tighten the defense first?",
    "happy to continue — push for an attack, or settle the position down first?",
    "we can go either way — map an attacking plan, or patch the weak spots first?",
    "more to do here — would you rather create threats, or neutralize theirs first?",
)


def _arch_clarify(messages, scenario, annotated):
    seed = scenario.seed
    messages.append(_user(scenario, tone.pick(seed, TURN2_CLARIFY)))
    return ask(tone.pick(seed * 31 + 5, _CLARIFY_OFFERS), seed, 4), [], list(_REF_RULES)


def _arch_self_correct(messages, scenario, annotated):
    seed = scenario.seed
    messages.append(_user(scenario, tone.pick(seed, TURN2_EVAL)))
    _reload_and_read(messages, annotated)
    messages.append(tool_call_msg("eval", {"depth": 99}))
    messages.append(tool_result_msg("eval", "error: invalid_syntax"))
    messages.append(tool_call_msg("eval", {"depth": 15}))     # diagnose + retry, don't give up
    messages.append(tool_result_msg("eval", score_text(annotated)))
    return ask(f"{tone.pick(seed, BACKREF)} {score_phrase(annotated)}", seed, 4), \
        ["chess-coach"], list(_TOOL_RULES)


# Order fixes the seed%N distribution; only _arch_clarify is tool-free (1 of 10).
ARCHETYPES = (
    _arch_why, _arch_tool, _arch_eval, _arch_plan, _arch_threats, _arch_line,
    _arch_alts, _arch_stuck, _arch_self_correct, _arch_clarify,
)
