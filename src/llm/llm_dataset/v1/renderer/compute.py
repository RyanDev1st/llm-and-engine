"""V1_R compute-grounding renderer — the Stage-0 keystone slice (pure-chess v5).

Lesson (verification-as-tool-use): when the answer hinges on a number, the agent does
NOT compute or assert it in its head (a 4B fabricates arithmetic). It fills the ONE
canonical calculator template (catalog.CALC_TEMPLATE), runs it through the `python`
tool, reads the real stdout, and states the VERDICT grounded in that value. The numbers
are CHESS numbers — material points, game accuracy, tournament score, rating, eval in
pawns — so the behavior is taught in the product's own domain. No skill fits a plain
compute ask, so the model goes tool-direct (contract: load a skill only when one fits —
don't reflexively load game-reviewer for a bare arithmetic question).

Two shapes, mixed ~70/30: verify-then-claim (a judgment — "am I averaging 80%?") and
compute-on-request (a raw figure). Grounding is enforced: the tool result is produced by
the REAL executor (backend.sandbox.run_python), so train == serve, and the final cites
only the computed two-decimal value (narration_grounded / validate._FACT)."""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from backend.sandbox import run_python

from ..catalog import CALC_TEMPLATE
from ..sampler import Scenario
from .tags import tool_call_msg, tool_result_msg
from .thinking import pick_mode

_VERIFY_SHARE = 0.70
_ACC_T = (75, 80, 85, 90)
_PT_T = (12, 15, 18, 22, 26)
_CLOSERS = ("Want the breakdown?", "Anything else to check?", "Want me to round it?",
            "Want another run?", "", "", "")


def _wrap(expr: str) -> str:
    """Plug the expression into the ONE canonical calculator template — every row reuses
    the identical known-good wrapper, varying only the expression, so a weak coder model
    plug-and-plays it. Two-decimal print keeps the output groundable by validate._FACT."""
    return CALC_TEMPLATE.replace("EXPR", expr)


@lru_cache(maxsize=16384)
def _exec(code: str) -> str:
    return run_python(code)            # real executor -> train == serve output


def _ints(r: random.Random, n: int, lo: int, hi: int) -> list[int]:
    return [r.randint(lo, hi) for _ in range(n)]


def _army(r: random.Random) -> tuple[str, str]:
    """A random own-army description + the python expr for its material points."""
    rk, mi, p = r.randint(1, 2), r.randint(0, 3), r.randint(3, 8)
    parts = [f"{rk} rook{'s' if rk > 1 else ''}"]
    if mi:
        parts.append(f"{mi} minor piece{'s' if mi > 1 else ''}")
    parts.append(f"{p} pawns")
    return ", ".join(parts), f"{rk}*5 + {mi}*3 + {p}*1"


# --- verify-then-claim families: (prompt, code, build(val_str) -> grounded verdict) ---

def _material_check(r):
    army, expr = _army(r)
    t = r.choice(_PT_T)
    prompt = r.choice((
        f"I've got {army} — is that at least {t} points of material?",
        f"my material is {army}; does it come to {t} points or more?",
        f"do {army} add up to {t}+ points?"))

    def build(val):
        return (f"That's {val} points of material, at least {t}." if float(val) >= t
                else f"That's {val} points of material, under {t}.")
    return prompt, _wrap(expr), build


def _accuracy_check(r):
    a = _ints(r, r.choice((3, 4)), 60, 99)
    t = r.choice(_ACC_T)
    shown = ", ".join(f"{x}%" for x in a)
    prompt = r.choice((
        f"My last games were {shown} accuracy — am I averaging at least {t}%?",
        f"accuracy over recent games: {shown}; is the average {t}% or higher?",
        f"are {shown} averaging above {t}%?"))

    def build(val):
        v = float(val)
        if abs(v - t) < 0.005:
            return f"Your average accuracy is {val}% — right on the {t}% line."
        return f"Your average accuracy is {val}%, {'above' if v > t else 'below'} {t}%."
    return prompt, _wrap(f"sum({a}) / {len(a)}"), build


def _score_check(r):
    w, d, l = r.randint(2, 7), r.randint(0, 4), r.randint(0, 4)
    t = r.choice((3, 4, 5, 6, 7))
    prompt = r.choice((
        f"I went {w}W {d}D {l}L — is that at least {t} points?",
        f"with {w} wins, {d} draws, {l} losses, did I reach {t} points?",
        f"do {w} wins and {d} draws make {t}+ points?"))

    def build(val):
        return (f"That's {val} points, at least {t}." if float(val) >= t
                else f"That's {val} points, short of {t}.")
    return prompt, _wrap(f"{w}*1 + {d}*0.5"), build


def _swing_check(r):
    a, b = r.randint(-150, 50), r.randint(80, 400)
    t = r.choice((1, 2, 3))
    prompt = r.choice((
        f"The eval went from {a} to {b} centipawns — did I gain more than {t} pawns?",
        f"my move swung the eval {a}cp to {b}cp; is that over {t} pawns?",
        f"from {a} to {b} centipawns — more than {t} pawns gained?"))

    def build(val):
        return (f"That's a {val}-pawn swing, more than {t}." if float(val) > t
                else f"That's a {val}-pawn swing, not over {t}.")
    return prompt, _wrap(f"({b} - {a}) / 100"), build


def _rating_check(r):
    start = r.choice((1100, 1250, 1400, 1550, 1700))
    end = start + r.choice((40, 60, 80, 120)) + r.choice((-20, 0, 25))
    t = r.choice((50, 75, 100))
    prompt = r.choice((
        f"My rating went {start} to {end} — did I gain at least {t}?",
        f"from {start} up to {end}; is that a {t}+ point gain?",
        f"{start} then {end} — at least {t} rating gained?"))

    def build(val):
        return (f"You gained {val} rating, at least {t}." if float(val) >= t
                else f"You gained {val} rating, under {t}.")
    return prompt, _wrap(f"{end} - {start}"), build


# --- compute-on-request families: (prompt, code, build(val_str) -> raw figure) ---

def _calc_make(prompt: str, expr: str, fmt: str):
    return prompt, _wrap(expr), (lambda val: fmt.format(val=val))


def _material_points(r):
    army, expr = _army(r)
    p = r.choice((f"How many points is {army}?", f"material value of {army}?"))
    return _calc_make(p, expr, "That's {val} points of material.")


def _cp_to_pawns(r):
    cp = r.choice((45, 80, 120, 175, 240, 310, 420))
    p = r.choice((f"The eval is {cp} centipawns — how many pawns is that?",
                  f"convert {cp}cp to pawns"))
    return _calc_make(p, f"{cp} / 100", f"{cp} centipawns is {{val}} pawns.")


def _score_points(r):
    w, d = r.randint(2, 8), r.randint(0, 5)
    p = r.choice((f"{w} wins and {d} draws — how many points?", f"score for {w}W {d}D?"))
    return _calc_make(p, f"{w} + {d}*0.5", "That's {val} points.")


def _accuracy_avg(r):
    a = _ints(r, r.choice((3, 4)), 60, 99)
    shown = ", ".join(f"{x}%" for x in a)
    p = r.choice((f"What's my average accuracy across {shown}?", f"mean accuracy of {shown}?"))
    return _calc_make(p, f"sum({a}) / {len(a)}", "Your average accuracy is {val}%.")


def _rating_avg(r):
    a = _ints(r, r.choice((3, 4)), 1100, 1900)
    shown = ", ".join(map(str, a))
    p = r.choice((f"Average of my ratings {shown}?", f"mean rating across {shown}?"))
    return _calc_make(p, f"sum({a}) / {len(a)}", "Your average rating is {val}.")


_VERIFY = (_material_check, _accuracy_check, _score_check, _swing_check, _rating_check)
_CALC = (_material_points, _cp_to_pawns, _score_points, _accuracy_avg, _rating_avg)


def _scene(seed: int) -> tuple[str, str, str]:
    """(user prompt, python script, final reply with the grounded value/verdict)."""
    r = random.Random(seed * 137 + 11)
    families = _VERIFY if r.random() < _VERIFY_SHARE else _CALC
    prompt, code, build = r.choice(families)(r)
    val = _exec(code).split("output: ", 1)[1]        # exact executor output -> grounded
    closer = r.choice(_CLOSERS)
    final = build(val)
    return prompt, code, (f"{final} {closer}".strip() if closer else final)


def render_compute_row(scenario: Scenario) -> dict[str, Any]:
    seed = scenario.seed
    mode = pick_mode(seed)
    prompt, code, final = _scene(seed)
    messages = [
        {"role": "user", "content": prompt},
        tool_call_msg("python", {"code": code}),
        tool_result_msg("python", _exec(code)),
        {"role": "assistant", "content": final},
    ]
    return {
        "id": f"v1_{scenario.slice.lower()}_{seed:09d}",
        "slice": scenario.slice,
        "kind": "compute",
        "reasoning_mode": mode,
        "intent": scenario.intent,
        "plugin_context": scenario.plugin_context,
        "skills_index": [dict(s) for s in scenario.skills_index],
        "selected_skills": [],                       # no skill fits a bare compute ask
        "tool_manifest": list(scenario.tool_manifest),
        "expected_tool_calls": ["python"],
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "narration_grounded"],
        "position_fen": None,
        "stockfish_truth": None,
    }
