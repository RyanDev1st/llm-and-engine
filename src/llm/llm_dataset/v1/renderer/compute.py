"""V1_R compute-grounding renderer — the Stage-0 keystone slice.

Lesson (verification-as-tool-use): when the answer hinges on a number, the agent
does NOT compute or assert it in its head (a 4B fabricates arithmetic). It fills
the ONE canonical calculator template (catalog.CALC_TEMPLATE), runs it through the
`python` tool, reads the real stdout, and states the VERDICT grounded in that
value — the way Claude verifies a claim instead of guessing. This is box-auditing
in miniature: a box is a claim ("is the average above 85?"), audited by running a
tool and reading the output, never by asserting. No domain skill fits a plain
compute ask, so the model goes tool-direct (contract: load a skill only when one
fits).

Two shapes, mixed ~70/30 so the model learns BOTH triggers for reaching the tool:
- verify-then-claim (~70%): the user asks a judgment ("am I averaging above 85?");
  the model verifies the number, then asserts the grounded verdict. This is the
  seed of the Stage-1/2 checkbox audit.
- compute-on-request (~30%): the user asks for a raw number; the model computes it
  rather than guessing. Covers the "user wants a figure" trigger.

Grounding is enforced, not hoped for: the tool result is produced by the REAL
executor (backend.sandbox.run_python), so train == serve exactly, and every final
cites only the computed value (a two-decimal token that appears in the tool
output) plus integer/percent context — `narration_grounded` (validate._FACT)
rejects any final stating a number absent from the tool output.
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from backend.sandbox import run_python

from ..catalog import CALC_TEMPLATE
from ..sampler import Scenario
from .leadins import lead
from .thinking import gated_answer, gated_think, pick_mode

_GOAL = "settle this with the real number"
_VERIFY_SHARE = 0.70           # ~70% verify-then-claim, ~30% compute-on-request

# Money operands as strings -> the rendered script is byte-exact; they appear only
# in the script (an action message, never fact-checked) or the user prompt.
_BILLS = ("38.20", "48.50", "59.90", "72.80", "86.40", "95.40", "33.33", "47.25")
_PCTS = (10, 12, 15, 18, 20, 25)
_BASES = (40, 60, 80, 120, 128, 150, 200, 240)
_CLOSERS = ("Want the breakdown?", "Anything else to check?", "Want me to round it?",
            "Need another one run?", "", "", "")


def _wrap(expr: str) -> str:
    """Plug the expression into the ONE canonical calculator template (catalog.
    CALC_TEMPLATE) — every row reuses the identical known-good wrapper and varies
    only the expression, so a weak coder model plug-and-plays it. Two-decimal print
    keeps the output groundable by validate._FACT."""
    return CALC_TEMPLATE.replace("EXPR", expr)


@lru_cache(maxsize=16384)
def _exec(code: str) -> str:
    return run_python(code)            # real executor -> train == serve output


def _ints(r: random.Random, n: int, lo: int, hi: int) -> list[int]:
    return [r.randint(lo, hi) for _ in range(n)]


# --- verify-then-claim families: (prompt, code, build(val_str) -> grounded verdict) ---

def _avg_check(r):
    s = _ints(r, r.choice((3, 4)), 60, 99)
    t = r.choice((70, 75, 80, 85, 90))
    prompt = r.choice((
        f"I scored {', '.join(map(str, s))} — am I averaging at least {t}?",
        f"my marks are {', '.join(map(str, s))}; is my average {t} or higher?",
        f"are {', '.join(map(str, s))} averaging above {t}?"))
    code = _wrap(f"sum({s}) / {len(s)}")

    def build(val):
        v = float(val)
        if abs(v - t) < 0.005:
            return f"Your average is {val} — exactly the {t} mark, not above it."
        return f"Your average is {val}, {'above' if v > t else 'below'} the {t} mark."
    return prompt, code, build


def _budget_check(r):
    sp = [r.choice(_BILLS) for _ in range(3)]
    t = r.choice((120, 150, 180, 200, 250))
    prompt = r.choice((
        f"I spent ${', $'.join(sp)} this week — did I stay under ${t}?",
        f"this week's spends were ${', $'.join(sp)}; under ${t} total?",
        f"do ${', $'.join(sp)} add up to less than ${t}?"))
    code = _wrap(" + ".join(sp))

    def build(val):
        v = float(val)
        return (f"Your total is ${val}, under the ${t} budget." if v <= t
                else f"Your total is ${val}, over your ${t} budget.")
    return prompt, code, build


def _tip_check(r):
    bill, pct, t = r.choice(_BILLS), r.choice(_PCTS), r.choice((8, 10, 12, 15, 20))
    prompt = r.choice((
        f"Is a {pct}% tip on ${bill} more than ${t}?",
        f"would {pct}% on ${bill} come to over ${t}?",
        f"does a {pct}% tip on ${bill} top ${t}?"))
    code = _wrap(f"{pct / 100} * {bill}")

    def build(val):
        v = float(val)
        return (f"That tip works out to ${val}, more than ${t}." if v > t
                else f"That tip is ${val}, not more than ${t}.")
    return prompt, code, build


def _discount_check(r):
    price = r.choice((60, 80, 120, 150, 200, 250))
    pct, t = r.choice(_PCTS), r.choice((50, 75, 100, 150, 180))
    prompt = r.choice((
        f"A ${price} item is {pct}% off — is the final price under ${t}?",
        f"after {pct}% off ${price}, do I pay less than ${t}?",
        f"is ${price} at {pct}% off below ${t}?"))
    code = _wrap(f"{price} * {(100 - pct) / 100}")

    def build(val):
        v = float(val)
        return (f"After the discount it's ${val}, under ${t}." if v <= t
                else f"After the discount it's ${val}, still over ${t}.")
    return prompt, code, build


def _split_check(r):
    bill, n, t = r.choice(_BILLS), r.choice((2, 3, 4, 5, 6)), r.choice((10, 15, 20, 25, 30))
    prompt = r.choice((
        f"Split ${bill} among {n} of us — is each share under ${t}?",
        f"if {n} people split ${bill}, does each pay less than ${t}?",
        f"${bill} divided by {n} — under ${t} each?"))
    code = _wrap(f"{bill} / {n}")

    def build(val):
        v = float(val)
        return (f"Each person pays ${val}, under the ${t} cap." if v <= t
                else f"Each person pays ${val}, over the ${t} cap.")
    return prompt, code, build


def _save_check(r):
    amt, weeks, goal = r.choice(_BILLS), r.choice((6, 8, 10, 12, 16, 20)), r.choice((400, 600, 800, 1000))
    prompt = r.choice((
        f"If I save ${amt} a week for {weeks} weeks, do I reach ${goal}?",
        f"saving ${amt} weekly over {weeks} weeks — enough for ${goal}?",
        f"will ${amt}/week for {weeks} weeks hit ${goal}?"))
    code = _wrap(f"{amt} * {weeks}")

    def build(val):
        v = float(val)
        return (f"You'd save ${val}, enough to clear ${goal}." if v >= goal
                else f"You'd save ${val}, short of the ${goal} goal.")
    return prompt, code, build


# --- compute-on-request families: (prompt, code, build(val_str) -> raw figure) ---

def _calc_make(prompt: str, expr: str, fmt: str):
    return prompt, _wrap(expr), (lambda val: fmt.format(val=val))


def _tip(r):
    bill, pct = r.choice(_BILLS), r.choice(_PCTS)
    p = r.choice((f"What's a {pct}% tip on a ${bill} bill?",
                  f"whats {pct} percent tip on {bill}?"))
    return _calc_make(p, f"{pct / 100} * {bill}", f"A {pct}% tip on that bill is ${{val}}.")


def _percent_of(r):
    pct, base = r.choice(_PCTS), r.choice(_BASES)
    p = r.choice((f"What's {pct}% of {base}?", f"give me {pct} percent of {base}"))
    return _calc_make(p, f"{pct / 100} * {base}", f"{pct}% of {base} is {{val}}.")


def _weekly(r):
    amt, weeks = r.choice(_BILLS), r.choice((4, 6, 8, 10, 12, 16, 20))
    p = r.choice((f"Save ${amt} a week for {weeks} weeks — total?",
                  f"${amt} per week times {weeks} weeks?"))
    return _calc_make(p, f"{amt} * {weeks}", f"Over {weeks} weeks that's ${{val}} saved.")


def _convert(r):
    mi = r.choice((3, 5, 8, 10, 12, 15, 20, 26))
    p = r.choice((f"How many km is {mi} miles?", f"convert {mi} miles to kilometers"))
    return _calc_make(p, f"{mi} * 1.60934", f"{mi} miles is about {{val}} km.")


def _average(r):
    s = _ints(r, r.choice((3, 4)), 55, 99)
    p = r.choice((f"What's the average of {', '.join(map(str, s))}?",
                  f"mean of {', '.join(map(str, s))}?"))
    return _calc_make(p, f"sum({s}) / {len(s)}", "The average is {val}.")


def _total(r):
    s = _ints(r, r.choice((4, 5)), 8, 80)
    p = r.choice((f"Add up {', '.join(map(str, s))}.",
                  f"total of {', '.join(map(str, s))}?"))
    return _calc_make(p, f"sum({s})", "That totals {val}.")


_VERIFY = (_avg_check, _budget_check, _tip_check, _discount_check, _split_check, _save_check)
_CALC = (_tip, _percent_of, _weekly, _convert, _average, _total)


def _scene(seed: int) -> tuple[str, str, str]:
    """(user prompt, python script, final reply with the grounded value/verdict)."""
    r = random.Random(seed * 137 + 11)
    families = _VERIFY if r.random() < _VERIFY_SHARE else _CALC
    prompt, code, build = r.choice(families)(r)
    val = _exec(code).split("output: ", 1)[1]        # exact executor output -> grounded
    final = build(val)
    closer = r.choice(_CLOSERS)
    return prompt, code, (f"{final} {closer}".strip() if closer else final)


def render_compute_row(scenario: Scenario) -> dict[str, Any]:
    seed = scenario.seed
    mode = pick_mode(seed)
    prompt, code, final = _scene(seed)
    think = gated_think(seed, "python", 0, mode=mode, kind="decide", goal=_GOAL, have="")
    call = "\n".join(p for p in (think, lead(seed, "python", 0), f"<tool>python code={code}</tool>") if p)
    ans = gated_answer(seed, _GOAL, mode=mode)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": call},
        {"role": "tool", "content": _exec(code)},
        {"role": "assistant", "content": f"{ans}\n{final}" if ans else final},
    ]
    return {
        "id": f"v1_{scenario.slice.lower()}_{seed:09d}",
        "slice": scenario.slice,
        "kind": "compute",
        "reasoning_mode": mode,
        "intent": scenario.intent,
        "plugin_context": scenario.plugin_context,
        "skills_index": [dict(s) for s in scenario.skills_index],
        "selected_skills": [],                       # no domain skill fits a compute ask
        "tool_manifest": list(scenario.tool_manifest),
        "expected_tool_calls": ["python"],
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "narration_grounded"],
        "position_fen": None,
        "stockfish_truth": None,
    }
