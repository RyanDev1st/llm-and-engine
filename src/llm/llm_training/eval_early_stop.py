"""Early-stop reduction eval: does the `<goal>` anchor stop the model giving a
half-answer on a MULTI-STEP request?

The Stage-1 failure a small model makes: a request needs several tools, the model
fires one (or zero), then writes a confident partial answer. This harness rolls
each COMPOUND case (two pure-chess specialists -> two required tools) forward to
the model's final reply against a SCRIPTED executor, measures how many required
tools fired FIRST, and A/Bs the goal anchor ON (plan mode) vs OFF (fast mode).

Per condition:
- silent_early_stop_rate : final before both tools, no blocker named (PRIMARY, lower better)
- completion_rate        : both required tools ran before the final
- honest_partial_rate    : final before both, but the reply names a blocker (acceptable)
- mean_steps             : assistant action steps taken before the final

reduction = silent_early_stop_rate(off) - silent_early_stop_rate(on)   (>0 => goal helps)

Scoring is by OBJECTIVE (both specialist tools' findings gathered), not exact action
match, so a different-but-valid path is not wrongly flagged. Needs the trained
model (noisy on a tiny scout — meaningful at the full E4B run).

v5-native: the model emits Gemma's native tool calls (`<|tool_call>call:NAME{…}<tool_call|>`)
and the plan rides the native thinking channel, so the rollout parses `call:NAME` and
stops on the native markers. Pure-chess: compound cases pair two chess specialists, no
external domains module.

Run after training:  python -m llm_training.eval_early_stop runs/gemma4_chess
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_dataset.v1.renderer.tags import skill_call_msg, tool_call_msg, tool_result_msg  # noqa: E402
from llm_training.system_prompt import build_system  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MAX_STEPS = 8           # rollout cap (matches serve MAX_TOOL_CALLS headroom)
STEP_TOKENS = 96        # enough for one native action, or a short final
PLUGINS: dict = {}      # flat pure-chess catalog has no plugin context

_BLOCKER = re.compile(r"\b(block|can'?t|cannot|unable|couldn'?t|disabled|stuck)\b", re.I)
# native: an action is `…call:NAME{…}…` (load_skill is the skill action).
_CALL = re.compile(r"call:\s*([A-Za-z0-9_][\w-]*)")
_PANEL = re.compile(r"</?(?:goal|plan)>")     # plan-mode panel text (rides the thinking channel)
STOP = ["<tool_call|>", "<turn|>"]            # native end-of-action / end-of-turn markers


class ModelBackend(Protocol):
    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str: ...


@dataclass
class _Dom:
    """A pure-chess specialist: its skill, the tool it owns, and the scripted
    grounded body/result the executor returns. Mirrors the served specialists."""
    skill: str
    description: str
    tool: str
    tool_args: dict
    prompt: str
    body: str
    result: str


# Four chess specialists, drawn from the flat v5 catalog (catalog.SPECIALIST_*). Each
# owns ONE distinct tool so a compound prompt has two clearly-required tools.
CHESS_DOMAINS: list[_Dom] = [
    _Dom("opening-advisor", "What opening this is, or opening plans/theory.",
         "name_opening", {}, "what opening is this",
         "# opening-advisor\nCall name_opening, then report the opening and its plan.",
         "opening: Ruy Lopez, Morphy Defense"),
    _Dom("game-reviewer", "How the user played overall, accuracy, blunders.",
         "find_blunders", {"depth": "required"}, "find my blunders in this game",
         "# game-reviewer\nCall find_blunders, then report the blunders found.",
         "blunders: move 14 Qh5 (best was Nf3, -2.10 pawns)"),
    _Dom("tactical-puzzles", "Give a tactical puzzle to practice.",
         "fetch_puzzle", {}, "give me a tactical puzzle to solve",
         "# tactical-puzzles\nCall fetch_puzzle, then present the puzzle.",
         "puzzle: white to move, mate in 2, theme=back-rank"),
    _Dom("chess-coach", "Analyze the live position and choose moves.",
         "eval", {"depth": "required"}, "how am I doing right now",
         "# chess-coach\nUse board/eval tools before any claim.",
         "eval: +0.80 pawns (White slightly better)"),
]


@dataclass
class Case:
    a: _Dom
    b: _Dom
    prompt: str
    skills_index: list = field(default_factory=list)
    tool_manifest: list = field(default_factory=list)


def _skill_entry(d: _Dom) -> dict:
    return {"name": d.skill, "description": d.description}


def _tool_entry(d: _Dom) -> dict:
    return {"name": d.tool, "description": f"Specialist tool for {d.skill}.",
            "args": d.tool_args, "applies_when": "always"}


def build_cases(n: int) -> list[Case]:
    """n compound cases pairing two distinct chess specialists; up to 4 listed skills."""
    cases: list[Case] = []
    m = len(CHESS_DOMAINS)
    for i in range(n):
        a = CHESS_DOMAINS[i % m]
        b = CHESS_DOMAINS[(i * 7 + 3) % m]
        if b.skill == a.skill:
            b = CHESS_DOMAINS[(i + 1) % m]
        prompt = f"{a.prompt}, and also {b.prompt}"
        index = [_skill_entry(a), _skill_entry(b)]
        for d in CHESS_DOMAINS:
            if len(index) >= 4:
                break
            if d.skill not in (a.skill, b.skill):
                index.append(_skill_entry(d))
        cases.append(Case(a, b, prompt, index, [_tool_entry(a), _tool_entry(b)]))
    return cases


def _execute(name: str, kind: str, case: Case) -> str:
    """Scripted executor: a skill load returns its terse body; a tool returns its
    grounded result. Mirrors the compound renderer exactly."""
    if kind == "skill":
        d = {case.a.skill: case.a, case.b.skill: case.b}.get(name)
        return d.body if d else "error: unknown_skill"
    d = {case.a.tool: case.a, case.b.tool: case.b}.get(name)
    return d.result if d else "error: unknown_tool"


def _skill_arg(out: str) -> str:
    """Pull the skill name from a native load_skill call: `name:<|"|>NAME<|"|>` —
    strip the quote markers liberally so it survives the exact native render."""
    m = re.search(r"name:([^,}]+)", out)
    if not m:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]+", "", m.group(1))


def rollout(model: ModelBackend, system: str, case: Case) -> tuple[str, set, int]:
    """Drive the model to its final reply. Returns (final_text, tools_fired, steps)."""
    convo = [{"role": "system", "content": system}, {"role": "user", "content": case.prompt}]
    fired: set[str] = set()
    steps = 0
    for _ in range(MAX_STEPS):
        out = model.generate(convo, STEP_TOKENS, STOP).strip()
        call = _CALL.search(out)
        if call:
            name = call.group(1)
            steps += 1
            if name == "load_skill":
                skill = _skill_arg(out)
                convo += [skill_call_msg(skill),
                          tool_result_msg("load_skill", _execute(skill, "skill", case))]
            else:
                fired.add(name)
                convo += [tool_call_msg(name, {}),
                          tool_result_msg(name, _execute(name, "tool", case))]
        elif _PANEL.search(out):
            # a thinking-only panel turn (<goal>/<plan> with no call): commit it and KEEP
            # GOING — it is NOT the final answer. Native usually folds the panel into the
            # same turn as the first call (handled above); this is the defensive case.
            convo.append({"role": "assistant", "content": out})
        else:
            return out, fired, steps          # plain final reply
    return "", fired, steps                    # never finalized -> non-complete


def classify(final: str, fired: set, case: Case) -> str:
    required = {case.a.tool, case.b.tool}
    if required <= fired:
        return "complete"
    if final and _BLOCKER.search(final):
        return "honest_partial"
    return "silent_early_stop"


def run(model: ModelBackend, cases: list[Case]) -> dict:
    """A/B the goal anchor: plan mode (on) vs fast mode (off)."""
    report: dict = {}
    for cond, mode in (("goal_on", "plan"), ("goal_off", "fast")):
        counts = {"complete": 0, "honest_partial": 0, "silent_early_stop": 0}
        steps_total = 0
        for c in cases:
            system = build_system(c.skills_index, c.tool_manifest, PLUGINS, reasoning_mode=mode)
            final, fired, steps = rollout(model, system, c)
            counts[classify(final, fired, c)] += 1
            steps_total += steps
        n = max(1, len(cases))
        report[cond] = {
            "n": len(cases),
            "silent_early_stop_rate": counts["silent_early_stop"] / n,
            "completion_rate": counts["complete"] / n,
            "honest_partial_rate": counts["honest_partial"] / n,
            "mean_steps": steps_total / n,
            "counts": counts,
        }
    report["reduction"] = (report["goal_off"]["silent_early_stop_rate"]
                           - report["goal_on"]["silent_early_stop_rate"])
    return report


def _format(adapter, report: dict) -> str:
    on, off = report["goal_on"], report["goal_off"]
    lines = [
        "Parent: none", "", "# Early-stop reduction eval", "", "## Status",
        f"Goal anchor reduces silent early-stops by {report['reduction']:+.1%} "
        f"(off={off['silent_early_stop_rate']:.1%} -> on={on['silent_early_stop_rate']:.1%}).", "",
        "## Scope", f"Adapter: `{adapter}`. {on['n']} compound (two-tool) cases, A/B goal on/off.", "",
        "## Evidence",
    ]
    for cond, d in (("goal ON (plan)", on), ("goal OFF (fast)", off)):
        lines.append(f"- {cond}: complete={d['completion_rate']:.0%} "
                     f"silent_early_stop={d['silent_early_stop_rate']:.0%} "
                     f"honest_partial={d['honest_partial_rate']:.0%} "
                     f"mean_steps={d['mean_steps']:.1f}")
    lines += ["", "## Next",
              "1. If reduction <= ~0, the goal anchor is not earning its tokens — reconsider.",
              "2. Re-run at the full E4B checkpoint (scout numbers are noisy)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    from backend.model_hf import HFModel
    model = HFModel(adapter=adapter, temperature=0.0)
    report = run(model, build_cases(n))
    text = _format(adapter, report)
    from datetime import date
    out = REPO / "docs" / f"{date.today():%Y-%m-%d}-early-stop-eval.md"
    out.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
