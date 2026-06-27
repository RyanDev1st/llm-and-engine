"""Harness logic for the early-stop eval — verified with scripted fake models, so
classify/rollout/run are correct without needing trained weights.

v5-native: the scripted models emit Gemma's native tool calls (`call:NAME{…}`); the
rollout parses those (not the old <skill>/<tool> tags)."""
import re

from llm_training.eval_early_stop import build_cases, classify, rollout, run


def _names(system: str, header: str) -> list[str]:
    section = system.split(header, 1)[-1]
    return re.findall(r"^- ([A-Za-z0-9_-]+)", section, re.M)


class ScriptModel:
    """Emits a fixed sequence of generations regardless of input."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.script[self.i] if self.i < len(self.script) else "final answer"
        self.i += 1
        return out


class ABModel:
    """Reads the case's skills/tools out of the system prompt and completes BOTH
    boxes when the prompt signals PLAN (goal on), but stops after the first when it
    signals FAST (goal off) — so run() must show a positive reduction."""

    def generate(self, messages, max_new_tokens, stop):
        system = messages[0]["content"]
        if len(messages) == 2:                      # fresh rollout -> reset
            self.step = 0
            self.skills = _names(system, "AVAILABLE SKILLS")
            self.tools = _names(system, "AVAILABLE TOOLS")
            self.plan = "PLAN" in system
        seq = [f"call:load_skill{{name:{self.skills[0]}}}", f"call:{self.tools[0]}{{depth:12}}"]
        if self.plan:
            seq += [f"call:load_skill{{name:{self.skills[1]}}}", f"call:{self.tools[1]}{{depth:12}}"]
        seq += ["Here is the combined answer."]
        out = seq[self.step] if self.step < len(seq) else "answer"
        self.step += 1
        return out


def test_classify_complete_honest_partial_and_silent():
    case = build_cases(1)[0]
    both = {case.a.tool, case.b.tool}
    assert classify("done", both, case) == "complete"
    assert classify("I can't finish the second part — that tool is disabled.",
                    {case.a.tool}, case) == "honest_partial"
    assert classify("Here's your answer.", {case.a.tool}, case) == "silent_early_stop"


def test_rollout_complete_fires_both_tools():
    case = build_cases(1)[0]
    model = ScriptModel([f"call:load_skill{{name:{case.a.skill}}}", f"call:{case.a.tool}{{depth:12}}",
                         f"call:load_skill{{name:{case.b.skill}}}", f"call:{case.b.tool}{{depth:12}}",
                         "Both parts handled."])
    final, fired, steps = rollout(model, "sys", case)
    assert fired == {case.a.tool, case.b.tool}
    assert steps == 4 and "Both parts" in final
    assert classify(final, fired, case) == "complete"


def test_rollout_silent_early_stop_when_second_tool_skipped():
    case = build_cases(1)[0]
    model = ScriptModel([f"call:load_skill{{name:{case.a.skill}}}", f"call:{case.a.tool}{{depth:12}}",
                         "Here's the first part, hope that helps."])
    final, fired, _ = rollout(model, "sys", case)
    assert fired == {case.a.tool}
    assert classify(final, fired, case) == "silent_early_stop"


def test_rollout_treats_plan_panel_as_continue_not_final():
    # A plan-mode model may emit the <goal>/<plan> panel as a standalone thinking turn.
    # The rollout must not mistake that for the final answer (an instant early-stop).
    case = build_cases(1)[0]
    model = ScriptModel([
        f"<goal>1) {case.a.skill}; 2) {case.b.skill}</goal>\n<plan>\n"
        f"- [ ] a ({case.a.skill})\n- [ ] b ({case.b.skill})\n- [ ] synth (none)\n</plan>",
        f"call:load_skill{{name:{case.a.skill}}}", f"call:{case.a.tool}{{depth:12}}",
        f"call:load_skill{{name:{case.b.skill}}}", f"call:{case.b.tool}{{depth:12}}",
        "Combined answer for both parts."])
    final, fired, _ = rollout(model, "sys", case)
    assert fired == {case.a.tool, case.b.tool}
    assert classify(final, fired, case) == "complete"


def test_run_reports_positive_reduction_when_goal_helps():
    report = run(ABModel(), build_cases(3))
    assert report["goal_on"]["completion_rate"] == 1.0
    assert report["goal_off"]["silent_early_stop_rate"] == 1.0
    assert report["reduction"] == 1.0           # off(1.0) - on(0.0)
