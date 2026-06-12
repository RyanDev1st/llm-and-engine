"""StagedLoop: the serve-time thinking harness. One Controller model call per step
(verify goal -> emit one tool or DONE); a deterministic coverage set force-routes any
required-but-missing tool so compound requests are guaranteed; then one Narrator call
writes the grounded reply. Same return shape as CoachLoop.respond, plus `trace`."""
from __future__ import annotations

from ..inference import _build_window, contains_tool_call, _fallback_reply
from ..tool_hints import matched_calls
from ..toolfmt import parse_call
from .parse import parse_controller
from .prompts import (board_facts, build_controller_system, build_narrator_system,
                      controller_user, narrator_user)

MAX_STEPS = 10


def _name(call: str) -> str:
    return parse_call(call)[0] or ""


class StagedLoop:
    def __init__(self, model, executor, agent_overlay: str = "", plugin_context=None, window=None) -> None:
        self.model = model
        self.executor = executor
        self.agent_overlay = agent_overlay
        self.plugin_context = plugin_context
        self.window = window or _build_window(model)

    def run(self, history: list[dict], user_message: str) -> dict:
        goal = user_message
        game_over = self.executor.game.over_status()
        coverage = {} if game_over else matched_calls(user_message)   # tool -> canonical call
        required = list(coverage)
        facts: list[tuple[str, str]] = []
        seen: set[str] = set()
        tool_calls: list[str] = []
        tool_results: list[str] = []
        trace: list[dict] = []

        for _ in range(MAX_STEPS):
            outstanding = [t for t in required if t not in seen]
            system = build_controller_system(self.agent_overlay, self.plugin_context,
                                             user_message, game_over, outstanding)
            convo = [{"role": "system", "content": system},
                     {"role": "user", "content": controller_user(goal, facts, board_facts(self.executor.game), outstanding)}]
            raw = self.model.generate(convo, max_new_tokens=96, stop=["</tool>", "</tool_code>"]).strip()
            trace.append({"stage": "controller", "output": raw[:120]})
            kind, call = parse_controller(raw)
            if kind == "done" or (call is not None and _name(call) in seen):
                if not outstanding:
                    break                              # goal covered (or nothing new) -> narrate
                call = coverage[outstanding[0]]         # backstop: gather a required-but-missing fact
            name = _name(call)
            result = self.executor.execute(call)
            facts.append((name, result))
            seen.add(name)
            tool_calls.append(call)
            tool_results.append(result)
            trace.append({"stage": "execute", "tool": name, "result": result[:120]})

        system_n = build_narrator_system(self.agent_overlay)
        convo_n = [{"role": "system", "content": system_n},
                   {"role": "user", "content": narrator_user(goal, facts)}]
        reply = self.model.generate(convo_n, max_new_tokens=160, stop=[]).strip()
        if contains_tool_call(reply) or not reply:
            reply = _fallback_reply(tool_calls, tool_results)
        trace.append({"stage": "narrator", "output": reply[:120]})

        _kept, ctx = self.window.fit(system_n, history, user_message)
        return {
            "reply": reply,
            "tool_call": tool_calls[-1] if tool_calls else None,
            "tool_result": tool_results[-1] if tool_results else None,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "turns": [{"role": "user", "content": user_message},
                      {"role": "assistant", "content": reply}],
            "context": ctx.as_payload(),
            "trace": trace,
        }
