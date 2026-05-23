"""The spec section-4 turn loop: decide -> (execute tool) -> narrate.

The model uses the role of the latest message to pick Mode 1 vs Mode 2; we keep
the same system prompt across phases. A `ModelBackend` just needs generate()."""
from __future__ import annotations

from typing import Protocol

from llm_training.system_prompt import SYSTEM_PROMPT

from .tools import ToolExecutor


class ModelBackend(Protocol):
    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str:
        ...


class CoachLoop:
    def __init__(self, model: ModelBackend, executor: ToolExecutor) -> None:
        self.model = model
        self.executor = executor

    def respond(self, history: list[dict], user_message: str) -> dict:
        """history: prior user/assistant/tool turns (no system). Returns the
        new turns plus display fields (tool_call, tool_result, reply)."""
        convo = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                 {"role": "user", "content": user_message}]
        decision = self.model.generate(convo, max_new_tokens=64, stop=["</tool>"]).strip()

        if decision.startswith("<tool>"):
            if not decision.endswith("</tool>"):
                decision += "</tool>"
            tool_result = self.executor.execute(decision)
            convo += [{"role": "assistant", "content": decision},
                      {"role": "tool", "content": tool_result}]
            reply = self.model.generate(convo, max_new_tokens=160, stop=[]).strip()
            new_turns = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": decision},
                {"role": "tool", "content": tool_result},
                {"role": "assistant", "content": reply},
            ]
            return {"reply": reply, "tool_call": decision, "tool_result": tool_result, "turns": new_turns}

        new_turns = [{"role": "user", "content": user_message},
                     {"role": "assistant", "content": decision}]
        return {"reply": decision, "tool_call": None, "tool_result": None, "turns": new_turns}
