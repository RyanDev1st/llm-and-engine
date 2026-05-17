from __future__ import annotations

from llm_runtime.chess_engine import ChessToolBackend
from llm_runtime.turn_runner import ModelBackend, run_turn


class StubModel(ModelBackend):
    def __init__(self, router_text: str, narrator_text: str) -> None:
        self.router_text = router_text
        self.narrator_text = narrator_text

    def generate(self, phase: str, payload: object) -> str:
        return self.router_text if phase == "router" else self.narrator_text


def evaluate_stub_turns() -> dict:
    model = StubModel('{"type":"tool_call","tool":"best_move","args":{}}', '{"type":"narration_reply","text":"Best move here is e2e4."}')
    result = run_turn([{"role": "system", "content": "Follow strict JSON phase contract."}], "start position", "best move?", model, ChessToolBackend())
    return {
        "message_count": len(result.messages),
        "failure_count": len(result.failures),
        "ok": len(result.failures) == 0,
    }
