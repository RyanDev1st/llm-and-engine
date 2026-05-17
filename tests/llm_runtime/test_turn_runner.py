from llm_runtime.chess_engine import ChessToolBackend
from llm_runtime.turn_runner import ModelBackend, run_turn


class StubModel(ModelBackend):
    def __init__(self, router_text: str, narrator_text: str) -> None:
        self.router_text = router_text
        self.narrator_text = narrator_text

    def generate(self, phase: str, payload: object) -> str:
        return self.router_text if phase == "router" else self.narrator_text


def test_direct_reply_terminates() -> None:
    model = StubModel('{"type":"direct_reply","text":"hello"}', '{"type":"narration_reply","text":"unused"}')
    result = run_turn([{"role": "system", "content": "x"}], "summary", "hi", model, ChessToolBackend())
    assert result.failures == []
    assert result.messages[-1]["content"]["type"] == "direct_reply"


def test_tool_call_runs_narrator() -> None:
    model = StubModel('{"type":"tool_call","tool":"best_move","args":{}}', '{"type":"narration_reply","text":"Best move here is e2e4."}')
    result = run_turn([{"role": "system", "content": "x"}], "summary", "hi", model, ChessToolBackend())
    assert result.failures == []
    assert result.messages[-1]["content"]["type"] == "narration_reply"
