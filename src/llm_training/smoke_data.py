from __future__ import annotations


def smoke_records() -> list[dict]:
    history = [{"role": "system", "content": "Follow strict JSON phase contract."}]
    return [
        {
            "id": "router-1",
            "phase": "router",
            "summary": "User asks best move from start position.",
            "history": history,
            "input": {"user_message": "What is best move?"},
            "target": {"type": "tool_call", "tool": "best_move", "args": {}},
            "metadata": {"source": "smoke", "validated": True},
        },
        {
            "id": "narrator-1",
            "phase": "narrator",
            "summary": "Tool found e2e4 as best move.",
            "history": [*history, {"role": "user", "content": "What is best move?"}, {"role": "assistant", "content": {"type": "tool_call", "tool": "best_move", "args": {}}}, {"role": "tool", "content": {"ok": True, "tool": "best_move", "move": "e2e4", "top_k": ["e2e4", "d2d4"]}}],
            "input": {"latest_tool_result": {"ok": True, "tool": "best_move", "move": "e2e4", "top_k": ["e2e4", "d2d4"]}},
            "target": {"type": "narration_reply", "text": "Best move here is e2e4."},
            "metadata": {"source": "smoke", "validated": True},
        },
    ]
