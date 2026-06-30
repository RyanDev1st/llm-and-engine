from llm_training.data_pipeline import IGNORE_INDEX, tokenize_with_assistant_mask


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(ch) for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, messages, **kwargs):
        out = []
        for msg in messages:
            if msg.get("role") == "system":
                out.append("SYS:" + msg.get("content", "") + "\n")
            elif msg.get("role") == "user":
                out.append("USER:" + msg.get("content", "") + "\n")
            elif msg.get("role") == "assistant":
                if msg.get("reasoning_content"):
                    out.append("<|channel>thought\n" + msg["reasoning_content"] + "<channel|>")
                for call in msg.get("tool_calls") or []:
                    name = call["function"]["name"]
                    out.append(f"<|tool_call>call:{name}{{}}<tool_call|>")
                out.append(msg.get("content", ""))
            elif msg.get("role") == "tool":
                out.append("<|tool_response>" + msg.get("content", "") + "<tool_response|>")
        return "".join(out)


def test_think_reasoning_is_rendered_but_masked_from_loss():
    messages = [
        {"role": "system", "content": "sys", "_reasoning_mode": "think"},
        {"role": "user", "content": "best move?"},
        {"role": "assistant", "reasoning": "<think>inspect board</think>", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "board_state", "arguments": {}}}]},
    ]

    ids, labels, _ = tokenize_with_assistant_mask(messages, TinyTokenizer(), 1000)
    text = "".join(chr(i) for i in ids)
    think_at = text.index("inspect board")
    call_at = text.index("call:board_state")

    assert all(label == IGNORE_INDEX for label in labels[think_at:think_at + len("inspect board")])
    assert any(label != IGNORE_INDEX for label in labels[call_at:call_at + len("call:board_state")])


def test_plan_reasoning_stays_trained():
    messages = [
        {"role": "system", "content": "sys", "_reasoning_mode": "plan"},
        {"role": "user", "content": "audit"},
        {"role": "assistant", "reasoning": "<goal>audit</goal>\n<plan>- [ ] inspect</plan>",
         "content": "", "tool_calls": [{"type": "function", "function": {"name": "board_state", "arguments": {}}}]},
    ]

    ids, labels, _ = tokenize_with_assistant_mask(messages, TinyTokenizer(), 1000)
    text = "".join(chr(i) for i in ids)
    goal_at = text.index("<goal>audit</goal>")

    assert any(label != IGNORE_INDEX for label in labels[goal_at:goal_at + len("<goal>audit</goal>")])
