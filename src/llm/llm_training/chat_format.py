"""Chat-template normalization shared by training and serving.

Gemma 4's chat template silently DROPS messages with role="tool": their
content never reaches the rendered prompt. Our SFT corpus and the serve loop
both carry engine results as role="tool", so without this fix the model is
trained and served blind to every tool result — it can only fabricate eval
numbers and moves it cannot see.

The fix maps each tool turn to a user turn wrapped in <tool_result> markers,
which the template renders reliably. It MUST be applied identically at train
(data_pipeline) and serve (model_hf) so the model sees the same shape both
times. Keep the loop/corpus semantics on role="tool"; remap only here, at the
single tokenization boundary.
"""
from __future__ import annotations

TOOL_OPEN = "<tool_result>"
TOOL_CLOSE = "</tool_result>"


def remap_tool_messages(messages: list[dict]) -> list[dict]:
    """Return a new message list with every role="tool" turn rewritten as a
    user turn the chat template will render. Non-tool turns (incl. their
    `train` flags) pass through untouched."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({"role": "user",
                        "content": f"{TOOL_OPEN}\n{m.get('content', '')}\n{TOOL_CLOSE}"})
        else:
            out.append(m)
    return out
