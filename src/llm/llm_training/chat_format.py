"""Chat-template normalization shared by training and serving.

Gemma 4's chat template silently DROPS messages with role="tool": their
content never reaches the rendered prompt. Our SFT corpus and the serve loop
both carry engine results as role="tool", so without this fix the model is
trained and served blind to every tool result — it can only fabricate eval
numbers and moves it cannot see.

The fix maps each tool turn to a user turn wrapped in <tool_result> markers,
which the template renders reliably. It MUST be applied identically at train
(data_pipeline) and at EVERY serve path — model_hf (apply_chat_template) AND
model_gguf (llama.cpp create_chat_completion uses the GGUF's embedded Gemma
template, which drops role="tool" the same way) — so the model sees the same
shape it trained on. A serve path that forgets this leaves the model blind to
its tool results: it fabricates outcomes and its tool-call format degrades to
pretrained Gemma tokens. Keep the loop/corpus semantics on role="tool"; remap
only here, at the single tokenization boundary.
"""
from __future__ import annotations

TOOL_OPEN = "<tool_result>"
TOOL_CLOSE = "</tool_result>"

# v4.1 hybrid: native reasoning is enabled via the enable_thinking SIGNAL (a <|think|>
# in the system turn), NOT by injecting a native <|channel>thought into the row —
# Gemma's template STRIPS native thought from completed assistant turns, so an injected
# thought would vanish. We keep the custom <think> (masked from loss in data_pipeline,
# never trained) to position the action after a reasoning step; the model's own native
# reasoning fills the slot at serve. See probe_hybrid_thinking.


def wants_thinking(messages: list[dict]) -> bool:
    """True if any assistant turn carries a thought (custom or native) — drives the
    per-row enable_thinking so train and serve agree (fast rows have none -> off)."""
    return any(m.get("role") == "assistant"
               and ("<think>" in (m.get("content") or "") or "<|channel>thought" in (m.get("content") or ""))
               for m in messages)


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
