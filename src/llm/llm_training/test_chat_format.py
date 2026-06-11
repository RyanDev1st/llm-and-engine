"""Guard: tool results MUST survive into the rendered/tokenized prompt.

The original bug — Gemma's template silently dropping role="tool" — produced a
model trained blind to engine output. These tests fail loudly if a tool result
ever stops reaching the model again.
"""
from pathlib import Path

import pytest

from llm_training.chat_format import remap_tool_messages

BASE = Path(__file__).resolve().parents[1] / "models" / "gemma4_e2b"


def test_remap_rewrites_tool_to_user_and_preserves_others():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "play d6"},
        {"role": "assistant", "content": "ok <tool>move san=d6</tool>", "train": False},
        {"role": "tool", "content": "success: d6"},
        {"role": "assistant", "content": "Played d6."},
    ]
    out = remap_tool_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant"]
    assert "success: d6" in out[3]["content"] and "<tool_result>" in out[3]["content"]
    assert out[2].get("train") is False  # non-tool turns pass through untouched
    assert msgs[3]["role"] == "tool"     # input not mutated


@pytest.mark.skipif(not BASE.exists(), reason="local gemma4_e2b tokenizer not present")
def test_tool_result_tokens_survive_real_template():
    from transformers import AutoTokenizer

    from llm_training.data_pipeline import IGNORE_INDEX, tokenize_with_assistant_mask
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "how am I doing?"},
        {"role": "assistant", "content": "Let me check.\n<tool>eval depth=12</tool>"},
        {"role": "tool", "content": "score: -4.24 pawns from white POV"},
        {"role": "assistant", "content": "Black is clearly better here."},
    ]
    ids, labels, weights = tokenize_with_assistant_mask(msgs, tok, max_len=512)
    decoded = tok.decode(ids)
    assert "-4.24" in decoded, "tool result dropped from the prompt — grounding broken"
    # the tool result is context, not a training target: it must be masked
    assert any(lab != IGNORE_INDEX for lab in labels)   # the narration IS trained
    narration_ids = [i for i, lab in zip(ids, labels) if lab != IGNORE_INDEX]
    assert "-4.24" not in tok.decode(narration_ids)     # but the score is not a label
