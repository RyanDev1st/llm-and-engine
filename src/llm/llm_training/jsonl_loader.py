from __future__ import annotations

from pathlib import Path
from typing import Any


def load_jsonl_chat(path: Path, max_examples: int) -> list[list[dict]]:
    from llm_dataset.v1.jsonl_io import read_rows  # transparent .jsonl/.jsonl.gz

    from .system_prompt import build_system
    records: list[list[dict]] = []
    for obj in read_rows(path):
        msgs = obj.get("messages")
        if not (isinstance(msgs, list) and msgs):
            continue
        system = build_system(
            obj.get("skills_index", []),
            obj.get("tool_manifest", []),
            obj.get("plugin_context", {}),
        )
        body = [m for m in msgs if m.get("role") != "system"]
        records.append([{"role": "system", "content": system}, *body])
        if len(records) >= max_examples:
            break
    return records


def render_chat(messages_list: list[list[dict]], tokenizer: Any) -> list[str]:
    rendered: list[str] = []
    for messages in messages_list:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)
        rendered.append(text)
    return rendered


def tokenize_chat(texts: list[str], tokenizer: Any, max_len: int) -> dict:
    if not texts:
        texts = [""]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return encoded
