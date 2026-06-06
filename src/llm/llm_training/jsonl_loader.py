from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_chat(path: Path, max_examples: int) -> list[list[dict]]:
    from .system_prompt import build_system
    records: list[list[dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
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
