from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

IGNORE_INDEX = -100


def load_jsonl_chat(path: Path, max_examples: int) -> list[list[dict]]:
    from llm_dataset.v1.jsonl_io import read_rows  # transparent .jsonl/.jsonl.gz

    from .system_prompt import build_system
    records: list[list[dict]] = []
    for obj in read_rows(path):
        msgs = obj.get("messages")
        if not (isinstance(msgs, list) and msgs):
            continue
        # The harness contract is rendered per-row from the envelope so the
        # model conditions on the exact skills/tools it is allowed to use.
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


def tokenize_with_assistant_mask(messages: list[dict], tokenizer: Any, max_len: int) -> tuple[list[int], list[int]]:
    # Render each cumulative prefix to TEXT (cheap) and tokenize only the new
    # delta text once per turn. The old version re-tokenized the whole growing
    # prefix — incl. the large progressive-disclosure system prompt — on every
    # message, which is O(n^2) tokens per row and hangs on big corpora.
    input_ids: list[int] = []
    labels: list[int] = []
    prev_text = ""
    for i, msg in enumerate(messages):
        try:
            text = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        except Exception:
            text = _fallback_render(messages[: i + 1])
        delta_text = text[len(prev_text):]
        prev_text = text
        delta = tokenizer(delta_text, add_special_tokens=False)["input_ids"]
        if msg.get("role") == "assistant":
            input_ids.extend(delta)
            labels.extend(delta)
        else:
            input_ids.extend(delta)
            labels.extend([IGNORE_INDEX] * len(delta))
        if len(input_ids) >= max_len:
            break
    return input_ids[:max_len], labels[:max_len]


def _fallback_render(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)


def build_examples(records: list[list[dict]], tokenizer: Any, max_len: int) -> list[dict]:
    out: list[dict] = []
    total = len(records)
    for i, msgs in enumerate(records):
        ids, labs = tokenize_with_assistant_mask(msgs, tokenizer, max_len)
        if any(lab != IGNORE_INDEX for lab in labs):
            out.append({"input_ids": ids, "labels": labs})
        if (i + 1) % 2000 == 0:
            print(f"  tokenized {i + 1}/{total} -> {len(out)} kept", flush=True)
    return out


def collate_batch(items: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in items)
    input_ids = torch.full((len(items), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_batches(examples: list[dict], batch_size: int, pad_token_id: int, shuffle: bool, seed: int) -> list[dict]:
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    batches = []
    for i in range(0, len(idx), batch_size):
        chunk = [examples[j] for j in idx[i:i + batch_size]]
        batches.append(collate_batch(chunk, pad_token_id))
    return batches
