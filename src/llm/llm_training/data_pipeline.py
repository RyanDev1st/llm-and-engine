from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch

IGNORE_INDEX = -100
GROUND_WEIGHT = 5.0  # loss multiplier on "fact" tokens (eval numbers + SAN moves)
# These are the tokens the model must COPY from the tool result, not invent.
# Up-weighting them stops fabrication being ~free (a wrong move is 1-2 tokens of
# a ~30-token narration, so plain mean loss barely penalizes it).
_FACT = re.compile(r"[+-]?\d+\.\d{2}|O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?")

# v5-native: we train IN Gemma's own native format (single-token <|tool_call> 48/49,
# <|channel> 100/101, etc.), so there is NO custom prior to fight — FORMAT_WEIGHT and the
# _CONTROL/_THINK regex machinery are GONE. The environment-injected tool RESPONSE is masked
# by token id (it lives in a role="tool" turn, but the template leaves a dangling open marker
# in the preceding assistant delta). The native thinking channel, when present (plan-mode rows
# carry the <goal>/<plan> there), IS trained — it's the model's own output, not a stub.
TOOL_RESPONSE_IDS = {50, 51}   # <|tool_response> / <tool_response|> — env data, never trained


def _fact_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _FACT.finditer(text)]


def _overlaps(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    s, e = offset
    if e <= s:  # special/empty token has no surface span
        return False
    return any(s < se and ss < e for ss, se in spans)


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
            reasoning_mode=obj.get("reasoning_mode", ""),
        )
        body = [m for m in msgs if m.get("role") != "system"]
        records.append([{"role": "system", "content": system}, *body])
        if len(records) >= max_examples:
            break
    return records


def tokenize_with_assistant_mask(
    messages: list[dict], tokenizer: Any, max_len: int
) -> tuple[list[int], list[int], list[float]]:
    # Render each cumulative prefix to TEXT (cheap) and tokenize only the new
    # delta once per turn (O(n) vs the old O(n^2) full re-tokenize). Also emit a
    # per-token loss weight: fact tokens (eval numbers, SAN moves) in assistant
    # turns get GROUND_WEIGHT so the model is penalized for fabricating them.
    #
    # v5-native: NO remap — role="tool" survives the native template (it folds into a
    # <|tool_response> block after the assistant's structured tool_calls). Train with
    # enable_thinking=False: we never train the system <|think|> signal or shallow think
    # stubs; the base's native reasoning is invoked at serve. (Plan-mode rows still carry
    # a real plan in the reasoning channel — that renders regardless and IS trained.)
    input_ids: list[int] = []
    labels: list[int] = []
    weights: list[float] = []
    prev_text = ""
    for i, msg in enumerate(messages):
        try:
            text = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False,
                                                 add_generation_prompt=False, enable_thinking=False)
        except TypeError:  # older template without the kwarg
            text = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        except Exception:
            text = _fallback_render(messages[: i + 1])
        delta_text = text[len(prev_text):]
        prev_text = text
        # A turn marked train:false is CONTEXT only (e.g. a prior conversational
        # turn in a multi-turn row) — keep it in the prompt but mask it from loss.
        assistant = msg.get("role") == "assistant" and msg.get("train", True)
        try:
            enc = tokenizer(delta_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except Exception:  # slow tokenizer / no offsets -> fall back to weight 1.0
            enc = tokenizer(delta_text, add_special_tokens=False)
            offsets = None
        delta = enc["input_ids"]
        fact_spans = _fact_spans(delta_text) if (assistant and offsets) else []
        for j, tid in enumerate(delta):
            input_ids.append(tid)
            # Train assistant-generated tokens (tool calls, native thinking channel, final
            # answer); mask the env-injected tool-response marker that lands in this delta.
            if assistant and tid not in TOOL_RESPONSE_IDS:
                labels.append(tid)
                w = GROUND_WEIGHT if (offsets and _overlaps(offsets[j], fact_spans)) else 1.0
                weights.append(w)
            else:
                labels.append(IGNORE_INDEX)
                weights.append(0.0)
        if len(input_ids) >= max_len:
            break
    return input_ids[:max_len], labels[:max_len], weights[:max_len]


def _fallback_render(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)


def build_examples(records: list[list[dict]], tokenizer: Any, max_len: int) -> list[dict]:
    out: list[dict] = []
    total = len(records)
    for i, msgs in enumerate(records):
        ids, labs, wts = tokenize_with_assistant_mask(msgs, tokenizer, max_len)
        if any(lab != IGNORE_INDEX for lab in labs):
            out.append({"input_ids": ids, "labels": labs, "weights": wts})
        if (i + 1) % 2000 == 0:
            print(f"  tokenized {i + 1}/{total} -> {len(out)} kept", flush=True)
    return out


def collate_batch(items: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in items)
    input_ids = torch.full((len(items), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
    weights = torch.zeros((len(items), max_len), dtype=torch.float)
    attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
        weights[i, :n] = torch.tensor(x.get("weights", [1.0] * n), dtype=torch.float)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}


def make_batches(examples: list[dict], batch_size: int, pad_token_id: int, shuffle: bool, seed: int) -> list[dict]:
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    batches = []
    for i in range(0, len(idx), batch_size):
        chunk = [examples[j] for j in idx[i:i + batch_size]]
        batches.append(collate_batch(chunk, pad_token_id))
    return batches
