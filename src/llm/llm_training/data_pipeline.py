from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch

IGNORE_INDEX = -100
GROUND_WEIGHT = 5.0  # loss multiplier on "fact" tokens (eval numbers + SAN moves)
FINAL_PROSE_WEIGHT = 0.1  # weak reference only; do not overwrite Gemma's natural voice
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
TOOL_RESPONSE_IDS = {50, 51}   # <|tool_response> / <tool_response|> on the E2B tokenizer;
# DEFAULT/reference only — the actual ids are derived from the live tokenizer at train time
# (the E4B/unsloth base could number these differently; a wrong hardcoded id silently UN-masks
# the env-injected tool output and trains the model to FABRICATE tool results). See _tool_response_ids.
_TOOL_RESPONSE_MARKERS = ("<|tool_response>", "<tool_response|>")
_MASKED_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _tool_response_ids(tokenizer: Any) -> set[int]:
    """Tool-response marker ids FOR THIS tokenizer. Each marker must be a SINGLE special
    token; if it isn't (unknown base / broken vocab), fall back to the E2B reference ids so
    masking still fires on the common case rather than silently masking nothing."""
    cached = getattr(tokenizer, "_chess_tr_ids", None)
    if cached is not None:
        return cached
    ids: set[int] = set()
    for marker in _TOOL_RESPONSE_MARKERS:
        try:
            enc = tokenizer(marker, add_special_tokens=False)["input_ids"]
        except Exception:
            continue
        if len(enc) == 1:
            ids.add(enc[0])
    ids = ids or set(TOOL_RESPONSE_IDS)
    try:
        tokenizer._chess_tr_ids = ids
    except Exception:
        pass
    return ids


def _fact_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _FACT.finditer(text)]


def _overlaps(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    s, e = offset
    if e <= s:  # special/empty token has no surface span
        return False
    return any(s < se and ss < e for ss, se in spans)


def _masked_think_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _MASKED_THINK.finditer(text)]


def load_jsonl_chat(path: Path, max_examples: int) -> list[list[dict]]:
    from llm_dataset.v1.jsonl_io import read_rows  # transparent .jsonl/.jsonl.gz

    from .system_prompt import build_system
    from .native_tools import manifest_to_native_tools
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
            include_tools=False,
        )
        native_tools = manifest_to_native_tools(obj.get("tool_manifest", []), include_descriptions=False)
        body = [m for m in msgs if m.get("role") != "system"]
        records.append([{"role": "system", "content": system, "_native_tools": native_tools,
                         "_reasoning_mode": obj.get("reasoning_mode", "")}, *body])
        if len(records) >= max_examples:
            break
    return records


def tokenize_with_assistant_mask(
    messages: list[dict], tokenizer: Any, max_len: int
) -> tuple[list[int], list[int], list[float]]:
    # Render cumulative prefixes and train only assistant deltas. v5-native keeps
    # role="tool" and structured tool_calls in the model's native template. Reasoning
    # traces are input-only context: mask <think> stubs while plan boxes remain trained.
    input_ids: list[int] = []
    labels: list[int] = []
    weights: list[float] = []
    tr_ids = _tool_response_ids(tokenizer)   # derived from THIS tokenizer, not hardcoded
    from .native_tools import template_messages, tools_for_messages
    native_tools = tools_for_messages(messages)
    mode = (messages[0].get("_reasoning_mode", "") if messages else "").strip().lower()
    enable_thinking = mode not in ("", "fast")
    prev_text = ""
    for i, msg in enumerate(messages):
        prefix = template_messages(messages[: i + 1])
        try:
            text = tokenizer.apply_chat_template(prefix, tokenize=False,
                                                 add_generation_prompt=False, enable_thinking=enable_thinking,
                                                 tools=native_tools or None)
        except TypeError:  # older template without the kwarg
            text = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = _fallback_render(prefix)
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
        final_turn = assistant and not msg.get("tool_calls")
        fact_spans = _fact_spans(delta_text) if (assistant and offsets) else []
        masked_think_spans = _masked_think_spans(delta_text) if (assistant and offsets) else []
        for j, tid in enumerate(delta):
            input_ids.append(tid)
            # Train assistant-generated tokens (tool calls, native thinking channel, final
            # answer); mask the env-injected tool-response marker that lands in this delta.
            if assistant and tid not in tr_ids and not (offsets and _overlaps(offsets[j], masked_think_spans)):
                labels.append(tid)
                if offsets and _overlaps(offsets[j], fact_spans):
                    w = GROUND_WEIGHT
                elif final_turn and offsets and offsets[j][1] > offsets[j][0]:
                    w = FINAL_PROSE_WEIGHT
                else:
                    w = 1.0
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
