import argparse
import json
from pathlib import Path

from llm_training.data_pipeline import IGNORE_INDEX, build_examples, load_jsonl_chat
from llm_training.eval_routing import VAL, first_turn, gold_tool, mode2_messages
from llm_training.run_train import DATA, build_config


def test_run_train_defaults_to_v1_2_dataset():
    args = argparse.Namespace(
        smoke=False,
        max_steps=5,
        grad_accum=16,
        epochs=3,
        max_seq=1280,
        rank=16,
        targets="all-linear",
        lr=2e-4,
        eval_every=50,
        max_val=128,
        output="gemma4_chess",
    )

    cfg = build_config(args)

    assert cfg.data_path == DATA / "v1_2_train.jsonl"
    assert cfg.val_path == DATA / "v1_2_val.jsonl"
    assert cfg.max_seq_len == 1280
    assert cfg.lora_rank == 16
    assert cfg.lora_targets == "all-linear"
    assert cfg.grad_accum_steps == 16


def test_run_train_smoke_uses_low_memory_defaults():
    args = argparse.Namespace(
        smoke=True,
        max_steps=5,
        grad_accum=16,
        epochs=3,
        max_seq=1280,
        rank=16,
        targets="all-linear",
        lr=2e-4,
        eval_every=50,
        max_val=128,
        output="gemma4_chess",
    )

    cfg = build_config(args)

    assert cfg.max_seq_len == 1280
    assert cfg.lora_rank == 4
    assert cfg.lora_targets == "qv"
    assert cfg.grad_accum_steps == 1


def test_run_train_smoke_keeps_explicit_memory_settings():
    args = argparse.Namespace(
        smoke=True,
        max_steps=5,
        grad_accum=2,
        epochs=3,
        max_seq=768,
        rank=8,
        targets="qkvo",
        lr=2e-4,
        eval_every=50,
        max_val=128,
        output="gemma4_chess",
    )

    cfg = build_config(args)

    assert cfg.max_seq_len == 768
    assert cfg.lora_rank == 8
    assert cfg.lora_targets == "qkvo"
    assert cfg.grad_accum_steps == 2


def test_run_train_bounded_non_smoke_uses_max_steps():
    args = argparse.Namespace(
        smoke=False,
        max_steps=500,
        grad_accum=1,
        epochs=3,
        max_seq=1280,
        rank=4,
        targets="qv",
        lr=2e-4,
        eval_every=50,
        max_val=128,
        output="gemma4_chess_bounded",
    )

    cfg = build_config(args)

    assert cfg.smoke is False
    assert cfg.max_steps == 500
    assert cfg.max_examples == 1_000_000


    row = {
        "messages": [
            {"role": "user", "content": "Please play e4."},
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "Use board tools before claims."},
            {"role": "assistant", "content": "<tool>board_state fields=basic</tool>"},
        ],
        "skills_index": [], "tool_manifest": [], "plugin_context": {},
    }
    messages = row["messages"]

    prompt = first_turn(row)

    assert gold_tool(messages) == "load_skill"
    # system is now rebuilt per-row from the envelope (not a fixed constant)
    assert prompt[0]["role"] == "system" and prompt[0]["content"]
    assert prompt[1] == {"role": "user", "content": "Please play e4."}

    m2 = mode2_messages(row)
    assert m2[0]["role"] == "system"
    assert m2[1:] == messages[:3]


def test_v1_2_loader_builds_assistant_masked_chat_examples(tmp_path):
    row = {
        "id": "loader-smoke",
        "slice": "A",
        "messages": [
            {"role": "user", "content": "Please play e4."},
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "Use board tools before claims."},
            {"role": "assistant", "content": "Played e4."},
        ],
        "plugin_context": {"source": "chess-official"},
        "stockfish_truth": {"best_san": "e4"},
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    tokenizer = WhitespaceTokenizer()

    records = load_jsonl_chat(path, max_examples=1)
    examples = build_examples(records, tokenizer, max_len=2048)

    assert len(records) == 1
    assert records[0][0]["role"] == "system"
    assert len(examples) == 1
    assert any(label != IGNORE_INDEX for label in examples[0]["labels"])


def test_train_false_turn_is_masked_from_loss():
    from llm_training.data_pipeline import tokenize_with_assistant_mask
    base = [
        {"role": "user", "content": "who is winning"},
        {"role": "assistant", "content": "Black is on top"},   # turn-1 answer
        {"role": "user", "content": "best move"},
        {"role": "assistant", "content": "play Nf3 now"},       # turn-2 answer (always trained)
    ]
    trained = [dict(m) for m in base]
    context = [dict(m) for m in base]
    context[1]["train"] = False  # mark turn-1 answer as context-only
    n_trained = sum(l != IGNORE_INDEX for l in tokenize_with_assistant_mask(trained, WhitespaceTokenizer(), 2048)[1])
    n_context = sum(l != IGNORE_INDEX for l in tokenize_with_assistant_mask(context, WhitespaceTokenizer(), 2048)[1])
    # masking turn-1 removes exactly its tokens from the loss; turn-2 still trains
    assert n_context < n_trained
    assert n_context > 0


def test_tokenizer_receives_native_tools_without_private_message_keys():
    from llm_training.data_pipeline import tokenize_with_assistant_mask

    native_tools = [{"type": "function", "function": {"name": "eval", "parameters": {"type": "object", "properties": {}}}}]
    tok = RecordingTokenizer()
    tokenize_with_assistant_mask([
        {"role": "system", "content": "SYS", "_native_tools": native_tools, "_reasoning_mode": "auto"},
        {"role": "user", "content": "score?"},
        {"role": "assistant", "content": "Level."},
    ], tok, 2048)
    assert tok.seen_tools and all(t == native_tools for t in tok.seen_tools)
    assert tok.seen_thinking and all(t is True for t in tok.seen_thinking)
    assert not any("_native_tools" in m for batch in tok.seen_messages for m in batch)
    assert not any("_reasoning_mode" in m for batch in tok.seen_messages for m in batch)


def test_final_nonfact_prose_is_reference_weight_only():
    from llm_training.data_pipeline import FINAL_PROSE_WEIGHT, GROUND_WEIGHT, tokenize_with_assistant_mask

    tok = OffsetTokenizer()
    ids, labels, weights = tokenize_with_assistant_mask([
        {"role": "user", "content": "best move?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "best_move", "arguments": {"depth": 15}}}
        ]},
        {"role": "tool", "name": "best_move", "content": "best_line: Qh5, score: +1.23"},
        {"role": "assistant", "content": "Qh5 is best at +1.23."},
    ], tok, 2048)
    trained = [(labels[i], weights[i], tok.text_for(ids[i])) for i in range(len(labels)) if labels[i] != IGNORE_INDEX]

    assert any(text == "best_move" and weight == 1.0 for _, weight, text in trained)
    assert any(text == "Qh5" and weight == GROUND_WEIGHT for _, weight, text in trained)
    assert any(text == "+1.23" and weight == GROUND_WEIGHT for _, weight, text in trained)
    assert any(text == "is" and weight == FINAL_PROSE_WEIGHT for _, weight, text in trained)


def test_v5_native_sft_targets_fit_profile_sequence_window():
    from transformers import AutoTokenizer
    from llm_dataset.v1.profiles import profile

    p = profile("v5")
    tok = AutoTokenizer.from_pretrained("src/llm/models/gemma4_e2b", trust_remote_code=True)
    records = load_jsonl_chat(Path(str(p.train_path) + ".gz"), max_examples=16)
    examples = build_examples(records, tok, p.max_seq)

    assert len(records) == 16
    assert len(examples) == len(records)
    trained = [sum(label != IGNORE_INDEX for label in ex["labels"]) for ex in examples]
    assert min(trained) > 0


class WhitespaceTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # tokenize_with_assistant_mask renders prefixes with tokenize=False and
        # expects a STRING (it diffs cumulative text); return tokens only when asked.
        if not tokenize:
            return " ".join(f"{m['role']} {m['content']}" for m in messages)
        return {"input_ids": [token for msg in messages for token in self._tokens(msg)]}

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [len(part) for part in text.split()]}

    def _tokens(self, msg):
        return [len(msg["role"]), *[len(part) for part in msg["content"].split()]]


class RecordingTokenizer(WhitespaceTokenizer):
    def __init__(self):
        self.seen_tools = []
        self.seen_thinking = []
        self.seen_messages = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        self.seen_tools.append(kwargs.get("tools"))
        self.seen_thinking.append(kwargs.get("enable_thinking"))
        self.seen_messages.append([dict(m) for m in messages])
        return super().apply_chat_template(messages, tokenize=tokenize,
                                           add_generation_prompt=add_generation_prompt)


class OffsetTokenizer:
    def __init__(self):
        self.vocab = {}
        self.rev = {}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        parts = []
        for m in messages:
            parts.append(m.get("role", ""))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                parts.append(fn.get("name", ""))
                parts.extend(str(v) for v in (fn.get("arguments") or {}).values())
            if m.get("content"):
                parts.append(m["content"])
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids, offsets = [], []
        for match in __import__("re").finditer(r"\S+", text):
            token = match.group(0).strip(".,;:!?()[]{}")
            if token not in self.vocab:
                idx = len(self.vocab) + 1
                self.vocab[token] = idx
                self.rev[idx] = token
            ids.append(self.vocab[token])
            offsets.append((match.start(), match.start() + len(token)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def text_for(self, token_id):
        return self.rev[token_id]
