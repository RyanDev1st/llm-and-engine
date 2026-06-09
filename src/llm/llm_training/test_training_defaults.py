import argparse
import json

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
