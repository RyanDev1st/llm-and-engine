"""Unsloth training engine (opt-in via --engine unsloth) — Gemma 4 native, ~2x
faster + ~70% less VRAM than the stock HF path, so E2B@seq1280 fits a T4 with
room (the DDP wall is gone). The proven HF engine in train_cuda.py is untouched
and remains the fallback.

We keep OUR validated logic: the 5x grounding-weighted loss and the train:false
multi-turn masking (both live in data_pipeline). Unsloth supplies only the fused
kernels + memory — its speedups are in the decoder layers, orthogonal to our
lm_head-swap loss, so the two compose. Unsloth's own fused cross-entropy is NOT
used (it can't apply per-token weights); we compute the weighted CE ourselves.

Single-GPU only (no accelerate/DDP here) — Unsloth's memory win removes the need.
UNVALIDATED until the 4h-account smoke: model load + lm_head locate on Unsloth's
module tree is the one integration point a CPU box can't check.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import re
from typing import Any

import torch
import torch.nn.functional as F

from .train_gemma4_lora import TrainConfig

# attn-only -> attention adapters only (mirrors lora_target_modules("attn-only")).
_TARGET_FLAGS = {
    "attn-only": dict(finetune_attention_modules=True, finetune_mlp_modules=False),
    "qv": dict(finetune_attention_modules=True, finetune_mlp_modules=False),
    "all-linear": dict(finetune_attention_modules=True, finetune_mlp_modules=True),
}

# Weight names that are SUPPOSED to be created at load (the LoRA adapters + any
# fresh head/norm PEFT adds) — these are not the anomaly.
_OK_REINIT = re.compile(r"lora_|modules_to_save|adapter|\bscore\b", re.IGNORECASE)
# transformers warns like: "Some weights of X were not initialized ... and are
# newly initialized: ['...k_proj...', ...]. You should probably TRAIN this model".
_REINIT_LINE = re.compile(r"newly initialized[:\s]*\[([^\]]*)\]", re.IGNORECASE | re.DOTALL)


class BaseReinitError(RuntimeError):
    """Raised when the loader re-initialized BASE weights (the documented Unsloth
    + QAT-checkpoint anomaly). Training on a partially-random base produces a
    useless adapter — fail loud instead of wasting the GPU slot."""


def _reinit_base_weights(log_text: str) -> list[str]:
    """Parse captured load warnings; return BASE weight names that were newly
    initialized (excludes the expected LoRA/head adapters). Empty == clean load."""
    bad: list[str] = []
    for block in _REINIT_LINE.findall(log_text):
        for name in re.findall(r"['\"]([^'\"]+)['\"]", block):
            if not _OK_REINIT.search(name):
                bad.append(name)
    return bad


@contextlib.contextmanager
def _capture_load_warnings():
    """Collect transformers' modeling-load warnings (where the 'newly initialized'
    notice is emitted) so we can assert the base loaded fully."""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _H(level=logging.WARNING)
    loggers = [logging.getLogger("transformers"), logging.getLogger("transformers.modeling_utils")]
    prev = [(lg, lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.WARNING)
        lg.propagate = True
    try:
        yield records
    finally:
        for lg, lvl, prop in prev:
            lg.removeHandler(handler)
            lg.setLevel(lvl)
            lg.propagate = prop


def _lm_head_holder(model: Any):
    """Find the module owning lm_head across HF/PEFT/Unsloth Gemma-4 trees."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", None)
    candidates = [base, inner, getattr(base, "language_model", None),
                  getattr(inner, "language_model", None)]
    for c in candidates:
        if c is not None and getattr(c, "lm_head", None) is not None:
            return c
    raise AttributeError("lm_head not found on Unsloth model (smoke must confirm tree)")


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_loss(model: Any, batch: dict, labels: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    # Swap lm_head -> Identity so the forward returns hidden states (no 262k-vocab
    # logits materialized); apply the real head only on supervised tokens. This is
    # the train_cuda trick, kept here with an Unsloth-aware head locator.
    holder = _lm_head_holder(model)
    real_head = holder.lm_head
    holder.lm_head = torch.nn.Identity()
    try:
        hidden = model(**batch).logits
    finally:
        holder.lm_head = real_head
    labels = labels.to(hidden.device)
    flat_hidden = hidden[..., :-1, :].contiguous().view(-1, hidden.size(-1))
    flat_labels = labels[..., 1:].contiguous().view(-1)
    sel = flat_labels != -100
    if not sel.any():
        return hidden.sum() * 0.0
    sel_logits = real_head(flat_hidden[sel]).float()
    sel_labels = flat_labels[sel]
    if weights is None:
        return F.cross_entropy(sel_logits, sel_labels)
    sel_weights = weights.to(hidden.device)[..., 1:].contiguous().view(-1)[sel]
    return _weighted_ce(sel_logits, sel_labels, sel_weights)


@torch.no_grad()
def _evaluate(model: Any, batches: list[dict], device: torch.device) -> float:
    model.eval()
    total_loss = total_tok = 0.0
    for raw in batches:
        b = {k: v.to(device) for k, v in raw.items()}
        labels = b.pop("labels")
        b.pop("weights", None)  # val loss unweighted, for comparability
        n = int((labels != -100).sum().item())
        total_loss += _masked_loss(model, b, labels, None).item() * n
        total_tok += n
    model.train()
    return total_loss / total_tok if total_tok else float("nan")


def _loop(model, train_examples, val_batches, optimizer, scheduler, device, pad_id, config, total_updates) -> tuple:
    from .data_pipeline import make_batches
    losses: list[float] = []
    val_history: list[dict] = []
    best_val = float("inf")
    update = epoch = 0
    while update < total_updates:
        batches = make_batches(train_examples, config.batch_size, pad_id, config.shuffle, config.seed + epoch)
        for micro_idx in range(0, len(batches), config.grad_accum_steps):
            optimizer.zero_grad()
            accum_loss = 0.0
            chunk = batches[micro_idx: micro_idx + config.grad_accum_steps]
            for raw in chunk:
                b = {k: v.to(device) for k, v in raw.items()}
                labels = b.pop("labels")
                weights = b.pop("weights", None)
                loss = _masked_loss(model, b, labels, weights) / len(chunk)
                loss.backward()
                accum_loss += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            scheduler.step()
            update += 1
            losses.append(accum_loss)
            print(f"upd {update}/{total_updates} ep {epoch+1} loss={accum_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            if val_batches and update % config.eval_every == 0:
                vloss = _evaluate(model, val_batches, device)
                val_history.append({"update": update, "val_loss": vloss})
                print(f"  val_loss={vloss:.4f}", flush=True)
                if vloss < best_val:
                    best_val = vloss
                    model.save_pretrained(str(config.output_dir / "best"))
                    print(f"  saved best (val_loss={vloss:.4f})", flush=True)
            torch.cuda.empty_cache()
            if update >= total_updates:
                break
        epoch += 1
    return losses, val_history


def run_unsloth_training(config: TrainConfig) -> dict:
    from unsloth import FastModel  # MUST precede transformers import for patches
    from .train_cuda import _materialize
    from .optim_sched import build_optimizer, build_scheduler

    with _capture_load_warnings() as load_log:
        model, processor = FastModel.from_pretrained(
            model_name=str(config.model_path), max_seq_length=config.max_seq_len,
            load_in_4bit=config.load_in_4bit, dtype=None)
    # Guard the documented anomaly: Unsloth re-initializing base k/v on a QAT
    # checkpoint it doesn't fully recognize. Use an Unsloth-published base
    # (unsloth/gemma-3n-E4B-it...) to avoid it; this asserts it didn't happen.
    reinit = _reinit_base_weights("\n".join(load_log))
    if reinit:
        raise BaseReinitError(
            f"loader re-initialized {len(reinit)} BASE weights (e.g. {reinit[:4]}). "
            "Training would start from a partially-random base. Point --model at an "
            "Unsloth-published base (unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit), not a "
            "raw QAT checkpoint.")
    # Gemma4 is multimodal -> FastModel returns a PROCESSOR whose patched __call__
    # treats the first positional arg as `images` (text= is keyword-only). Our
    # text-only pipeline calls tokenizer(delta_text, return_offsets_mapping=True),
    # so unwrap to the raw text tokenizer which has the normal (text, ...) call.
    tokenizer = getattr(processor, "tokenizer", None) or processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    flags = _TARGET_FLAGS.get(config.lora_targets, _TARGET_FLAGS["attn-only"])
    model = FastModel.get_peft_model(
        model, r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
        bias="none", random_state=config.seed, finetune_vision_layers=False,
        finetune_language_layers=True, use_gradient_checkpointing="unsloth", **flags)

    train_examples = _materialize(config.data_path, config, tokenizer, label="train")
    if not train_examples:
        raise ValueError("no training examples produced; check --data-path")
    val_examples = _materialize(config.val_path, config, tokenizer, label="val") if config.val_path else []
    from .data_pipeline import make_batches
    eval_subset = val_examples[: config.max_val_examples]
    val_batches = make_batches(eval_subset, config.batch_size, tokenizer.pad_token_id, False, config.seed) if eval_subset else []

    micro_per_epoch = math.ceil(len(train_examples) / config.batch_size)
    updates_per_epoch = math.ceil(micro_per_epoch / config.grad_accum_steps)
    total_updates = config.max_steps or updates_per_epoch * config.epochs
    warmup = max(1, int(config.warmup_ratio * total_updates))
    print(f"[unsloth] train_ex={len(train_examples)} total_updates={total_updates} warmup={warmup}", flush=True)

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = build_scheduler(optimizer, warmup, total_updates)
    device = torch.device("cuda:0")
    losses, val_history = _loop(model, train_examples, val_batches, optimizer, scheduler,
                                device, tokenizer.pad_token_id, config, total_updates)

    model.save_pretrained(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))
    result = {"engine": "unsloth", "total_updates": total_updates, "training_completed": True,
              "final_train_loss": losses[-1] if losses else None, "train_losses": losses,
              "val_losses": val_history, "output_dir": str(config.output_dir)}
    (config.output_dir / "train_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
