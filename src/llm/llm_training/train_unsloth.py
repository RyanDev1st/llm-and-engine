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
    sel_hidden = flat_hidden[sel]
    sel_labels = flat_labels[sel]
    sel_weights = (weights.to(hidden.device)[..., 1:].contiguous().view(-1)[sel]
                   if weights is not None else None)
    # Chunk the vocab projection: real_head(sel_hidden) over a 262k vocab in fp32 is a
    # multi-hundred-MB spike on the GPU that holds lm_head (the one that OOMs under
    # naive model-parallel). Projecting in row-chunks bounds that peak; result is
    # identical (weighted sum / total weight).
    CH = 256
    tot_loss = sel_hidden.new_zeros(())
    tot_w = sel_hidden.new_zeros(())
    for i in range(0, sel_hidden.size(0), CH):
        logits_i = real_head(sel_hidden[i:i + CH]).float()
        labels_i = sel_labels[i:i + CH]
        if sel_weights is None:
            tot_loss = tot_loss + F.cross_entropy(logits_i, labels_i, reduction="sum")
            tot_w = tot_w + labels_i.numel()
        else:
            w_i = sel_weights[i:i + CH]
            tot_loss = tot_loss + (F.cross_entropy(logits_i, labels_i, reduction="none") * w_i).sum()
            tot_w = tot_w + w_i.sum()
    return tot_loss / tot_w.clamp(min=1.0)


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


def _free_multimodal_towers(model) -> None:
    """We train TEXT only, but FastModel loads the full Gemma-4 multimodal stack — the
    vision + audio encoders sit in VRAM unused (~GBs). Detach them so that memory is
    reclaimed on BOTH shards. Safe: with no pixel/audio inputs the text forward never
    enters those branches. Defensive — any failure just leaves them loaded. Also prints
    the top-level module names so if the towers are named differently we can target them."""
    import gc
    targets = ("vision_tower", "audio_tower", "vision_model", "audio_model", "vision",
               "audio", "multi_modal_projector", "audio_projector", "vision_projector",
               "embed_vision", "embed_audio")
    roots = []
    seen = set()
    cur = model
    for _ in range(3):  # model -> .model -> .model.model
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        roots.append(cur)
        cur = getattr(cur, "model", None)
    freed = []
    for root in roots:
        try:
            names = [n for n, _ in root.named_children()]
        except Exception:
            continue
        for name in names:
            if name in targets:
                try:
                    setattr(root, name, None)
                    freed.append(f"{type(root).__name__}.{name}")
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[mem] freed multimodal towers: {freed or 'none found'}", flush=True)
    # one-time structure hint so we can target exact names if 'none found'
    try:
        top = [n for n, _ in model.named_children()]
        inner = getattr(model, "model", None)
        sub = [n for n, _ in inner.named_children()] if inner is not None else []
        print(f"[mem] module tree top={top} inner={sub}", flush=True)
    except Exception:
        pass


def _flush_mem() -> None:
    """Release cached/garbage GPU memory — called after a heavy save/eval so a long
    multi-session run doesn't accumulate fragmentation toward an OOM."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_ckpt(model, optimizer, scheduler, update, epoch, best_val, config, tag="checkpoint") -> None:
    """Persist adapter + trainer state so a crash / 12h timeout / next session (even a
    different Kaggle account) can resume. Adapter save is the critical part; optimizer
    state is best-effort (resume degrades to a warm-start if it can't reload)."""
    out = config.output_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))                       # adapter weights (critical)
    state = {"update": update, "epoch": epoch, "best_val": best_val, "seed": config.seed}
    try:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict()
    except Exception as exc:                              # paged 8-bit state can be odd
        print(f"  [ckpt] optimizer/scheduler state not saved ({exc}); resume will warm-start", flush=True)
    torch.save(state, out / "trainer_state.pt")
    print(f"  [ckpt] saved {tag} @ update {update}", flush=True)
    _flush_mem()


def _load_ckpt(model, optimizer, scheduler, config) -> tuple[int, int, float]:
    """Resume from output_dir/checkpoint if --resume. Returns (start_update, start_epoch,
    best_val). Adapter reload FAILURE raises (a multi-day run must NOT silently restart
    from zero); optimizer reload failure only warm-starts (loud warning)."""
    if not getattr(config, "resume", False):
        return 0, 0, float("inf")
    ckpt = config.output_dir / "checkpoint"
    state_path = ckpt / "trainer_state.pt"
    if not state_path.exists():
        print(f"  [resume] no checkpoint at {ckpt} — starting fresh", flush=True)
        return 0, 0, float("inf")
    # adapter weights (critical) — fail loud if the format doesn't load
    from peft import set_peft_model_state_dict
    adapter_sd = None
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        p = ckpt / fname
        if p.exists():
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                adapter_sd = load_file(str(p))
            else:
                adapter_sd = torch.load(str(p), map_location="cpu")
            break
    if adapter_sd is None:
        raise FileNotFoundError(f"[resume] {ckpt} has trainer_state but no adapter weights — refusing to restart from zero")
    set_peft_model_state_dict(model, adapter_sd)
    state = torch.load(state_path, map_location="cpu")
    try:
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
    except Exception as exc:
        print(f"  [resume] optimizer/scheduler not restored ({exc}); warm-starting from saved adapter", flush=True)
    su, se, bv = state.get("update", 0), state.get("epoch", 0), state.get("best_val", float("inf"))
    print(f"  [resume] continuing from update {su} (epoch {se}, best_val {bv:.4f})", flush=True)
    _flush_mem()
    return su, se, bv


def _loop(model, train_examples, val_batches, optimizer, scheduler, device, pad_id, config,
          total_updates, start_update=0, start_epoch=0, best_val=float("inf")) -> tuple:
    import time
    from .data_pipeline import make_batches
    losses: list[float] = []
    val_history: list[dict] = []
    update, epoch = start_update, start_epoch
    save_every = config.save_every or config.eval_every  # crash-safety cadence
    _t_prev = time.time()
    try:
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
                _now = time.time()
                _dt = _now - _t_prev
                _t_prev = _now
                _eta_h = _dt * (total_updates - update) / 3600.0
                print(f"upd {update}/{total_updates} ep {epoch+1} loss={accum_loss:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {_dt:.1f}s/step eta~{_eta_h:.1f}h", flush=True)
                if val_batches and update % config.eval_every == 0:
                    vloss = _evaluate(model, val_batches, device)
                    val_history.append({"update": update, "val_loss": vloss})
                    print(f"  val_loss={vloss:.4f}", flush=True)
                    if vloss < best_val:
                        best_val = vloss
                        model.save_pretrained(str(config.output_dir / "best"))
                        print(f"  saved best (val_loss={vloss:.4f})", flush=True)
                        _flush_mem()           # release memory once a best is banked
                if update % save_every == 0:   # crash/timeout-safe resumable checkpoint
                    _save_ckpt(model, optimizer, scheduler, update, epoch, best_val, config)
                torch.cuda.empty_cache()
                if update >= total_updates:
                    break
            epoch += 1
    except (KeyboardInterrupt, Exception) as exc:
        # 12h cutoff / OOM / disconnect: persist progress so the next session resumes.
        print(f"\n[interrupted: {type(exc).__name__}: {exc}] saving checkpoint before exit...", flush=True)
        try:
            _save_ckpt(model, optimizer, scheduler, update, epoch, best_val, config)
        except Exception as save_exc:
            print(f"[checkpoint-on-exit FAILED: {save_exc}]", flush=True)
        raise
    return losses, val_history


def run_unsloth_training(config: TrainConfig) -> dict:
    from unsloth import FastModel  # MUST precede transformers import for patches
    from .train_cuda import _materialize
    from .optim_sched import build_optimizer, build_scheduler

    # Multi-GPU (e.g. Kaggle 2xT4): SHARD the model across GPUs (model-parallel) so
    # E4B @ seq 1664 fits where one 16 GB T4 cannot (forward alone OOMs single-card).
    # This is device_map="balanced" — NOT DDP, which REPLICATES the model per GPU and
    # OOMs (memory ddp-not-viable-e2b-t4). On 1 GPU, omit device_map so Unsloth uses
    # its normal single-device placement.
    n_gpus = torch.cuda.device_count()
    load_kwargs = dict(model_name=str(config.model_path), max_seq_length=config.max_seq_len,
                       load_in_4bit=config.load_in_4bit, dtype=None)
    if n_gpus > 1:
        load_kwargs["device_map"] = "balanced"
        print(f"[unsloth] {n_gpus} GPUs -> device_map='balanced' (model-parallel shard, not DDP)", flush=True)
    with _capture_load_warnings() as load_log:
        model, processor = FastModel.from_pretrained(**load_kwargs)
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
    # Reclaim the unused vision/audio encoders BEFORE LoRA-wrapping (we train text only).
    try:
        _free_multimodal_towers(model)
    except Exception as exc:
        print(f"[mem] tower-free skipped ({exc})", flush=True)
    flags = _TARGET_FLAGS.get(config.lora_targets, _TARGET_FLAGS["attn-only"])
    model = FastModel.get_peft_model(
        model, r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
        bias="none", random_state=config.seed, finetune_vision_layers=False,
        finetune_language_layers=True, use_gradient_checkpointing="unsloth", **flags)
    # No KV cache during training (we never generate here) — frees activation memory
    # at long seq. Gradient checkpointing already implies this, but set it explicitly.
    for cfg_obj in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg_obj is not None:
            cfg_obj.use_cache = False

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
    # Resume from a prior session's checkpoint (Kaggle 12h ceiling / account switch).
    start_update, start_epoch, best_val = _load_ckpt(model, optimizer, scheduler, config)
    if start_update >= total_updates:
        print(f"[unsloth] already at/over target ({start_update}/{total_updates}); nothing to do. "
              "Raise --max-steps to train further.", flush=True)
    losses, val_history = _loop(model, train_examples, val_batches, optimizer, scheduler,
                                device, tokenizer.pad_token_id, config, total_updates,
                                start_update=start_update, start_epoch=start_epoch, best_val=best_val)

    model.save_pretrained(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))
    result = {"engine": "unsloth", "total_updates": total_updates, "training_completed": True,
              "final_train_loss": losses[-1] if losses else None, "train_losses": losses,
              "val_losses": val_history, "output_dir": str(config.output_dir)}
    (config.output_dir / "train_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
