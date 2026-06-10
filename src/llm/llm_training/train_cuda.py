from __future__ import annotations

import json
import math
import torch
from pathlib import Path
from typing import Any

from .train_gemma4_lora import TrainConfig


def _real_training(config: TrainConfig) -> dict:
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from .data_pipeline import load_jsonl_chat, build_examples, make_batches
    from .optim_sched import build_optimizer, build_scheduler, lora_target_modules
    from .eval_loop import evaluate

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(config.seed)

    from accelerate import Accelerator
    accelerator = Accelerator()
    main_proc = accelerator.is_main_process
    quant_config = _build_quant_config(config)
    device_map = _device_map(config, accelerator.local_process_index, accelerator.num_processes > 1)
    max_memory = _max_memory(device_map)
    if main_proc:
        print(f"Loading model 4bit={quant_config is not None} device_map={device_map} procs={accelerator.num_processes}", flush=True)

    tokenizer = _load_tokenizer(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _strip_compressed_tensors_config(config.model_path)

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_path, local_files_only=True, device_map=device_map, max_memory=max_memory,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, quantization_config=quant_config,
    )

    if accelerator.num_processes > 1:
        freed = _drop_towers(model)
        if main_proc:
            print(f"DDP: freed unused towers {freed} to fit the T4", flush=True)

    # Freeze the base + enable input grads for LoRA + gradient checkpointing —
    # needed for both 4-bit and bf16 (unquantized) bases.
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=lora_target_modules(config.lora_targets),
        modules_to_save=None,
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)", flush=True)

    train_examples = _materialize(config.data_path, config, tokenizer, label="train")
    val_examples = _materialize(config.val_path, config, tokenizer, label="val") if config.val_path else []
    if not train_examples:
        raise ValueError("no training examples produced; check --data-path")
    # Data parallel: each rank trains a DISJOINT shard (true DP, not duplicate work).
    if accelerator.num_processes > 1:
        train_examples = train_examples[accelerator.process_index::accelerator.num_processes]
    # In-loop eval only needs a signal, not all of val — full val at seq 1280 cost
    # ~27 min/eval and ate a third of the wall-clock. Cap it; final quality is
    # judged by eval_routing.py, not this loss.
    eval_subset = val_examples[: config.max_val_examples] if val_examples else []
    val_batches = make_batches(eval_subset, config.batch_size, tokenizer.pad_token_id, shuffle=False, seed=config.seed) if eval_subset else []
    if val_examples:
        print(f"val: using {len(eval_subset)}/{len(val_examples)} examples for in-loop eval", flush=True)

    micro_per_epoch = math.ceil(len(train_examples) / config.batch_size)
    updates_per_epoch = math.ceil(micro_per_epoch / config.grad_accum_steps)
    total_updates = config.max_steps or updates_per_epoch * config.epochs
    warmup = max(1, int(config.warmup_ratio * total_updates))
    print(f"Train ex={len(train_examples)} val ex={len(val_examples)} micro/epoch={micro_per_epoch} updates/epoch={updates_per_epoch} total_updates={total_updates} warmup={warmup}", flush=True)

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    # Wrap for (optional) DDP: num_processes==1 -> no wrap (the proven single-GPU
    # path); ==2 -> a DDP replica per GPU. bnb 4-bit + prepare verified on 2x T4
    # by ddp_probe.py (1.86x throughput).
    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = build_scheduler(optimizer, warmup, total_updates)
    target_device = accelerator.device

    losses, val_history = _train_loop(
        accelerator, model, train_examples, val_batches, optimizer, scheduler, target_device,
        tokenizer.pad_token_id, config, total_updates, evaluate,
    )

    accelerator.wait_for_everyone()
    if main_proc:
        _save(accelerator.unwrap_model(model), tokenizer, config)
    result = {
        "phase": config.phase, "device": config.device, "dry_run": False, "smoke": config.smoke,
        "total_updates": total_updates, "training_started": True, "training_completed": True,
        "final_train_loss": losses[-1] if losses else None, "train_losses": losses,
        "val_losses": val_history, "trainable_params": trainable, "total_params": total,
        "output_dir": str(config.output_dir), "model_path": str(config.model_path),
    }
    if main_proc:
        (config.output_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    return result


def _find_lm_head(model: Any):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "lm_head"):
        return base
    if hasattr(base, "model") and hasattr(base.model, "lm_head"):
        return base.model
    raise AttributeError("lm_head not found")


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-token weighted cross-entropy. All-ones weights == standard mean CE."""
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp(min=1.0)


def fused_masked_loss(model: Any, batch: dict, labels: torch.Tensor,
                      weights: torch.Tensor | None = None, unwrapped: Any = None) -> torch.Tensor:
    # Under DDP `model` is the wrapped module — run the forward through it so grads
    # sync, but swap lm_head on the UNWRAPPED base (the wrapper has no .get_base_model).
    holder = _find_lm_head(unwrapped if unwrapped is not None else model)
    real_head = holder.lm_head
    holder.lm_head = torch.nn.Identity()
    try:
        out = model(**batch)
        hidden = out.logits
    finally:
        holder.lm_head = real_head
    # Keep labels on the same device as hidden (no-op on single GPU; cheap guard).
    labels = labels.to(hidden.device)
    shift_labels = labels[..., 1:].contiguous()
    flat_hidden = hidden[..., :-1, :].contiguous().view(-1, hidden.size(-1))
    flat_labels = shift_labels.view(-1)
    sel = flat_labels != -100
    if not sel.any():
        return hidden.sum() * 0.0
    sel_logits = real_head(flat_hidden[sel]).float()
    sel_labels = flat_labels[sel]
    if weights is None:
        return torch.nn.functional.cross_entropy(sel_logits, sel_labels)
    sel_weights = weights.to(hidden.device)[..., 1:].contiguous().view(-1)[sel]
    return _weighted_ce(sel_logits, sel_labels, sel_weights)


def _train_loop(accelerator, model, train_examples, val_batches, optimizer, scheduler, target_device, pad_id, config, total_updates, evaluate):
    from .data_pipeline import make_batches
    from torch.nn.attention import sdpa_kernel, SDPBackend
    unwrapped = accelerator.unwrap_model(model)
    main_proc = accelerator.is_main_process
    losses: list[float] = []
    val_history: list[dict] = []
    best_val = float("inf")
    update = 0
    epoch = 0
    skipped = 0
    while update < total_updates:
        # +process_index: each rank shuffles its own shard independently.
        batches = make_batches(train_examples, config.batch_size, pad_id, config.shuffle,
                               config.seed + epoch + accelerator.process_index)
        for micro_idx in range(0, len(batches), config.grad_accum_steps):
            optimizer.zero_grad()
            accum_loss = 0.0
            oom = False
            chunk = batches[micro_idx: micro_idx + config.grad_accum_steps]
            for raw in chunk:
                batch = {k: v.to(target_device) for k, v in raw.items()}
                labels = batch.pop("labels")
                weights = batch.pop("weights", None)
                # Mem-efficient SDPA on T4/sm_75 (faster than MATH at seq 1280); MATH
                # stays as an automatic per-op fallback. The fused loss forwards through
                # the wrapped model (so DDP syncs LoRA grads) but swaps lm_head on the
                # unwrapped base.
                try:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        loss = fused_masked_loss(model, batch, labels, weights, unwrapped=unwrapped) / len(chunk)
                    accelerator.backward(loss)
                    accum_loss += loss.item()
                except torch.cuda.OutOfMemoryError:
                    # A rare long sample (seq ~1280) can tip the T4 over during backward.
                    # Drop the whole micro-step (partial grads may be inconsistent) and move
                    # on rather than killing a multi-hour run. NPROC=1 only: no DDP collective
                    # to desync. ~0.1% of samples; cost is one skipped update.
                    oom = True
                    skipped += 1
                    seqlen = int(labels.shape[-1])
                    batch = labels = weights = loss = None
                    optimizer.zero_grad(set_to_none=True)
                    if config.device == "cuda":
                        torch.cuda.empty_cache()
                    if main_proc:
                        print(f"  [OOM-skip] step dropped (seq={seqlen}); total skipped={skipped}", flush=True)
                    break
            if oom:
                continue
            accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            scheduler.step()
            update += 1
            losses.append(accum_loss)
            if main_proc:
                print(f"upd {update}/{total_updates} ep {epoch+1} loss={accum_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
            if val_batches and update % config.eval_every == 0:
                # eval on main only (unwrapped model -> no DDP collective); ranks sync
                # at the barriers so no process deadlocks while main evaluates.
                accelerator.wait_for_everyone()
                if main_proc:
                    vloss = evaluate(unwrapped, val_batches, target_device)
                    val_history.append({"update": update, "val_loss": vloss})
                    print(f"  val_loss={vloss:.4f}", flush=True)
                    if vloss < best_val:
                        best_val = vloss
                        _save(unwrapped, None, config, suffix="best")
                        print(f"  saved best (val_loss={vloss:.4f})", flush=True)
                accelerator.wait_for_everyone()
            if config.device == "cuda":
                torch.cuda.empty_cache()
            if update >= total_updates:
                break
        epoch += 1
    return losses, val_history


def _materialize(path: Path | None, config: TrainConfig, tokenizer: Any, label: str) -> list:
    from .data_pipeline import load_jsonl_chat, build_examples
    if path is None:
        return []
    records = load_jsonl_chat(path, config.max_examples)
    print(f"{label}: read {len(records)} records, tokenizing...", flush=True)
    examples = build_examples(records, tokenizer, config.max_seq_len)
    print(f"{label}: loaded {len(records)} records -> {len(examples)} examples (after mask filter)", flush=True)
    return examples


def _build_quant_config(config: TrainConfig):
    from transformers import BitsAndBytesConfig
    if not (config.load_in_4bit and config.device == "cuda"):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )


def _drop_towers(model: Any) -> list[str]:
    """Free the vision/audio towers — unused in text-only training. Under DDP we
    can't offload them to CPU (accelerate forbids a multi-device map) and the full
    model + seq-1280 activations OOM a T4, so reclaim that memory by dropping them.
    Safe: the text-only forward routes straight to language_model, never the towers."""
    freed: list[str] = []
    for holder in (getattr(model, "model", None), model):
        if holder is None:
            continue
        for attr in ("vision_tower", "audio_tower"):
            if getattr(holder, attr, None) is not None:
                setattr(holder, attr, None)
                freed.append(attr)
    torch.cuda.empty_cache()
    return freed


def _device_map(config: TrainConfig, local_index: int = 0, distributed: bool = False):
    if config.device == "cpu":
        return {"": "cpu"}
    if distributed:
        # accelerate refuses to DDP-prepare a model loaded with a cpu-offload /
        # multi-device map. Put the WHOLE model (incl. frozen vision/audio towers)
        # on this rank's GPU — E2B 4-bit fits a T4 that way (proven by ddp_probe.py).
        return {"": local_index}
    # Always load the LM on ONE GPU; offload the unused multimodal towers to CPU
    # (text-only training). We deliberately do NOT use device_map="auto": naive
    # sharding across 2x T4 put the back half of the model + lm_head on GPU 1 and
    # filled it, so the cross-entropy over Gemma's ~262k vocab had no room to
    # allocate (OOM on GPU 1, even for tiny E2B-4bit). E2B fits a single 16GB T4
    # with the towers on CPU — 4-bit ~2.5GB weights, bf16 ~10GB.
    # Under DDP each process pins its OWN GPU (local_index) — a full replica per
    # GPU (data parallel), NOT the auto-shard that OOM'd.
    return {"": local_index, "model.vision_tower": "cpu", "model.audio_tower": "cpu"}


def _max_memory(device_map):
    """Single-GPU load (we never use device_map='auto'), so accelerate needs no
    per-device cap. Sharding was the source of the GPU-1 OOM; it's gone."""
    return None


def _save(model, tokenizer, config: TrainConfig, suffix: str = "") -> None:
    out = config.output_dir if not suffix else config.output_dir / suffix
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)


def _strip_compressed_tensors_config(model_path: Path) -> None:
    cfg = model_path / "config.json"
    data = json.loads(cfg.read_text())
    qc = data.get("quantization_config")
    if qc and qc.get("quant_method") == "compressed-tensors":
        backup = model_path / "config.json.compressed_backup"
        if not backup.exists():
            backup.write_text(json.dumps(data, indent=2))
        data.pop("quantization_config", None)
        cfg.write_text(json.dumps(data, indent=2))
        print(f"Stripped compressed-tensors config (backup: {backup.name})", flush=True)


def _load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer
    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_path, local_files_only=False)
