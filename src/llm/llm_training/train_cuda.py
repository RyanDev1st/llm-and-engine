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

    quant_config = _build_quant_config(config)
    device_map = _device_map(config, quant_config is not None)
    print(f"Loading model 4bit={quant_config is not None} device_map={device_map}", flush=True)

    tokenizer = _load_tokenizer(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _strip_compressed_tensors_config(config.model_path)

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_path, local_files_only=True, device_map=device_map,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, quantization_config=quant_config,
    )

    if quant_config is not None:
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
    val_batches = make_batches(val_examples, config.batch_size, tokenizer.pad_token_id, shuffle=False, seed=config.seed) if val_examples else []

    micro_per_epoch = math.ceil(len(train_examples) / config.batch_size)
    updates_per_epoch = math.ceil(micro_per_epoch / config.grad_accum_steps)
    total_updates = config.max_steps or updates_per_epoch * config.epochs
    warmup = max(1, int(config.warmup_ratio * total_updates))
    print(f"Train ex={len(train_examples)} val ex={len(val_examples)} micro/epoch={micro_per_epoch} updates/epoch={updates_per_epoch} total_updates={total_updates} warmup={warmup}", flush=True)

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = build_scheduler(optimizer, warmup, total_updates)
    target_device = torch.device("cuda:0") if config.device == "cuda" else torch.device("cpu")

    losses, val_history = _train_loop(
        model, train_examples, val_batches, optimizer, scheduler, target_device,
        tokenizer.pad_token_id, config, total_updates, evaluate,
    )

    _save(model, tokenizer, config)
    result = {
        "phase": config.phase, "device": config.device, "dry_run": False, "smoke": config.smoke,
        "total_updates": total_updates, "training_started": True, "training_completed": True,
        "final_train_loss": losses[-1] if losses else None, "train_losses": losses,
        "val_losses": val_history, "trainable_params": trainable, "total_params": total,
        "output_dir": str(config.output_dir), "model_path": str(config.model_path),
    }
    (config.output_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    return result


def _find_lm_head(model: Any):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "lm_head"):
        return base
    if hasattr(base, "model") and hasattr(base.model, "lm_head"):
        return base.model
    raise AttributeError("lm_head not found")


def fused_masked_loss(model: Any, batch: dict, labels: torch.Tensor) -> torch.Tensor:
    holder = _find_lm_head(model)
    real_head = holder.lm_head
    holder.lm_head = torch.nn.Identity()
    try:
        out = model(**batch)
        hidden = out.logits
    finally:
        holder.lm_head = real_head
    shift_hidden = hidden[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)
    sel = flat_labels != -100
    if not sel.any():
        return hidden.sum() * 0.0
    sel_hidden = flat_hidden[sel]
    sel_labels = flat_labels[sel]
    sel_logits = real_head(sel_hidden).float()
    return torch.nn.functional.cross_entropy(sel_logits, sel_labels)


def _train_loop(model, train_examples, val_batches, optimizer, scheduler, target_device, pad_id, config, total_updates, evaluate):
    from .data_pipeline import make_batches
    from torch.nn.attention import sdpa_kernel, SDPBackend
    losses: list[float] = []
    val_history: list[dict] = []
    best_val = float("inf")
    update = 0
    epoch = 0
    while update < total_updates:
        batches = make_batches(train_examples, config.batch_size, pad_id, config.shuffle, config.seed + epoch)
        for micro_idx in range(0, len(batches), config.grad_accum_steps):
            optimizer.zero_grad()
            accum_loss = 0.0
            chunk = batches[micro_idx: micro_idx + config.grad_accum_steps]
            for raw in chunk:
                batch = {k: v.to(target_device) for k, v in raw.items()}
                labels = batch.pop("labels")
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    loss = fused_masked_loss(model, batch, labels) / len(chunk)
                loss.backward()
                accum_loss += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            scheduler.step()
            update += 1
            losses.append(accum_loss)
            lr_now = scheduler.get_last_lr()[0]
            print(f"upd {update}/{total_updates} ep {epoch+1} loss={accum_loss:.4f} lr={lr_now:.2e}", flush=True)
            if val_batches and update % config.eval_every == 0:
                vloss = evaluate(model, val_batches, target_device)
                val_history.append({"update": update, "val_loss": vloss})
                print(f"  val_loss={vloss:.4f}", flush=True)
                if vloss < best_val:
                    best_val = vloss
                    _save(model, None, config, suffix="best")
                    print(f"  saved best (val_loss={vloss:.4f})", flush=True)
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


def _device_map(config: TrainConfig, is_4bit: bool):
    if config.device == "cpu":
        return {"": "cpu"}
    if is_4bit:
        return {"": 0, "model.vision_tower": "cpu", "model.audio_tower": "cpu"}
    return {"": 0}


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
