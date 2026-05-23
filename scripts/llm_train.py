from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_training.train_gemma4_lora import TrainConfig, run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma4 LoRA trainer (unified)")
    p.add_argument("--phase", default="unified", choices=["router", "narrator", "unified"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="override total updates (0=use epochs)")
    p.add_argument("--max-examples", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="qv", choices=["all-linear", "attn-only", "qv"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="paged_adamw_8bit", choices=["paged_adamw_8bit", "adamw"])
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--loss-mask", default="assistant-only", choices=["assistant-only", "all"])
    p.add_argument("--model-path", type=Path, default=Path("src/models/gemma4"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/gemma4_lora"))
    p.add_argument("--max-gpu-memory", default="7GiB")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--val-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = TrainConfig(
        phase=a.phase, device=a.device, allow_cuda=a.allow_cuda, dry_run=a.dry_run, smoke=a.smoke,
        max_steps=a.max_steps, max_examples=a.max_examples, batch_size=a.batch_size,
        grad_accum_steps=a.grad_accum_steps, epochs=a.epochs, max_seq_len=a.max_seq_len,
        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        lora_targets=a.lora_targets, learning_rate=a.lr, warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay, grad_clip=a.grad_clip, optimizer=a.optimizer,
        eval_every=a.eval_every, shuffle=not a.no_shuffle, loss_mask=a.loss_mask,
        model_path=a.model_path, output_dir=a.output_dir, max_gpu_memory=a.max_gpu_memory,
        load_in_4bit=not a.no_4bit, data_path=a.data_path, val_path=a.val_path, seed=a.seed,
    )
    result = run_training(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
