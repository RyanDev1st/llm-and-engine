"""CLI entry point for QLoRA fine-tuning. Wires the v1.2 dataset and the
correct base model (gemma4_e2b) into the existing trainer.

Run from repo root:
  python -m llm_training.run_train --smoke        # quick load+step sanity
  python -m llm_training.run_train                # full 3-epoch run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_gemma4_lora import TrainConfig, run_training

LLM_DIR = Path(__file__).resolve().parents[1]   # src/llm
REPO = Path(__file__).resolve().parents[3]       # repo root
DATA = REPO / "data" / "sft"
MODEL = LLM_DIR / "models" / "gemma4_e2b"


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        phase="unified", device="cuda", allow_cuda=True, dry_run=False,
        smoke=args.smoke,
        max_steps=args.max_steps if args.smoke else 0,
        max_examples=32 if args.smoke else 1_000_000,
        batch_size=1, grad_accum_steps=args.grad_accum, epochs=args.epochs,
        max_seq_len=args.max_seq, lora_rank=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=0.05, lora_targets=args.targets, learning_rate=args.lr,
        warmup_ratio=0.03, optimizer="paged_adamw_8bit", eval_every=args.eval_every,
        loss_mask="assistant-only", load_in_4bit=True,
        model_path=MODEL,
        data_path=DATA / "v1_2_train.jsonl",
        val_path=DATA / "v1_2_val.jsonl",
        output_dir=REPO / "runs" / args.output,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=5, dest="max_steps")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=1280, dest="max_seq")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", default="all-linear")
    ap.add_argument("--grad-accum", type=int, default=16, dest="grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=50, dest="eval_every")
    ap.add_argument("--output", default="gemma4_chess")
    args = ap.parse_args()
    if args.smoke:
        args.output = "gemma4_chess_smoke"
    cfg = build_config(args)
    print(f"model={cfg.model_path} data={cfg.data_path}", flush=True)
    print(f"smoke={cfg.smoke} epochs={cfg.epochs} seq={cfg.max_seq_len} rank={cfg.lora_rank} "
          f"targets={cfg.lora_targets} out={cfg.output_dir}", flush=True)
    result = run_training(cfg)
    print(json.dumps({k: v for k, v in result.items() if k not in ("train_losses",)}, indent=2,
                     default=str), flush=True)


if __name__ == "__main__":
    main()
