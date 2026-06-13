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
MODELS = LLM_DIR / "models"
DEFAULT_MODEL = "gemma4_e2b"


def build_config(args: argparse.Namespace) -> TrainConfig:
    rank = args.rank
    targets = args.targets
    grad_accum = args.grad_accum
    if args.smoke:
        rank = args.rank if args.rank != 16 else 4
        targets = args.targets if args.targets != "all-linear" else "qv"
        grad_accum = args.grad_accum if args.grad_accum != 16 else 1
    model = getattr(args, "model", DEFAULT_MODEL)
    cap = getattr(args, "max_examples", None)
    if args.smoke:
        max_examples = 32
    elif cap is not None:
        max_examples = cap
    else:
        max_examples = 1_000_000
    return TrainConfig(
        phase="unified", device="cuda", allow_cuda=True, dry_run=False,
        smoke=args.smoke,
        max_steps=args.max_steps,
        max_examples=max_examples,
        batch_size=1, grad_accum_steps=grad_accum, epochs=args.epochs,
        max_seq_len=args.max_seq, lora_rank=rank, lora_alpha=2 * rank,
        lora_dropout=0.05, lora_targets=targets, learning_rate=args.lr,
        warmup_ratio=0.03, optimizer="paged_adamw_8bit", eval_every=args.eval_every,
        max_val_examples=args.max_val,
        loss_mask="assistant-only", load_in_4bit=not getattr(args, "no_4bit", False),
        engine=getattr(args, "engine", "cuda"),
        model_path=MODELS / model,
        data_path=DATA / "v1_2_train.jsonl",
        val_path=DATA / "v1_2_val.jsonl",
        output_dir=REPO / "runs" / args.output,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=5, dest="max_steps")
    ap.add_argument("--max-examples", type=int, default=None, dest="max_examples",
                    help="cap training rows loaded (quick local de-risk runs); smoke forces 32")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=1664, dest="max_seq")  # v1_2 floor: max row 1655, median 1291; lower truncates finals
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", default="all-linear")
    ap.add_argument("--grad-accum", type=int, default=16, dest="grad_accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=50, dest="eval_every")
    ap.add_argument("--max-val", type=int, default=128, dest="max_val",
                    help="cap val examples for in-loop eval (full val at seq 1280 is ~27 min/eval)")
    ap.add_argument("--output", default="gemma4_chess")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="base model dir under src/llm/models (e.g. gemma4_e4b, gemma4_e2b)")
    ap.add_argument("--no-4bit", action="store_true", dest="no_4bit",
                    help="train in bf16 (no bitsandbytes 4-bit) — fits E2B on one T4 at full seq")
    ap.add_argument("--engine", default="cuda", choices=["cuda", "unsloth"],
                    help="cuda = proven HF path; unsloth = Gemma4-native, ~2x faster + ~70%% less VRAM")
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
