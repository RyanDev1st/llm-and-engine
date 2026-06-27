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


def _seq_for(data_stem: str, explicit: int | None) -> int:
    """Train seq ceiling. Honor an explicit --max-seq; else derive it from the data-stem's
    profile (v5 -> 1920 for the bigger flat catalog + plan-mode; v1_2 -> 1664), so the
    right seq can't be silently forgotten and truncate finals. Unknown stems -> 1664."""
    if explicit is not None:
        return explicit
    try:
        from llm_dataset.v1.profiles import profile
        return profile(data_stem).max_seq
    except Exception:
        return 1664


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
        max_seq_len=_seq_for(getattr(args, "data_stem", "v1_2"), args.max_seq),
        lora_rank=rank, lora_alpha=2 * rank,
        lora_dropout=0.05, lora_targets=targets, learning_rate=args.lr,
        warmup_ratio=0.03, optimizer="paged_adamw_8bit", eval_every=args.eval_every,
        max_val_examples=args.max_val,
        loss_mask="assistant-only", load_in_4bit=not getattr(args, "no_4bit", False),
        engine=getattr(args, "engine", "cuda"),
        save_every=getattr(args, "save_every", 0), resume=getattr(args, "resume", False),
        model_path=MODELS / model,
        data_path=DATA / f"{getattr(args, 'data_stem', 'v1_2')}_train.jsonl",
        val_path=DATA / f"{getattr(args, 'data_stem', 'v1_2')}_val.jsonl",
        output_dir=REPO / "runs" / args.output,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=5, dest="max_steps")
    ap.add_argument("--max-examples", type=int, default=None, dest="max_examples",
                    help="cap training rows loaded (quick local de-risk runs); smoke forces 32")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=None, dest="max_seq",
                    help="train seq ceiling; default derives from --data-stem profile (v5=1920, v1_2=1664)")
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
    ap.add_argument("--save-every", type=int, default=0, dest="save_every",
                    help="checkpoint adapter+optimizer every N updates for resume (0 = use --eval-every)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from output_dir/checkpoint (multi-session / after a 12h timeout)")
    ap.add_argument("--data-stem", default="v1_2", dest="data_stem",
                    help="dataset stem under data/sft: 'v5' (pure-chess), 'v1_2' (full), 'v1_2_lean'")
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
