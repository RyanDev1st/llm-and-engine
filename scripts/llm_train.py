from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_training.train_gemma4_lora import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()
    result = run_training(TrainConfig(args.phase, args.device, args.allow_cuda, args.dry_run, args.smoke, args.max_steps, args.max_examples))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
