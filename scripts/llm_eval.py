from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_training.evaluate_phases import evaluate_stub_turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.parse_args()
    print(json.dumps(evaluate_stub_turns(), indent=2))


if __name__ == "__main__":
    main()
