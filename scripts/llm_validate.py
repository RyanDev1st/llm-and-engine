from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_training.smoke_data import smoke_records
from llm_training.validate_jsonl import load_jsonl, validate_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = smoke_records() if args.smoke else load_jsonl(Path(args.data))
    failures = validate_records(records)
    print(json.dumps({"ok": not failures, "failures": failures, "count": len(records)}, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
