from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_training.smoke_data import smoke_records


if __name__ == "__main__":
    print(json.dumps(smoke_records(), indent=2))
