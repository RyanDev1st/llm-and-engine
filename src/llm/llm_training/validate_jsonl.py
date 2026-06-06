from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_runtime.contracts import validate_mode_invariants
from llm_runtime.grounding import validate_narration
from llm_runtime.json_outputs import parse_narrator_output, parse_router_output
from llm_training.dataset import validate_record_shape


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    from llm_dataset.v1.jsonl_io import read_rows  # transparent .jsonl/.jsonl.gz
    return list(read_rows(path))


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for record in records:
        failures.extend(_validate_record(record))
    return failures


def _validate_record(record: dict[str, Any]) -> list[str]:
    failures = [item.error_id for item in validate_record_shape(record)]
    if failures:
        return failures
    target = record["target"]
    if record["phase"] == "router":
        failures.extend(item.error_id for item in parse_router_output(json.dumps(target)).violations)
    else:
        failures.extend(item.error_id for item in parse_narrator_output(json.dumps(target)).violations)
        latest = record["input"].get("latest_tool_result", {})
        failures.extend(item.error_id for item in validate_narration(target.get("text", ""), latest))
    failures.extend(item.invariant_id for item in validate_mode_invariants(record["history"]))
    return failures
