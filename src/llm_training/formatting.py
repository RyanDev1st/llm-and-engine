from __future__ import annotations

import json


def format_record(record: dict) -> str:
    history = json.dumps(record["history"], ensure_ascii=False)
    user_input = json.dumps(record["input"], ensure_ascii=False)
    target = json.dumps(record["target"], ensure_ascii=False)
    return "\n".join([
        f"phase={record['phase']}",
        f"summary={record['summary']}",
        f"history={history}",
        f"input={user_input}",
        f"target={target}",
    ])


def format_target_json(record: dict) -> str:
    return json.dumps(record["target"], ensure_ascii=False, sort_keys=True)
