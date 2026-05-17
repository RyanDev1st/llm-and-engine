from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BatchMeta:
    batch_id: str
    slice_name: str
    generator: str
    created_at: str
    target_count: int = 25


@dataclass(frozen=True)
class BatchRecord:
    meta: BatchMeta
    conversations: list[dict[str, Any]]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_batch_id(slice_name: str, serial: int) -> str:
    return f"{slice_name}_{serial:04d}"


def new_batch(slice_name: str, serial: int, generator: str) -> BatchRecord:
    meta = BatchMeta(
        batch_id=make_batch_id(slice_name, serial),
        slice_name=slice_name,
        generator=generator,
        created_at=utc_now_iso(),
    )
    return BatchRecord(meta=meta, conversations=[])


def add_conversation(batch: BatchRecord, conversation: dict[str, Any]) -> BatchRecord:
    updated = [*batch.conversations, conversation]
    return BatchRecord(meta=batch.meta, conversations=updated)


def is_batch_complete(batch: BatchRecord) -> bool:
    return len(batch.conversations) >= batch.meta.target_count


def write_batch_json(batch: BatchRecord, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{batch.meta.batch_id}.json"
    payload = {
        "meta": asdict(batch.meta),
        "count": len(batch.conversations),
        "conversations": batch.conversations,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
