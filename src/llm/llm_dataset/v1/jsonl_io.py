"""Transparent gzip I/O for the SFT corpus. The corpus is stored gzipped
(.jsonl.gz, ~8x smaller — repeated per-row manifest) so the files clear GitHub's
100MB push limit, but the rest of the code speaks plain `.jsonl`: writers emit
`.gz`, readers prefer a `.gz` sibling and still accept a plain `.jsonl`."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator


def _is_gz(path: Path) -> bool:
    return str(path).endswith(".gz")


def resolve_read(path: str | Path) -> Path:
    """Given a logical path, return the file to read: the path itself if present,
    else its `.gz` sibling (so callers can keep passing `*.jsonl`)."""
    p = Path(path)
    if p.exists():
        return p
    gz = Path(str(p) + ".gz")
    return gz if gz.exists() else p


def gz_target(path: str | Path) -> Path:
    """The gzipped on-disk name for a logical path."""
    p = Path(path)
    return p if _is_gz(p) else Path(str(p) + ".gz")


def read_rows(path: str | Path) -> Iterator[dict]:
    p = resolve_read(path)
    opener = gzip.open if _is_gz(p) else open
    with opener(p, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_rows(path: str | Path, rows: Iterable[dict]) -> Path:
    """Write rows as gzipped json-lines; returns the actual `.gz` path written."""
    target = gz_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target
