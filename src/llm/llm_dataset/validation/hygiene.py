from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DuplicatePair:
    left_id: str
    right_id: str
    similarity: float


def normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def similarity_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    aset = set(normalize_text(a).split())
    bset = set(normalize_text(b).split())
    if not aset and not bset:
        return 1.0
    return len(aset & bset) / len(aset | bset)


def find_near_duplicates(records: list[dict], threshold: float = 0.85) -> list[DuplicatePair]:
    pairs: list[DuplicatePair] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            left = _first_user_text(records[i])
            right = _first_user_text(records[j])
            score = similarity_ratio(left, right)
            if score >= threshold:
                pairs.append(
                    DuplicatePair(
                        left_id=str(records[i].get("id", f"row_{i}")),
                        right_id=str(records[j].get("id", f"row_{j}")),
                        similarity=score,
                    )
                )
    return pairs


def _first_user_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""
