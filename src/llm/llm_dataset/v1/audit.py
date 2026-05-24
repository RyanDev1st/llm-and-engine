from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import RULES, SLICES
from .paths import OUT
from .validate import validate_row


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(gold_dir: Path = OUT) -> int:
    accepted = load_rows(gold_dir / "accepted.jsonl")
    rejected = load_rows(gold_dir / "rejected.jsonl")
    failures = []
    by_slice = Counter(row["slice"] for row in accepted)
    reject_by_slice = Counter(row["slice"] for row in rejected)
    rule_counts = Counter(rule for row in accepted for rule in row["acceptance_rules"])
    for row in accepted:
        failures.extend((row["id"], v.rule, v.reason) for v in validate_row(row))
    reject_passed = [row["id"] for row in rejected if not validate_row(row)]
    print(f"accepted={len(accepted)} rejected={len(rejected)}")
    print("accepted_by_slice=" + _fmt_counts(by_slice))
    print("rejected_by_slice=" + _fmt_counts(reject_by_slice))
    print("rule_counts=" + _fmt_counts(rule_counts))
    _print_samples(accepted)
    _print_rejects(rejected)
    missing = _missing(by_slice, reject_by_slice, rule_counts)
    for item in missing:
        print(f"MISSING: {item}")
    for row_id, rule, reason in failures:
        print(f"FAIL: {row_id} {rule} {reason}")
    for row_id in reject_passed:
        print(f"FAIL: rejected row passed validation {row_id}")
    return 1 if failures or reject_passed or missing else 0


def _fmt_counts(counts: Counter) -> str:
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _print_samples(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if row["slice"] in seen:
            continue
        seen.add(row["slice"])
        final = row["messages"][-1]["content"]
        print(f"SAMPLE {row['slice']} {row['id']}: {final}")


def _print_rejects(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["slice"]].append(row["reject_reason"])
    for slice_name in sorted(grouped):
        print(f"REJECTS {slice_name}: " + "; ".join(grouped[slice_name]))


def _missing(by_slice: Counter, reject_by_slice: Counter, rule_counts: Counter) -> list[str]:
    out = []
    for slice_name in SLICES:
        if by_slice[slice_name] < 20:
            out.append(f"{slice_name} accepted < 20")
        if reject_by_slice[slice_name] < 5:
            out.append(f"{slice_name} rejected < 5")
    for rule in RULES:
        if rule_counts[rule] == 0:
            out.append(f"rule has no accepted coverage: {rule}")
    return out


if __name__ == "__main__":
    raise SystemExit(audit())
