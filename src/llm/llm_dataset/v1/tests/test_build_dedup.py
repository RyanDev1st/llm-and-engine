"""Train/val split invariants. De-leak keeps a per-slice VAL FLOOR for eval coverage,
so a few val rows may share final-answer TEXT with train (low-diversity slices have few
distinct finals). The hard guard is no EXACT-ROW dup between val and train; final-text
overlap is allowed only as the bounded floor top-up."""
from llm_dataset.v1.build import VAL_FLOOR, _row_hash, split_train_val


def _row(i, slice_name, final, q=None):
    return {
        "slice": slice_name,
        "messages": [
            {"role": "user", "content": q if q is not None else f"q{i}"},
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": final},
        ],
    }


def test_no_exact_row_leak_between_val_and_train():
    # An EXACT full-row dup (same q AND same final) must never sit in both splits.
    rows = [_row(i, "A", f"answer {i}") for i in range(22)]
    rows.append(_row(0, "A", "answer 0", q="q0"))   # exact dup of row 0
    train, val = split_train_val(rows)
    train_hashes = {_row_hash(r) for r in train}
    assert not any(_row_hash(r) in train_hashes for r in val), "exact row leaked across splits"


def test_per_slice_val_floor_kept_for_low_diversity_slice():
    # All rows share the SAME final (one distinct answer) but DIFFERENT prompts -> distinct
    # full rows. Blanket de-leak would empty val; the floor must keep coverage.
    rows = [_row(i, "A", "same answer", q=f"distinct prompt number {i}") for i in range(40)]
    train, val = split_train_val(rows)
    val_a = [r for r in val if r["slice"] == "A"]
    assert len(val_a) >= min(VAL_FLOOR, 4), f"floor not kept: {len(val_a)} val rows"
    # but still no exact-row leak
    th = {_row_hash(r) for r in train}
    assert not any(_row_hash(r) in th for r in val_a)


def test_degenerate_slice_yields_no_leak_even_if_no_val():
    # Every row identical (one unique full row, like V1_Q) -> cannot hold out a non-dup
    # val row -> that slice gets 0 val, and crucially NO exact dup leaks into val.
    rows = [_row(0, "Q", "fixed", q="same") for _ in range(30)]
    train, val = split_train_val(rows)
    th = {_row_hash(r) for r in train}
    assert not any(_row_hash(r) in th for r in val)


def test_split_keeps_all_unique_rows():
    rows = [_row(i, "A", f"answer {i}") for i in range(22)]
    train, val = split_train_val(rows)
    assert len(train) + len(val) == 22
