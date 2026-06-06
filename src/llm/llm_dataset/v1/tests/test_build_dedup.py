"""The train/val split must not leak: no val final-answer text may appear in train."""
from llm_dataset.v1.build import split_train_val


def _row(i, slice_name, final):
    return {
        "slice": slice_name,
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": final},
        ],
    }


def _final(row):
    return next(m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant")


def test_no_val_final_leaks_into_train():
    # 22 rows in one slice; idx 0,10,20 -> val. Make row 0's final duplicate row 1's
    # (a train row), so without dedup it would leak.
    rows = []
    for i in range(22):
        final = "answer 1" if i == 0 else f"answer {i}"
        rows.append(_row(i, "A", final))
    train, val = split_train_val(rows)
    train_finals = {_final(r) for r in train}
    assert val, "val split should be non-empty"
    assert not any(_final(r) in train_finals for r in val), "val final leaked into train"


def test_split_keeps_all_rows():
    rows = [_row(i, "A", f"answer {i}") for i in range(22)]
    train, val = split_train_val(rows)
    assert len(train) + len(val) == 22
