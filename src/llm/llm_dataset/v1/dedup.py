from __future__ import annotations

from llm_dataset.validation.hygiene import find_near_duplicates


def drop_near_duplicates(rows: list[dict], threshold: float = 0.85) -> list[dict]:
    """Return rows with near-duplicates removed (keeps left/earlier copy).

    find_near_duplicates(records, threshold) accepts list[dict].  It compares
    the first user-turn text from each record's "messages" list and identifies
    the duplicated right-side record via its "id" field (falling back to
    "row_{i}" when absent).  We inject a stable positional "id" so the
    returned DuplicatePair.right_id maps back to an index we can filter on.
    """
    # Inject a positional "id" so pair IDs are predictable index strings.
    tagged: list[dict] = [{"id": str(idx), **row} for idx, row in enumerate(rows)]

    pairs = find_near_duplicates(tagged, threshold=threshold)

    drop_ids = {pair.right_id for pair in pairs}

    # Strip the injected "id" only when the original row had no "id" field.
    original_had_id = {str(idx) for idx, row in enumerate(rows) if "id" in row}
    return [
        row if str(idx) in original_had_id else {k: v for k, v in row.items() if k != "id"}
        for idx, row in enumerate(tagged)
        if str(idx) not in drop_ids
    ]
