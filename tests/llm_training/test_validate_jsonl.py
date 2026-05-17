from llm_training.smoke_data import smoke_records
from llm_training.validate_jsonl import validate_records


def test_validate_smoke_records() -> None:
    assert validate_records(smoke_records()) == []


def test_validate_rejects_missing_summary() -> None:
    record = smoke_records()[0] | {"summary": ""}
    assert "DATASET_SUMMARY_REQUIRED" in validate_records([record])
