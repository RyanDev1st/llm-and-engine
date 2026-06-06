"""Corpus is stored gzipped to clear GitHub's 100MB limit; readers must be
transparent (prefer .gz, still accept plain .jsonl)."""
import gzip
import json

from llm_dataset.v1.jsonl_io import gz_target, read_rows, resolve_read, write_rows

ROWS = [{"id": "a", "n": 1}, {"id": "b", "n": 2}]


def test_write_rows_emits_gz_and_roundtrips(tmp_path):
    out = write_rows(tmp_path / "corpus.jsonl", ROWS)
    assert out.suffix == ".gz" and out.exists()
    assert not (tmp_path / "corpus.jsonl").exists()  # only the .gz on disk
    assert list(read_rows(tmp_path / "corpus.jsonl")) == ROWS  # logical name resolves to .gz


def test_gz_is_smaller_than_raw(tmp_path):
    big = [{"id": str(i), "manifest": ["t"] * 50} for i in range(2000)]
    out = write_rows(tmp_path / "big.jsonl", big)
    raw = sum(len(json.dumps(r)) + 1 for r in big)
    assert out.stat().st_size < raw  # compression actually helps


def test_read_plain_jsonl_still_works(tmp_path):
    p = tmp_path / "plain.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    assert list(read_rows(p)) == ROWS


def test_resolve_prefers_existing_then_gz(tmp_path):
    logical = tmp_path / "x.jsonl"
    assert resolve_read(logical) == logical  # neither exists -> logical
    gz = gz_target(logical)
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(ROWS[0]) + "\n")
    assert resolve_read(logical) == gz  # .gz exists -> use it
    logical.write_text("", encoding="utf-8")
    assert resolve_read(logical) == logical  # plain present -> prefer it
