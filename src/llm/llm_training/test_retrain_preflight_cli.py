from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_preflight():
    path = Path(__file__).resolve().parents[3] / "scripts" / "retrain_preflight.py"
    spec = importlib.util.spec_from_file_location("retrain_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_retrain_preflight_tokenizer_dir_can_be_overridden(monkeypatch):
    module = _load_preflight()
    explicit = Path("src/llm/models/gemma4_e4b")

    assert module._resolve_tok_dir(str(explicit)) == explicit

    monkeypatch.setenv("CHESS_TOK_DIR", str(explicit))
    assert module._resolve_tok_dir(None) == explicit
