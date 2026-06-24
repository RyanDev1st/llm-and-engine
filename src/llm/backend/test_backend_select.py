"""The serve backend selector (CHESS_BACKEND) — pure logic, no weights. 'auto' keeps the original
behavior (HF if adapter else GGUF); an explicit value forces that backend for a deliberate A/B."""
from backend.model_gguf import pick_backend


def test_auto_picks_hf_with_adapter_gguf_without():
    assert pick_backend("auto", "/path/adapter") == "hf"
    assert pick_backend("auto", "") == "gguf"
    assert pick_backend(None, "/path/adapter") == "hf"     # unset == auto
    assert pick_backend("", None) == "gguf"


def test_explicit_backend_overrides_adapter_presence():
    assert pick_backend("gguf", "/path/adapter") == "gguf"  # GGUF even though an adapter exists
    assert pick_backend("hf", "") == "hf"                   # HF (base only) even with no adapter
    assert pick_backend("GGUF", "/a") == "gguf"             # case-insensitive
    assert pick_backend("  hf ", "/a") == "hf"              # trimmed


def test_unknown_backend_falls_back_to_auto():
    assert pick_backend("weird", "/a") == "hf"
    assert pick_backend("weird", "") == "gguf"
