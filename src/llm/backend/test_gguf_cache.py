"""The GGUF prefix/KV cache wiring — verified with a fake llama_cpp (no model load).
The agentic loop re-calls generate() with a growing prompt each turn; the RAM cache
lets llama.cpp skip re-prefilling the shared prefix. On by default, off via env."""
import sys
import types


def _fake_llama_cpp(calls):
    m = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kw):
            pass

        def set_cache(self, cache):
            calls.append(cache)

        def n_ctx(self):
            return 4096

    class FakeRAMCache:
        def __init__(self, capacity_bytes=2 ** 31):
            self.capacity_bytes = capacity_bytes

    m.Llama = FakeLlama
    m.LlamaRAMCache = FakeRAMCache
    return m


def _make(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    from backend.model_gguf import GGUFModel
    return GGUFModel(gguf=gguf)


def test_cache_enabled_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp(calls))
    monkeypatch.delenv("CHESS_GGUF_CACHE", raising=False)
    _make(monkeypatch, tmp_path)
    assert len(calls) == 1 and calls[0].capacity_bytes == (1 << 30)   # RAM cache set, 1 GiB


def test_cache_disabled_by_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp(calls))
    monkeypatch.setenv("CHESS_GGUF_CACHE", "0")
    _make(monkeypatch, tmp_path)
    assert calls == []                                                # no cache set
