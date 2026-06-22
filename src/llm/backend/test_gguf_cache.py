"""The GGUF prefix/KV cache wiring — verified with a fake llama_cpp (no model load).
The agentic loop re-calls generate() with a growing prompt each turn; the RAM cache
lets llama.cpp skip re-prefilling the shared prefix. On by default, off via env."""
import sys
import types


def _fake_llama_cpp(calls, seen_messages=None, seen_kw=None):
    m = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kw):
            pass

        def set_cache(self, cache):
            calls.append(cache)

        def n_ctx(self):
            return 4096

        def create_chat_completion(self, messages, **kw):
            if seen_messages is not None:
                seen_messages.append(messages)
            if seen_kw is not None:
                seen_kw.append(kw)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

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


def test_gguf_remaps_tool_messages_so_model_sees_results(monkeypatch, tmp_path):
    # ROOT FIX: Gemma's embedded GGUF chat template drops role="tool", so the GGUF path
    # must remap tool turns to <tool_result> user turns (same as train + HF) — else the
    # model never sees its tool results and fabricates. Assert no role="tool" reaches
    # create_chat_completion and the result text survives as a user turn.
    seen = []
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp([], seen))
    model = _make(monkeypatch, tmp_path)
    convo = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what's the eval?"},
        {"role": "assistant", "content": "<tool>eval depth=18</tool>"},
        {"role": "tool", "content": "score: +0.37 pawns from white POV, depth=18"},
    ]
    model.generate(convo, 64, ["</tool>"])
    sent = seen[-1]
    assert all(m["role"] != "tool" for m in sent)                     # no dropped tool turns
    assert any("score: +0.37" in m["content"] and m["role"] == "user" for m in sent)
    assert any("<tool_result>" in m["content"] for m in sent)         # rendered, not dropped


def test_gguf_repeat_penalty_matches_hf_parity(monkeypatch, tmp_path):
    # The HF serve path runs decode penalties OFF (repetition_penalty=1.0) because they corrupt
    # NAME COPYING; GGUF used to hardcode 1.2, drifting from HF/train. GGUF must read the SAME
    # CHESS_REP_PENALTY env with the SAME default (1.0) so the two backends can't diverge.
    seen_kw = []
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp([], seen_messages=None, seen_kw=seen_kw))
    monkeypatch.delenv("CHESS_REP_PENALTY", raising=False)
    model = _make(monkeypatch, tmp_path)
    model.generate([{"role": "user", "content": "hi"}], 16, ["</tool>"])
    assert seen_kw[-1]["repeat_penalty"] == 1.0                        # HF parity by default
    # rollback to the old GGUF behaviour is free via the env var
    monkeypatch.setenv("CHESS_REP_PENALTY", "1.2")
    model.generate([{"role": "user", "content": "hi"}], 16, ["</tool>"])
    assert seen_kw[-1]["repeat_penalty"] == 1.2
