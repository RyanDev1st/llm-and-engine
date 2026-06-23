"""Prefix KV-cache bookkeeping (the part testable without a GPU/model): prefix math, the
reuse-eligibility rules, the env flag, and disable. The actual cache mechanics in model_hf
are GPU-validated by the self-guarding A/B check at serve time."""
import importlib

from backend import kv_cache as K


def _fresh(monkeypatch, val=None):
    if val is None:
        monkeypatch.delenv("CHESS_KV_REUSE", raising=False)
    else:
        monkeypatch.setenv("CHESS_KV_REUSE", val)
    importlib.reload(K)
    return K


def test_common_prefix_len():
    assert K.common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert K.common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert K.common_prefix_len([1, 2], [9, 2]) == 0


def test_env_flag_toggles_default_off(monkeypatch):
    # Default OFF: reuse diverges from a full prefill on transformers 5.x (A/B mismatch) and only
    # ever saved prefill, so it's opt-in. CHESS_KV_REUSE=1 re-enables it where proven to match.
    assert _fresh(monkeypatch, None).enabled() is False      # default OFF
    assert _fresh(monkeypatch, "1").enabled() is True
    assert _fresh(monkeypatch, "0").enabled() is False
    assert _fresh(monkeypatch, "false").enabled() is False


def test_reusable_requires_a_stored_prefix():
    c = K.PrefixCache()
    c.active = True
    assert c.reusable(list(range(200))) == 0                 # nothing stored yet


def test_reusable_only_on_pure_prefix_extension():
    # Reuse ONLY when the cached sequence is EXACTLY the start of the new one (extend, no crop —
    # Gemma's sliding-window cache can't be cropped). The cached LENGTH is the reuse point.
    c = K.PrefixCache()
    c.active = True
    base = list(range(100))                                  # > MIN_REUSE (64)
    c.store(base, kv=object())                              # prior full sequence
    assert c.reusable(base + [777, 888, 999]) == 100        # pure extension -> reuse all 100
    # divergence (cached has tokens beyond the shared prefix) must NOT reuse -> a clean full prefill
    c.store(base + [500, 501], kv=object())
    assert c.reusable(base + [777, 888, 999]) == 0          # shares 100 but cached is 102 -> 0


def test_reusable_zero_when_too_short_or_full_match():
    c = K.PrefixCache()
    c.active = True
    c.store(list(range(100)), kv=object())
    assert c.reusable([1, 2, 3] + list(range(50))) == 0      # not an extension -> skip
    assert c.reusable(list(range(100))) == 0                 # full match -> nothing new -> skip
    short = K.PrefixCache(); short.active = True
    short.store(list(range(20)), kv=object())               # cached < MIN_REUSE
    assert short.reusable(list(range(40))) == 0


def test_disable_is_sticky_and_clears():
    c = K.PrefixCache()
    c.active = True
    c.store(list(range(100)), kv=object())
    c.disable("self-check mismatch")
    assert c.active is False and c.kv is None
    assert c.reusable(list(range(100)) + [1]) == 0           # stays off


def test_gen_equal():
    import torch
    from backend.model_hf import _gen_equal
    a = torch.tensor([1, 2, 3, 10, 11, 12])
    b = torch.tensor([9, 9, 9, 10, 11, 12])                  # same generated tail past prompt_len=3
    assert _gen_equal(a, b, 3) is True
    c = torch.tensor([9, 9, 9, 10, 11, 99])
    assert _gen_equal(a, c, 3) is False
