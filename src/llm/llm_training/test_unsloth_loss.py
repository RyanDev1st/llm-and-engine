"""Engine-agnostic checks for the Unsloth masked weighted loss (no GPU/unsloth).

A fake model mimics a CausalLM: forward returns logits = lm_head(hidden). When
_masked_loss swaps lm_head -> Identity, forward returns hidden — exactly the
real path. This validates the weighted-CE math + label masking + lm_head locate
without importing unsloth (which needs CUDA)."""
import types

import torch

from llm_training.train_unsloth import _lm_head_holder, _masked_loss, _weighted_ce

VOCAB, HID, SEQ = 11, 8, 6


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HID)
        self.lm_head = torch.nn.Linear(HID, VOCAB, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **_):
        return types.SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


def _batch():
    ids = torch.randint(0, VOCAB, (1, SEQ))
    labels = ids.clone()
    labels[0, :2] = -100  # mask the first two (prompt) tokens
    weights = torch.ones(1, SEQ)
    return {"input_ids": ids, "attention_mask": torch.ones(1, SEQ)}, labels, weights


def test_lm_head_locator_finds_head():
    assert _lm_head_holder(FakeCausalLM()).lm_head is not None


def test_all_ones_weights_equal_unweighted():
    torch.manual_seed(0)
    m = FakeCausalLM()
    b, labels, weights = _batch()
    weighted = _masked_loss(m, dict(b), labels, weights)
    unweighted = _masked_loss(m, dict(b), labels, None)
    assert torch.allclose(weighted, unweighted, atol=1e-5)


def test_lm_head_restored_after_loss():
    m = FakeCausalLM()
    head_before = m.lm_head
    b, labels, weights = _batch()
    _masked_loss(m, dict(b), labels, weights)
    assert m.lm_head is head_before and isinstance(m.lm_head, torch.nn.Linear)


def test_upweighting_fact_tokens_changes_loss():
    torch.manual_seed(1)
    m = FakeCausalLM()
    b, labels, weights = _batch()
    base = _masked_loss(m, dict(b), labels, weights)
    weights[0, 3] = 5.0  # upweight one supervised token
    up = _masked_loss(m, dict(b), labels, weights)
    assert not torch.allclose(base, up)


def test_weighted_ce_matches_manual():
    logits = torch.randn(4, VOCAB)
    labels = torch.randint(0, VOCAB, (4,))
    w = torch.tensor([1.0, 5.0, 1.0, 5.0])
    per = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = (per * w).sum() / w.sum()
    assert torch.allclose(_weighted_ce(logits, labels, w), expected)
