"""Prefix KV-cache reuse across CoachLoop steps — a SAFE-BY-CONSTRUCTION speed optimization.

Why this and NOT "shrink the contract": the ~1062-token harness contract
(`system_prompt.build_system`) is the EXACT text the model trained on — frozen by training,
like the weights. Trimming it at serve would desync serve from train and hurt routing. KV
reuse instead avoids RE-ENCODING that identical prefix on every loop step WITHOUT changing a
single input token: same tokens -> same keys/values -> identical output under greedy decode.

The only risk is the cache mechanics (HF version drift), so correctness is guarded three ways:
  1. reuse ONLY an EXACT token-prefix match (identical ids => identical KV, by construction);
  2. any exception in the reuse path falls back to a normal full prefill;
  3. a one-time A/B self-check (run by the backend) DISABLES reuse for the session if the
     reuse output ever diverges from the full-prefill output.
So the worst case is "no speedup", never wrong output. This module holds the pure bookkeeping
(env flag, prefix math, the one-slot cache); the model forward/crop lives in model_hf.
"""
from __future__ import annotations

import os

MIN_REUSE = 64    # don't bother reusing a prefix shorter than this (overhead > saving)


def enabled() -> bool:
    """Reuse on by default; CHESS_KV_REUSE=0 forces the plain full-prefill path."""
    return os.environ.get("CHESS_KV_REUSE", "1") not in ("0", "false", "False", "")


def common_prefix_len(a, b) -> int:
    """Length of the shared leading run of two 1-D int token sequences (lists or 1-D tensors)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return i


class PrefixCache:
    """One-slot cache of the last (token ids, KV cache). The frozen contract + kept history +
    user turn are IDENTICAL across the loop steps of a turn, so step N+1 shares a long exact
    prefix with step N — we crop the prior KV to that shared length and prefill only the new
    tail. `verified` gates trusting reuse after the A/B self-check; `active` is the env flag."""

    def __init__(self) -> None:
        self.ids = None            # 1-D token ids of the last full sequence (prompt + generated)
        self.kv = None             # the backend KV object for self.ids
        self.active = enabled()
        self.verified = False      # set True once the A/B self-check confirms reuse matches

    def reusable(self, new_ids) -> int:
        """Shared exact-prefix length with the cached ids, or 0 when there's nothing worth
        reusing. The caller crops the cached KV to this length and prefills new_ids[L:]."""
        if not self.active or self.ids is None or self.kv is None:
            return 0
        n_new = len(new_ids)
        cap = min(len(self.ids), n_new)
        if cap < MIN_REUSE:
            return 0
        ln = common_prefix_len(self.ids, new_ids)
        if ln >= n_new:                 # full match -> nothing new to prefill; don't reuse
            return 0
        return ln if ln >= MIN_REUSE else 0

    def store(self, ids, kv) -> None:
        self.ids, self.kv = ids, kv

    def disable(self, reason: str = "") -> None:
        self.active = False
        self.ids = self.kv = None
        if reason:
            print(f"[kv] prefix reuse disabled: {reason}", flush=True)
