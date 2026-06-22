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
        """The number of cached positions to reuse, or 0. We reuse ONLY a PURE PREFIX EXTENSION
        — the cached sequence is exactly the start of `new_ids`, so we extend (never crop) the
        cache and prefill new_ids[L:]. No-crop is REQUIRED: Gemma's sliding-window cache cannot
        be cropped past its window (states are already evicted) — cropping was the live failure.
        Divergence (a different system/board) returns 0 -> a clean full prefill, reuse stays on."""
        if not self.active or self.ids is None or self.kv is None:
            return 0
        n_new, n_cached = len(new_ids), len(self.ids)
        if n_cached < MIN_REUSE or n_cached >= n_new:        # too short, or not an extension
            return 0
        return n_cached if common_prefix_len(self.ids, new_ids) == n_cached else 0

    def store(self, ids, kv) -> None:
        self.ids, self.kv = ids, kv

    def disable(self, reason: str = "") -> None:
        self.active = False
        self.ids = self.kv = None
        if reason:
            print(f"[kv] prefix reuse disabled: {reason}", flush=True)
