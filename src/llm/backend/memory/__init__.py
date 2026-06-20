"""Harness memory system. Three tiers (Claude Code analog):
- ephemeral: the in-turn tool scratchpad (lives in CoachLoop.respond, discarded after the turn);
- session: the running session note (context_window compaction);
- persistent: a per-user profile auto-captured each turn behind a write-discipline gate
  (store.py / extract.py) and re-injected into the system prompt every turn.

Public surface used by the serve loop / web app:
  memory.capture(user_message, user_id)  -> persist durable typed facts from a turn
  memory.memory_block(user_id)           -> the USER PROFILE block to inject into the prompt
"""
from __future__ import annotations

from .store import (add_fact, capture, load_profile, memory_block, render_profile,
                    save_profile)

__all__ = ["capture", "memory_block", "render_profile", "load_profile", "save_profile",
           "add_fact"]
