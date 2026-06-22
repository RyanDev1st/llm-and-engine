"""Harness memory system. Three tiers (Claude Code analog):
- ephemeral: the in-turn tool scratchpad (lives in CoachLoop.respond, discarded after the turn);
- session: a FEN-keyed cache of this-session analysis facts (session.py) — reused across turns
  with a strict freshness guard, plus context_window compaction on eviction;
- persistent: a per-user profile auto-captured each turn behind a write-discipline gate
  (store.py / extract.py) and re-injected into the system prompt every turn.
- episodic: a GLOBAL 'how-to-operate' store (episodic.py) — learns a tool's correct usage from a
  turn's error->fix recovery and recalls it for a similar later request (flag-gated CHESS_EPISODIC,
  default off). About TOOLS not the user, so it helps every user/machine when the dir is synced.

Public surface used by the serve loop / web app:
  memory.capture(user_message, user_id)  -> persist durable typed facts from a turn
  memory.memory_block(user_id)           -> the USER PROFILE block to inject into the prompt
  memory.observe(user_message, result, plugin_context)   -> harvest a how-to-operate episode
  memory.episodic_block(user_message, plugin_context)    -> the RECALLED hint to inject
  memory.session.update / render / clear -> the per-session FEN-keyed fact cache
"""
from __future__ import annotations

from .episodic import episodic_block, observe
from .store import (add_fact, capture, load_profile, memory_block, render_profile,
                    save_profile)

__all__ = ["capture", "memory_block", "render_profile", "load_profile", "save_profile",
           "add_fact", "observe", "episodic_block"]
