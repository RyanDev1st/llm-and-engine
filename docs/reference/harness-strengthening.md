# Harness strengthening — memory, routing, thinking

Living record of the three-pillar effort to make the FROZEN E4B harness smarter without
retraining: better **memory** (what the model sees), **routing** (how the loop feeds it),
and **thinking** (how it reasons). Grounded in real harnesses — Claude Code (3-tier memory,
compaction), OpenCode (Compaction/Summary agents), Cursor (the auto-memory rollback lesson),
Hermes (`<tool_call>` analog of our verbs), Anthropic (interleaved thinking). Plan of record:
`C:\Users\admin\.claude\plans\quiet-singing-storm.md`.

Each section = what changed, the files, why, and how to test it.

---

## Pillar 1 — Memory (DONE)

Adds the memory tiers the served harness lacked. New feature folder `src/llm/backend/memory/`.

### 1b — Persistent user memory + write gate
**What:** a per-user profile, auto-captured each turn, re-injected into the system prompt every
turn (the CLAUDE.md pattern) so the frozen model tailors to a user it never trained on.
**Files:** `memory/extract.py` (typed fact miner), `memory/store.py` (disk profile + the write
gate), `memory/__init__.py`; wired in `inference.CoachLoop.respond` (new `memory_block` param,
folded into the system prompt before the window budget) and `web_app.App._run` (inject + capture).
**The write gate (central — un-gated auto-memory rots; Cursor shipped then removed it):** only
TYPED durable categories are captured — `rating`, `style`, `weakness`, `pref`, plus an opt-in
"remember that I…" `note`. `store.add_fact` validates category against a whitelist, dedupes,
supersedes (rating) or caps lists (oldest drops), bounds length, and rejects FEN/email/PII even
on the note path. Runtime profiles live in `data/memory/<user_id>/profile.json` (gitignored);
keyed by `user_id` (`CHESS_USER_ID`, default `default`) so multi-user is free.
**Why:** the served coach forgot the user between sessions — the single biggest "smarter" gap.

### 1a — Session fact cache
**What:** a FEN-keyed cache of this-session analysis facts (eval/best/threats/review) so a
follow-up reuses them instead of re-calling Stockfish.
**Files:** `memory/session.py`; wired in `web_app.App` (per-session cache, cleared on reset) +
`_run` (`_context_block` injects profile + fresh facts; caches facts only from the primary
trained run AND only when the board was stable that turn).
**Freshness guard:** facts render next turn ONLY if the live FEN still matches the FEN they were
computed at — a move/undo silently invalidates the cache, so a stale fact never shows.
**Why:** the in-turn scratchpad is discarded each turn, so follow-ups re-computed known facts.

### How to test Pillar 1
- `python -m pytest src/llm/backend/memory -q` (13 tests: extractor, gate, session freshness).
- Manual: `cd src/llm && python -m backend.dev_serve`; tell the coach
  "I'm ~1200 and I always hang my queen, keep it short" → `data/memory/default/profile.json`
  gains exactly those typed facts (nothing transient); restart → the next session's first reply
  reflects the profile. Ask for an eval, then "why?" → the second turn reuses the cached eval
  (no re-call) while the board is unchanged.

---

## Pillar 2 — Routing loop (DONE)

### Finding that reshaped the pillar: the contract is FROZEN BY TRAINING
The planned "shrink the ~1062-token contract" (2b) was **dropped as unsafe**. `system_prompt.
build_system` renders the EXACT text the model trained on (~85% of every training row), so
trimming it at serve desyncs serve from train and would hurt routing on the frozen model — the
same train≠serve class of bug that broke earlier adapters. The contract is frozen like the
weights. The only safe way to stop re-paying its prefill cost is to not RE-ENCODE it — i.e. KV
reuse, which changes zero input tokens.

### 2a — Prefix KV-cache reuse (safe-by-construction)
**What:** across a turn's loop steps the system contract + history + user turn are an IDENTICAL
token prefix, re-prefilled from scratch each `generate()` (the cost behind the ~14s lag). Reuse
the prior step's KV for that exact prefix and prefill only the new tail.
**Files:** `backend/kv_cache.py` (env flag + prefix math + one-slot `PrefixCache`); `backend/
model_hf.py` (`_run_generate` does the cropped-cache continuation; `_gen_cached` orchestrates
reuse + the self-check).
**Three correctness guards (worst case = no speedup, NEVER wrong output):** (1) reuse only an
EXACT token-prefix match — identical ids ⇒ identical KV by construction; (2) any cache-mechanic
exception falls back to a full prefill; (3) a one-time **A/B self-check** runs both paths on the
first reuse opportunity, returns the trusted full-prefill output, and only enables reuse if the
two outputs match (else disables reuse for the session). Reuse is restricted to the greedy
(temperature 0) adapter-on path; the base/sampling paths bypass it.
**Flag:** `CHESS_KV_REUSE` (default on; `=0` forces the plain path).
**Status note:** the cache *bookkeeping* is unit-tested here (`test_kv_cache.py`); the GPU cache
*mechanics* are validated at serve time by the self-guarding A/B check — so enabling it on the T4
is safe (it self-disables if the installed transformers' cache API doesn't line up).

**Sliding-window refinement (live finding):** Gemma uses sliding-window attention, and its cache
**cannot be `crop`ped** past the window (early states are evicted — the live error
`Cannot crop a DynamicSlidingWindowLayer …`). So reuse is **pure-extension only**: the cache is
reused only when the cached sequence is exactly the START of the new one (each loop step appends,
so this is the common in-turn case), and the cache is EXTENDED, never cropped. A divergent prefix
(a different system/board, or a tokenizer boundary mismatch) returns 0 → a clean full prefill,
and reuse stays on for the next step. The self-guard had correctly disabled reuse on the crop
error before this fix — proof the safety net holds (correct output, just no speedup).

### 2c — Cross-turn tool-result reuse
Delivered in Pillar 1a (the session fact cache) — a follow-up reuses the prior eval/best instead
of re-calling the engine, with the FEN freshness guard.

### How to test Pillar 2
- `python -m pytest src/llm/backend/test_kv_cache.py -q` (prefix math, reuse rules, flag, disable).
- On the T4 (serving): a multi-step turn should drop in wall-clock vs `CHESS_KV_REUSE=0`; watch
  the log for a `[kv] prefix reuse disabled: …` line — if it appears, reuse self-disabled (correct
  output, no speedup) and we look at the transformers cache API. Output must be identical to the
  `CHESS_KV_REUSE=0` run either way.

---

## Pillar 3 — Thinking (DONE)

### 3a — Live-stream `<think>` to the panel
**What:** the model's reasoning now streams to a LIVE "🧠 thinking…" preview during generation
instead of being hidden by `clean()` and only shown post-hoc — finishing the open G1 follow-up
and the "redesign the think panel" ask. Reasoning still never touches the chat bubble.
**Files:** `gemma_chat_site/static/index.html` — `extractThink` (pulls closed + an unclosed
trailing `<think>` mid-stream), `showLiveThink`/`dropLiveThink` (the transient preview), wired
into the `streamChat` token/reply_chunk handlers; dropped on a `tool`/`think`/`done` event (the
reasoning is then recorded in the panel via the existing per-step / `{type:think}` records).

### 3c — Interleaved thinking is preserved (verified)
A `<think>` before a tool step survives in the turn's history (it rides `extract_call` →
`_to_skill_verb` into `new_turns`), so the next step can reason about the result — only the
FINAL reply has its `<think>` stripped (G1). This already held; locked with a regression test
(`test_intermediate_think_is_kept_in_history_but_stripped_from_final`).

### 3b — grounded reflection: deterministic guards, NOT a model round-trip
A model self-reflection pass was evaluated and **rejected**: it adds a full generate round-trip
(against the Pillar-2 goal) and a small model self-judging is weak. The existing DETERMINISTIC
guards in `_finalize` are stronger and free — `_correct_eval_number` (no fabricated eval),
`_correct_move_names` (no fabricated move list), `_ensure_required_narrated` (required facts
present). Adding a fuzzy eval-direction guard was also rejected (over-reach risk vs trained
phrasing — consistent with the deterministic-routing-restraint principle). So grounding stays
deterministic + zero-latency; no new code.

### How to test Pillar 3
- `python -m pytest src/llm/backend/test_serve_smoke.py -q` (incl. the interleaved-think test).
- On the T4 (serving): ask in `think`/`auto` mode → a live "🧠 thinking…" preview fills with the
  reasoning during generation, then the answer types into the chat bubble (no `<think>` leak).
