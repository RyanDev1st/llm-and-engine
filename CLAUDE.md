# LLM tool-calling — product workspace

Project memory for Claude Code and other agents. Loaded every session. Keep **under 200 lines**, signal-dense, and verifiable. Update this file when layout, commands, or rules change. 

! Archive bin is `./legacy [ignore]/` — gitignored. Never edit, import, or reference it from live code; only move dead files INTO it.
<!-- Maintainer: add path-scoped rules under .claude/rules/ when this file grows. -->

## Mission

Ship **working software** for reliable **LLM tool use**: tool selection, JSON schemas, multi-turn loops, argument validation, and failure handling.

| In scope | Out of scope (unless user says otherwise) |
| --- | --- |
| Implementation, integration, release-quality behavior | Toy or simulated backends (label test fixtures explicitly) |
| Research notes for rationale and citations | Secrets in repo, commits, or chat |

## Repository map

### What we train: the agent (LLM harness)

The product is a **chess-coach agent** = an LLM that **routes user intent to tools and narrates tool results** (see `src/llm/llm_training/system_prompt.py`). It does **not** compute chess — the engine/backend does. Goal: train Gemma 4 **E4B** QLoRA on **Kaggle T4** → LoRA adapter → serve **q4_0 GGUF locally on the RTX 4060** (E2B fallback). Active plan files at root: `implementation.md` (the plan, 3 phases) and `handoff.md`.

| Path | Purpose |
| --- | --- |
| `CLAUDE.md` | Team agent instructions (this file) |
| `src/llm/llm_dataset/v1/` | **ACTIVE** SFT generator. Spec = `contracts.py`; `profiles.py` writes the v1_2 corpus. Source of truth for harness behavior |
| `data/sft/v1_2_train.jsonl`, `data/sft/v1_2_val.jsonl`, `data/sft/v1_2/` | **ACTIVE** SFT corpus (split + accepted/rejected). The ONLY corpus trainers read |
| `src/llm/llm_training/` | QLoRA trainer (`run_train.py`, `train_cuda.py`), loader, `eval_routing.py`, `system_prompt.py` |
| `src/llm/backend/` | Environment the agent calls: tool executor + Stockfish engine + HTTP server. Live skills catalog = `src/llm/skills/` (loaded by `skills.load_skills`) |
| `src/llm/skills_demo/` | 40 chess SKILL.md fixtures for routing tests + presentation demo. `_specs.py` (data) + `_generate.py` (renderer); NOT auto-loaded by the backend (pass as `load_skills` root) |
| `src/llm/gemma_chat_site/` | Web app (board + chat UI) |
| `src/llm/runtime/llamacpp/` | Bundled llama.cpp for GGUF serving |
| `src/engine/research/` | Standalone custom chess engine (alt backend) |
| `docs/` | Durable docs + dated reports |
| `legacy [ignore]/` | **Archive bin** for superseded plans/code/data — gitignored, never imported by live code |
| `.claude/settings.local.json` | Personal permissions — **gitignored**, never commit |
| `.claude/scheduled_tasks.lock` | Scheduler lock — **gitignored**, ephemeral |
| `.claude/worktrees/` | Agent worktrees — **gitignored**; do not edit unless task names a path |

**Root policy:** only `CLAUDE.md`, `.gitignore`, the three plan files above, and documented config at repo root. Everything else lives under `src/`, `data/`, `docs/`, or `legacy [ignore]/`.

## Product principles

### File size (hard cap)

- No code source file may exceed **200 lines** (imports and blank lines count). Meaning all other files can exceed this. 
- If a change would exceed 200 lines: split into additional files under the **same feature folder** (next section), create and put it in a logcal hierarchy. Never bypass the cap with comments or string concatenation.

### Feature folders (colocation)

- One capability → **one directory** under `src/llm/` (or `src/engine/`) with a short domain name (≤3 words, `snake_case`), e.g. `src/llm/llm_dataset/`, `src/llm/backend/`, `src/llm/llm_training/`.
- All code for that capability stays in that folder. New capability → new folder. Extending a capability → existing folder only.
- Do not scatter the same feature across repo root, `docs/`, and unrelated `src/` siblings.

### Legacy archival (keep the workspace legible)

The repo accumulates dead plans, datasets, and build scripts. Hard rules to stay legible:

1. **One active plan set:** `implementation.md` (the plan) + `handoff.md`. Any other root plan (e.g. `IMPLEMENTATION_PLAN.md`, `*_spec_v3.md`, `implementation_fpt.md`, dead-path runbooks) → move to `legacy [ignore]/`.
2. **One active corpus:** `v1_2`. Older corpora (`v1_train/val`, `v1_gold/`, `chess_assistant_v3_*`, `slice/`, `slices/`) and their build scripts (`src/llm/llm_dataset/build/`) → move to `legacy [ignore]/`.
3. **Archive = move, never delete.** Never leave a live import or test pointing into `legacy [ignore]/`. If moving breaks a reference, the referencing code is dead too — move it as well.
4. **Supersede in the same change:** before adding a new plan/dataset/spec version, move the prior one to the archive in the same commit.
5. **Coherence check (do this when asked to clean up):** `implementation.md` + `handoff.md` must agree on model (Gemma 4 E4B preferred, E2B fallback), infra (train Kaggle T4 → serve local GGUF on 4060), and active corpus (v1_2). Fix drift in place; do not spawn parallel plans.

### Workspace hygiene (required before “done”)

1. No new root-level files except those listed in **Repository map**.
2. No `_copy`, `_old`, `temp`, or duplicate scripts.
3. Every new path is referenced by code, tests, or docs in the same change set.
4. New top-level or feature folder → add one row to **Repository map** in this file in the same change set.

### Reports (required layout)

- Path: `<scope-dir>/YYYY-MM-DD-<topic>-<artifact>.md`
- Allowed `<scope-dir>`: `docs/` (default). Superseded reports → `legacy [ignore]/`.
- Line 1: `Parent: <relative-path>` or `Parent: none`
- Sections in order: **Status**, **Scope**, **Evidence** (commands + outcomes), **Next** (numbered list)
- Same topic + same calendar date → append to the existing file **or** supersede as `…-v2.md` with a link to the prior file. Do not create a parallel sibling for the same topic.

## Verification

Claude performs best with explicit success criteria. Before claiming completion:

| Check | Command / rule |
| --- | --- |
| Tests | `python -m pytest src/llm/llm_dataset/v1/tests -q` and `python -m pytest src/llm/llm_training -q`, or the path named in the task |
| Lint / typecheck | Use project-standard commands when present; do not invent new tooling |
| Behavior | State expected output; if tests do not exist, give a manual repro the user can run |
| UI (if applicable) | Screenshot or browser snapshot compare against stated expectation |

If verification fails, fix or report the failure with the failing command and error excerpt. Do not claim “done” on assumptions.

## Engineering

- **Secrets:** never paste keys, cookies, tokens, or private URLs. Before any commit, confirm `.gitignore` covers `.env`, `*.pem`, `.claude/settings.local.json`, `.claude/scheduled_tasks.lock`, and `.claude/worktrees/`.
- **Dependencies:** prefer existing stack in `src/llm/`; justify new dependencies in the PR or report.
- **Tool-calling product code:** validate tool inputs against schemas; surface tool errors to the model loop; log failures without leaking secrets.
- **Real backends:** integrations must hit real services or documented local runtimes—not silent mocks—in production paths.

## Orchestration (multi-agent)

- After tasks are **confirmed with the user**, respond **AYE** once that turn (team convention).
- **Max four concurrent threads** (orchestrator + subagents). Do not fan out beyond four.
- Prefer parallel subagents for independent work. If a Claude subagent fails for more than 3 times, retry with **codex** (`codex-rescue` or project Codex runtime).
- RTK (token reduction) hooks: `~/.claude/RTK.md`

## Git and delivery

- **Commit** every turn.
- **Push / PR** only when the user explicitly requests it.
- Commit messages: conventional, scoped, one logical change per commit; subject states *why*.
- Never commit secrets, `.env`, or large generated artifacts unless they are intentional, documented fixtures.

## OpenSpec

When the user drives OpenSpec changes, use skills under `.claude/skills/openspec-*` and respect `openspec/config.yaml`. Do not bypass the OpenSpec workflow for tracked changes unless the user directs a hotfix path.

## Maintaining this file

Add a rule here when Claude makes the **same mistake twice** or you repeat the same correction across sessions. Remove stale rows from **Repository map** when folders are deleted. Prefer `.claude/rules/<topic>.md` with `paths:` frontmatter for file-type-specific rules instead of growing this file past 200 lines.

