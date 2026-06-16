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

The product is a **general agentic HARNESS operator** = an LLM that **chooses among the skills+tools listed in its prompt and thinks to complete a goal in ANY domain**, narrating tool results without computing them. Chess-coach is the **flagship demo domain, one of many** — not the whole product. The contract (`src/llm/llm_training/system_prompt.py`) has TWO verbs, one action per step: **`<skill>NAME</skill>`** loads a listed skill's body into context; **`<tool>NAME args</tool>`** calls a function (there is NO `load_skill` tool). Reasoning runs in 3 modes via a prompt signal: **fast** (no `<think>`), **think** (`<think>` every step), **auto** (`<think>` only on hard decisions — interleaved). Corpus mix is ~75% general / ~25% chess. Goal: train Gemma 4 **E4B** QLoRA via **Unsloth** (anomaly-guarded, seq 1664) on **Colab/Kaggle T4** → LoRA adapter → serve **q4_0 GGUF locally on the RTX 4060** (E2B fallback). Active plan files at root: `implementation.md` and `handoff.md`.

| Path | Purpose |
| --- | --- |
| `CLAUDE.md` | Team agent instructions (this file) |
| `src/llm/llm_dataset/v1/` | **ACTIVE** SFT generator. Spec = `contracts.py`; `profiles.py` writes the v1_2 corpus. Source of truth for harness behavior |
| `data/sft/v1_2_train.jsonl`, `data/sft/v1_2_val.jsonl`, `data/sft/v1_2/` | **ACTIVE** SFT corpus (split + accepted/rejected). The ONLY corpus trainers read |
| `src/llm/llm_training/` | QLoRA trainer (`run_train.py`, `train_cuda.py`), loader, `eval_routing.py`, `system_prompt.py` |
| `src/llm/backend/` | Environment the agent calls: tool executor + Stockfish engine + HTTP server + `sandbox.py` (the domain-neutral `python` verification tool — isolated subprocess, used to ground/verify a claim by running a script; Stage-0 keystone). Live skills catalog = `src/llm/skills/` (loaded by `skills.load_skills`). Plugin bundles = `backend/plugins/` (each contributes tools+skills+hooks; registry aggregates enabled ones into the served manifest — tests cross-bundle routing). Dev runtime: `model_server.py` (persistent weights service) + `model_remote.py` + `dev_serve.py` (weightless app auto-restart via `CHESS_MODEL_SERVER`) |
| `src/llm/skills_demo/` | 40 chess SKILL.md fixtures for routing tests + presentation demo. `_specs.py` (data) + `_generate.py` (renderer); NOT auto-loaded by the backend (pass as `load_skills` root) |
| `src/llm/gemma_chat_site/` | Web app (board + chat UI) |
| `src/llm/runtime/llamacpp/` | Bundled llama.cpp for GGUF serving |
| `src/engine/research/` | Standalone custom chess engine (alt backend) |
| `docs/` | Durable docs + dated reports; index = `docs/README.md`. Superseded docs → `docs/legacy/` (tracked archive) |
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
5. **Throwaways** (probes, one-off scripts, sample outputs) → name them `scratch_*` (already gitignored). Never commit them, never leave them in `src/`; promote to a real path (with refs) or delete before "done".

### Reports (required layout)

- Path: `<scope-dir>/YYYY-MM-DD-<topic>-<artifact>.md`
- Allowed `<scope-dir>`: `docs/` (default).
- Line 1: `Parent: <relative-path>` or `Parent: none`
- Sections in order: **Status**, **Scope**, **Evidence** (commands + outcomes), **Next** (numbered list)
- Same topic + same calendar date → append to the existing file **or** supersede as `…-v2.md` with a link to the prior file. Do not create a parallel sibling for the same topic.

### Docs hygiene (MANDATORY — keep `docs/` legible so the human knows what to do)

`docs/` holds ONLY current, in-use documentation. Superseded docs (old handoffs, finished
phase reports, abandoned-path runbooks, retired designs) move to **`docs/legacy/`** — a
tracked, discoverable archive (distinct from the gitignored `legacy [ignore]/` code/data bin).

1. **One index:** `docs/README.md` lists every active doc with a one-line purpose. Add a row when you add a doc; remove it when you archive one — in the **same change**.
2. **Archive on supersede:** when a doc is replaced, finished, or its path is abandoned, `git mv` it to `docs/legacy/` and add a one-line "why retired" note in `docs/legacy/README.md`. Move, never delete.
3. **No dangling refs:** never leave live code or an active plan pointing at a doc you moved to `docs/legacy/`. If a tool WRITES a report, it must write a fresh `docs/YYYY-MM-DD-…` path, not overwrite an archived one.
4. **Root plan files stay at root:** `implementation.md` + `handoff.md` are the active plan set and do NOT live under `docs/`.
5. **Check on cleanup:** when asked to tidy, audit `docs/` — anything not reachable from `docs/README.md` and not current → `docs/legacy/`.

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

## Git and delivery

- **Commit** every turn.
- **Push / PR** only when the user explicitly requests it.
- Commit messages: conventional, scoped, one logical change per commit; subject states *why*.
- Never commit secrets, `.env`, or large generated artifacts unless they are intentional, documented fixtures.

## Maintaining this file

Add a rule here when Claude makes the **same mistake twice** or you repeat the same correction across sessions. Remove stale rows from **Repository map** when folders are deleted. Prefer `.claude/rules/<topic>.md` with `paths:` frontmatter for file-type-specific rules instead of growing this file past 200 lines.

