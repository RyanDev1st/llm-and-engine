# LLM tool-calling — product workspace

Project memory for Claude Code and other agents. Loaded every session. Keep **under 200 lines**, signal-dense, and verifiable. Update this file when layout, commands, or rules change. 

! Do not touch ./legacy/ ignore it.
<!-- Maintainer: add path-scoped rules under .claude/rules/ when this file grows. -->

## Mission

Ship **working software** for reliable **LLM tool use**: tool selection, JSON schemas, multi-turn loops, argument validation, and failure handling.

| In scope | Out of scope (unless user says otherwise) |
| --- | --- |
| Implementation, integration, release-quality behavior | Toy or simulated backends (label test fixtures explicitly) |
| Research notes for rationale and citations | Secrets in repo, commits, or chat |

## Repository map

| Path | Purpose |
| --- | --- |
| `CLAUDE.md` | Team agent instructions (this file) |
| `docs/` | Durable project documentation |
| `src/llm/` | Chess-coach product: dataset build, QLoRA trainer, Stockfish backend, web app (see `src/llm/README.md`) |
| `data/sft/` | SFT dataset: human slices + assembled `chess_assistant_v3_{train,val}.jsonl` |
| `legacy/` | Active product code, eval, tests, findings, plans |
| `legacy/superpowers/plans/` | Dated implementation plans |
| `legacy/findings/` | Evidence-backed status and experiment write-ups |
| `legacy/product_demo/` | Demo and training scripts |
| `legacy/tests/` | Pytest suite (`python -m pytest …`) |
| `openspec/` | OpenSpec config (`openspec/config.yaml`) |
| `.claude/skills/openspec-*` | OpenSpec propose / apply / explore / archive flows (committed) |
| `.claude/settings.local.json` | Personal permissions/overrides — **gitignored**, never commit |
| `.claude/scheduled_tasks.lock` | Scheduler lock — **gitignored**, ephemeral per machine/session |
| `.claude/worktrees/` | Agent worktrees — **gitignored**; do not edit unless the task names a path |

**Root policy:** only `CLAUDE.md`, `.gitignore`, and documented config files at repo root. All other artifacts live under `docs/`, `legacy/<feature>/`, or `openspec/`.

## Default workflow

Follow Anthropic’s **explore → plan → implement → verify** loop. Skip planning only when the change is one file and one obvious edit.

1. **Explore (read-only):** read relevant paths; state assumptions if anything is ambiguous.
2. **Plan:** list files to touch, verification commands, and risks. Confirm scope with the user when requirements are unclear.
3. **Implement:** minimal diff; match existing naming and patterns.
4. **Verify:** run commands in **Verification**; report pass/fail with command output summarized.
5. **Report:** update or create a report per **Reports** when the task produces findings, plans, or milestone status.

## Product principles

### File size (hard cap)

- No code source file may exceed **200 lines** (imports and blank lines count). Meaning all other files can exceed this. 
- If a change would exceed 200 lines: split into additional files in the **same feature folder** (next section). Never bypass the cap with comments or string concatenation.

### Feature folders (colocation)

- One capability → **one directory** under `legacy/` with a short domain name (≤3 words, `snake_case` or `kebab-case`), e.g. `legacy/lm_router/`, `legacy/tool_schema/`.
- All code for that capability stays in that folder. New capability → new folder. Extending a capability → existing folder only.
- Do not scatter the same feature across repo root, `docs/`, and unrelated `legacy/` siblings.

### Workspace hygiene (required before “done”)

1. No new root-level files except those listed in **Repository map**.
2. No `_copy`, `_old`, `temp`, or duplicate scripts.
3. Every new path is referenced by code, tests, or docs in the same change set.
4. New top-level or feature folder → add one row to **Repository map** in this file in the same change set.

### Reports (required layout)

- Path: `<scope-dir>/YYYY-MM-DD-<topic>-<artifact>.md`
- Allowed `<scope-dir>`: `docs/`, `legacy/findings/`, `legacy/superpowers/plans/`.
- Line 1: `Parent: <relative-path>` or `Parent: none`
- Sections in order: **Status**, **Scope**, **Evidence** (commands + outcomes), **Next** (numbered list)
- Same topic + same calendar date → append to the existing file **or** supersede as `…-v2.md` with a link to the prior file. Do not create a parallel sibling for the same topic.

## Verification

Claude performs best with explicit success criteria. Before claiming completion:

| Check | Command / rule |
| --- | --- |
| Tests (when `legacy/tests/` applies) | `python -m pytest legacy/tests/ -q` or the path named in the task |
| Lint / typecheck | Use project-standard commands when present; do not invent new tooling |
| Behavior | State expected output; if tests do not exist, give a manual repro the user can run |
| UI (if applicable) | Screenshot or browser snapshot compare against stated expectation |

If verification fails, fix or report the failure with the failing command and error excerpt. Do not claim “done” on assumptions.

## Engineering

- **Secrets:** never paste keys, cookies, tokens, or private URLs. Before any commit, confirm `.gitignore` covers `.env`, `*.pem`, `.claude/settings.local.json`, `.claude/scheduled_tasks.lock`, and `.claude/worktrees/`.
- **Dependencies:** prefer existing stack in `legacy/`; justify new dependencies in the PR or report.
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

