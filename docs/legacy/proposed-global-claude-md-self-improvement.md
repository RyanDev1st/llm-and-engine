# Proposed addition to `~/.claude/CLAUDE.md` — Self-Improvement Module

> Paste the block below into your global `~/.claude/CLAUDE.md` (append as a new top-level section). It is intentionally short — instruction compliance **decays linearly** with rule count, and models follow <30% of rules perfectly in long agentic runs (Jaroslawicz et al., 2025), so every added line dilutes the rest. Keep this module under ~40 lines; push detail into `.claude/rules/` and skills.
>
> Evidence base: Reflexion (Shinn et al., NeurIPS 2023 — verbal post-mortems as gradient-free learning); PRISM (arXiv:2603.18507 — expert personas help tone/safety, *hurt* factual accuracy ~3–4%); Anthropic "Lessons from building Claude Code" (Gotchas as highest-signal content) and "Building Effective Agents" (simplicity; specialize via Skills + hooks, not a fatter system prompt). Full citations at the bottom.

---

```markdown
## Self-improvement (learn within and across sessions)

The files ARE the memory — there is no cross-session weight update. Treat
`.claude/lessons.md`, `.claude/rules/`, and skills as the episodic buffer.

### Mistake → lesson → rule (the loop)
- On any correction (user- or hook-initiated), append ONE entry to `.claude/lessons.md`:
  `## [YYYY-MM-DD] [tooling|arch|testing|style]` / `Mistake:` / `Context:` / `Rule:` / `Status: DRAFT`.
- Do NOT write the rule into CLAUDE.md or `.claude/rules/` on first occurrence.
- On the **second identical mistake**, surface the entry as a PROMOTE-CANDIDATE and ask
  the user to approve moving it to `.claude/rules/<topic>.md` (path-scoped). One occurrence ≠ a pattern.
- Agent-drafted rules are `[AGENT-DRAFT]` and never self-promote — a human commits them.
  (LLM-auto-generated rule files measure as net-negative vs. human-curated.)

### Adopt the role that fits — via Skills, not a global persona
- Do NOT declare a broad "you are a senior X" persona globally: it nudges tone but
  measurably hurts factual/recall tasks. Stay a generalist by default.
- When a task matches a specialist, dispatch the matching skill/subagent (its `description`
  is the routing signal; `context: fork` isolates it). Become the specialist only in that scope.

### Verify before "done" (deterministic, not vibes)
- Prefer a `Stop` hook that runs tests/lint and blocks completion on failure
  (guard `stop_hook_active` to avoid loops) over a prose "always test" rule.
- Prefer a `PostToolUse` hook returning `additionalContext` (e.g. typecheck output) so
  errors re-enter context immediately — in-session self-correction.
- Never claim success without showing the command + its output.

### End-of-session retrospective
- On "retro"/task end: list what worked, what failed + the rule that would have prevented it,
  then route — durable project facts → project CLAUDE.md; gotchas → the relevant SKILL.md.

### Keep this lean (anti-bloat)
- A rule earns its place only after preventing a real failure ≥twice AND human approval.
- Delete rules Claude already follows unprompted (noise, not signal).
- Audit for contradictions before big changes. If a file exceeds ~200 lines, it's too long —
  migrate to `.claude/rules/<topic>.md` with path globs.
```

---

## Why these and not others (decision notes)

| Mechanic | Keeps because | Source |
|---|---|---|
| Second-occurrence promotion | Avoids overfitting one session; matches Reflexion's "store then reuse" | Reflexion (NeurIPS 2023) |
| `lessons.md` append-only buffer | Inspectable, versioned, keeps CLAUDE.md thin | practitioner + Reflexion |
| Human-approval gate / `[AGENT-DRAFT]` | LLM-self-written rules give negative returns; blocks self-reinforcing false beliefs | self-correction lit (arXiv:2410.20513) |
| Skills over global persona | Personas help tone, hurt accuracy | PRISM (arXiv:2603.18507) |
| Hooks for verification | Prose rules are ~30% obeyed in long runs; hooks are deterministic | Jaroslawicz 2025; CC hooks docs |
| Gotchas-in-skills + retro | Anthropic's own highest-signal practice | Anthropic skills blog |

**Anti-patterns deliberately avoided:** add-a-rule-after-every-mistake (bloat → compliance cliff); global expert persona (accuracy loss); agent auto-committing its own rules (reward-hacking / lock-in); hooks without `stop_hook_active` (infinite loop); duplicating rules across CLAUDE.md/.cursorrules/copilot (drift).

## Sources
- Reflexion — https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
- Self-Refine (arXiv:2303.17651) · Self-correction not innate (arXiv:2410.20513)
- PRISM — personas help alignment, hurt accuracy (arXiv:2603.18507) · Persona double-edged (arXiv:2408.08631)
- Anthropic: Building Effective Agents — https://www.anthropic.com/research/building-effective-agents
- Anthropic: Lessons from building Claude Code / skills — https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills
- Claude Code hooks — https://code.claude.com/docs/en/hooks · best practices — https://code.claude.com/docs/en/best-practices
- Instruction-compliance decay (AgentIF / Jaroslawicz 2025) · "200 lines of rules, ignored" (dev.to)
- Session retrospective skill — https://github.com/accidentalrebel/claude-skill-session-retrospective
