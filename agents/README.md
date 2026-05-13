# Agents — reporting contract

## Naming

Save reports as:

`findings/<lowercase_agent_or_role>_<short_slug>.md`

Examples: `findings/gpt_benchmarks.md`, `findings/codex_rlhf_tools.md`

## Minimum content

Use `../templates/AGENT_FINDING.md`. At least: **topic**, **5+ bullets**, **one evidence table**, **implications**, **open questions**.

## After each report

1. Append one line to `../research-log.tsv` (tab-separated).
2. Add a row to `../findings/00_INDEX.md`.
3. If you added a canonical paper or benchmark, add a stub to `../literature/NOTES.md`.

## Coordination

- Prefer **non-overlapping** subtopics per agent; if overlap occurs, cross-link at the top of the report.
- Conflicting claims: keep both; flag in `SYNTHESIS.md` under Contradictions.
