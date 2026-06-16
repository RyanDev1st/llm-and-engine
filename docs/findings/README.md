# findings/ — dated investigations (immutable)

Audits, triage, inspections, eval reports — a snapshot of what was true on a date. **Immutable once written.**

- Name: `YYYY-MM-DD-<topic>.md`. Line 1: `Parent: <relative-path>` or `Parent: none`.
- Sections in order: **Status**, **Scope**, **Evidence** (commands + outcomes), **Next** (numbered).
- Same topic, newer date → write a fresh file; `git mv` the prior one to `../legacy/` (move, never delete).
- Tools that emit reports write here with a fresh dated name — never overwrite an existing finding or a `reference/` doc.
- Durable *lessons* from a finding → distill into a memory one-fact file, don't leave them only here.
