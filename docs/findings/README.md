# findings/ — dated investigations (immutable)

Audits, triage, inspections, eval reports — a snapshot of what was true on a date. **Immutable once written.**

**Before adding a finding, ask: should this update a `reference/` doc instead, or become an `adr/` decision?** A standing pile of dated files nobody reopens is a graveyard. A finding is justified only when the **dated snapshot itself** has lasting value — e.g. an ML experiment/eval log where reproducibility matters. Otherwise: fold the conclusion into `reference/` (and let the investigation die), or capture the *why* as an ADR.

- Name: `YYYY-MM-DD-<topic>.md`. Line 1: `Parent: <relative-path>` or `Parent: none`.
- Sections in order: **Status**, **Scope**, **Evidence** (commands + outcomes), **Next** (numbered).
- Same topic, newer date → write a fresh file; `git mv` the prior one to `../legacy/` (move, never delete).
- Tools that emit reports write here with a fresh dated name — never overwrite an existing finding or a `reference/` doc.
- Durable *lessons* from a finding → distill into a memory one-fact file, don't leave them only here.
