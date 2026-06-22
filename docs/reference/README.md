# reference/ — how the system works NOW

Living docs describing current architecture, contracts, and designs. **Mutable in place** — when the system changes, edit the doc; do not date-stamp or fork it.

- One concept → one doc. No dates in filenames (these track HEAD, not a moment).
- When a design is retired (not just changed), `git mv` it to `../legacy/` and note why in `../legacy/README.md`.
- Add a row to `../README.md` in the same change.
