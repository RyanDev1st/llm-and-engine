Parent: docs/findings/2026-06-21-routing-benchmark-interpretation.md

# Benchmark rerun (native-mode + version + e2b) → a harness refinement (2026-06-22)

**Status:** measured Kaggle results recorded; one prior claim CORRECTED; one harness fix shipped
(`9dabe183`). Completion-eval rerun (Cell 6.7, latest HEAD) in flight — its numbers land in a
follow-up.
**Scope:** the 6-hour Kaggle run (`kaggle_benchmark.ipynb`) produced version-trend, native-mode, and
e2b reports. This finding reads them for (a) the report and (b) what they say about refining the
serve harness. All numbers are RAW first-action routing (no CoachLoop) unless stated.

## Measured results

### Version trend (fast-mode adapter+harness, ~110-128 val rows/version)
| version | verb acc | exact-name | n |
|---|---|---|---|
| v2 (2026-06-17) | 10.0% | 10.0% | 110 |
| v3 (2026-06-18) | 99.2% | 51.2% | 127 |
| v4 (2026-06-19) | 99.2% | 66.4% | 128 |

### Native-mode FAIR test — each row scored in its TRAINED mode, 7 hard mode-dependent slices (n=172)
| metric | e4b-v4 adapter+harness | e4b base+harness |
|---|---|---|
| verb accuracy | **80.2%** | 41.9% |
| macro precision | 48.9% | 37.3% |
| weighted precision | 92.3% | 82.0% |
| exact-name accuracy | **50.0%** | 14.5% |
| format validity | 91.9% | 86.0% |

Per-slice (adapter): V1_R 84% · V1_S 80% · E 76% · F 56% · G 24% · H 18% · V1_T 8%.
Adapter beats base decisively on the same harness (+38.4 verb / +35.5 exact-name) — the SFT weights
clearly bought the routing.

### e2b condition (fast-mode, n=692) — DEGENERATE, do NOT publish as a model comparison
verb 3.9% · exact-name 0.3% · **format validity 17.1%**. Confusion: 553/646 gold-skill rows predicted
`none`; the one slice it "wins" is `V1_Q_no_skill_direct` (89%, where `none` is correct). This is the
prior E2B production model emitting mostly `none`/foreign tags on RAW first-action. **Most likely** the
E2B attn-only adapter's raw format is collapsed (see [[e4b-attn-only-freegen-collapse]]) and only the
serve scaffolding (extract_call recovery + tool_hints) rescued it in production — which a raw
first-action benchmark bypasses; a base-pairing bug in the disk-safe `--e2b-only` path is also possible.
NOT a valid E4B-vs-E2B generational comparison until the base pairing is verified. (E4B-v4's 91.9% raw
format validity vs 17.1% is, if the pairing checks out, the real generational story — but verify first.)

## CORRECTION to the 2026-06-21 interpretation — slice G was NOT "just a fast-mode artifact"
The prior finding called slice G's 0% a forced-fast EVAL ARTIFACT (the goal keyword `threats` leaking).
The native rerun PARTLY vindicates that (G: 0% → 24%, H: 7% → 18% once scored in trained mode) — but
**16 of 19 native G misses still emit `<skill>threats</skill>`** (and E→`skill:best_move`, H→
`skill:list_pieces`). So there is a REAL, mode-independent failure underneath: the model picks the
RIGHT tool name but wraps it in the `<skill>` verb (tool-as-skill confusion). Fast mode amplified it;
it did not create it. My earlier "resolved" framing was too generous.

## Harness refinement shipped (`9dabe183`) — verb coercion
Tool-as-skill is the dominant miss cluster (~28 of the native misses: G `threats` 16 + H `list_pieces`
7 + E `best_move` 5). In the live loop these only recovered IF the frozen model complied with the
`is a tool, not a skill` corrective — but the corpus NEVER trained that error type (`V1_H_error_recovery`
only trains `error: invalid_syntax` recovery). A CPU probe of the real CoachLoop confirmed the failure:
a non-compliant model re-emits the skill verb → `duplicate_tool_call` → break → forced **ungrounded
non-answer** (the tool never ran). Fix: `extract_call` now coerces `<skill>NAME</skill>` →
`<tool>NAME>` when NAME is a live tool (the model already chose the tool; fix the wrapper verb) — same
class as the existing tagless/malformed recovery, domain-neutral via the live manifest, honoring
[[chess-routing-fires-on-ood]]'s restraint (verb-fix, NOT routing). Read-only tools now run + ground
immediately; arg-needing tools hit a normal `error:` result the model DID train to recover from. The
routing benchmark scores RAW first-action, so the miss stays VISIBLE in the report for
diagnosis/retraining — only the live loop becomes robust. The completion eval will quantify the gain.

## Not fixed (with reason)
- **F `skill:code-reviewer` (7):** genuine WRONG-skill selection for a chess-rules prompt — a model
  routing error, not a verb/format error. The harness can't fix it without routing logic (restraint).
  A retraining signal.
- **V1_T `tool:python` (18), V1_S `tool:explain_plan` (4), V1_R `none` (3):** the model prefers
  verify-first / direct-answer over the gold plan verb. Plausibly still COMPLETES — the completion
  eval (`recovered`/`completed`) will decide; not a clear harness bug, so left for the data.

## COMPLETION EVAL — the harness validated end-to-end (latest HEAD, OOD STRESS, n=40)
The metric both peer reviews said was missing. Runs the FULL CoachLoop per row on the held-out
life-skills (unseen-domain) suite. **This is the number that proves the harness, not raw routing.**

| metric | rate | reading |
|---|---|---|
| first_ok | 35/40 (87.5%) | model's raw first action correct |
| **completed** | **39/40 (97.5%)** | every expected tool fired — the harness finishes the task |
| exec_ok | 28/40 (70.0%) | every expected tool's last result non-error — the one soft spot |
| args_ok | 40/40 (100%) | no missing-required / bad-enum calls |
| **grounded** | **40/40 (100%)** | every final answer cited the tool's result (Consumer C) |
| **recovered** | **4/40 (10%)** | wrong first route the loop self-corrected to a grounded answer |

**The headline:** the harness lifts first-action routing **87.5% → 97.5% task completion (+10pp)** on
UNSEEN domains, 100% grounded. The math is exact: 35 first_ok + 4 recovered = 39 completed — the
recovery layer (corrective + verb-coercion + grounding) accounts for the entire lift. This is the
product claim, measured end to end.

**Caveat — this run PREDATES `9dabe183` (verb-coercion) + `9402b59d` (skill-body extract-first).** So
it's the BASELINE; the final flight measures those.

**exec_ok 70% (12 rows) is the open item — and was UNDIAGNOSABLE** because the eval only aggregated.
Fixed (`c47d951a`): `run_completion` now logs each failing row (slice, gold, first action, failed
metric, the erroring results) and `_report` prints a "failing rows" table. The next flight will EXPLAIN
exec_ok instead of guessing. Hypothesis to confirm there (not asserted): some are tool-as-skill rows
whose corrective-error result counts against exec_ok pre-coercion, and/or decline rows where the model
over-acted into an erroring call — both addressed by the post-baseline fixes; the per-row log decides.

## Measured version trend (latest, supersedes the earlier partial numbers above)
v2 verb 10.7% / exact 9.9% (n=121) · v3 97.9% / 47.5% (n=141) · v4 **98.6% / 68.8%** (n=141). Same
diagnose→fix→measure story; v4 exact-name 68.8% is the best yet.

## Next
- **Final flight after the two post-baseline fixes:** re-run Cell 6.7 → read the new exec_ok + the
  failing-rows table; expect exec_ok up (tool-as-skill now coerced) and recipe-scaler over-ask gone.
- e2b: dropped from scope (not shipping) — do NOT spend cycles on it.
