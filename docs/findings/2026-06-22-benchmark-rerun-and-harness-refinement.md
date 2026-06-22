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

## Next
- Completion-eval rerun (Cell 6.7, latest HEAD) → record `completed/grounded/recovered`; expect the
  tool-as-skill cluster to show as completed/recovered now.
- Verify the e2b base pairing before any E4B-vs-E2B claim in the report.
- Final benchmark flight after these fixes (per the plan).
