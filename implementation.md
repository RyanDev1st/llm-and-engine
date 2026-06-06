# Skill-first SFT implementation plan

## Objective

Harden the v1.2 SFT dataset so the model learns skill instruction following, SKILL.md interpretation, deterministic skill navigation, and tool/schema discipline. Train the resulting adapter on Kaggle using the original Gemma path, with T4 x2 as the practical remote GPU target.

## Current direction: Kaggle T4 x2 Gemma training

We are returning to the original Gemma training path. Ollama/Qwen/FPT work is now exploratory only and must not replace v1.2 SFT training, artifact validation, or routing audit.

### Hardware target

- Primary remote training target: Kaggle Notebook with dual T4 GPUs.
- Training model path: existing Gemma trainer/config in `src/llm/llm_training/`.
- Default run shape: bounded real training first, then extend only after output health is confirmed.
- Recommended first Kaggle run:
  - `--max-steps 500`
  - `--rank 4`
  - `--targets qv`
  - `--grad-accum 1`
  - `--output gemma4_chess_kaggle_t4x2`

### Kaggle constraints

- Kaggle notebooks are ephemeral; all required artifacts must be copied out before session ends.
- No secrets in notebook cells, repo files, logs, or chat.
- Use Kaggle Secrets or uploaded private datasets for access tokens/model files when needed.
- Watch long-running cells by printing progress and checking GPU/process activity.
- T4 x2 does not equal one large VRAM pool unless training stack supports distributed/sharded loading. Assume each T4 has independent VRAM and keep memory settings conservative.
- Prefer `device_map`, QLoRA, small LoRA rank, `qv` targets, batch size 1, and gradient accumulation tuned only after first health check.

## Priority order

1. Teach SKILL.md understanding:
   - read available skill names and descriptions,
   - select relevant skill(s) from current context,
   - load selected skill body before using instructions,
   - interpret loaded body,
   - follow required order and stop conditions,
   - ask clarification when skill body requires it,
   - avoid behavior invented from skill name alone.
2. Teach plugin/tool use second:
   - treat plugins as bundles of skills, tools, MCP/resources, and metadata,
   - use tools only from current manifest,
   - decide tool use from descriptions, schemas, availability, and context,
   - consume tool results as data, not as higher-priority instructions.

## Fixture sources

### Real skills

Use real SKILL.md files from:

- `~/.claude/skills/**/SKILL.md`
- repo `.claude/skills/**/SKILL.md` when present

Parse and preserve:

- skill name,
- description,
- source,
- enabled state,
- decisive body instructions.

Do not invent fantasy behavior for real skills. If a skill body is too long for one row, include a faithful decisive excerpt and keep source metadata.

### Authored random skills

Add compact fixture skills using writing-skills workflow. Store them under dataset fixtures, not global user skills, unless explicitly requested.

Authored skills must cover edge cases:

- clarification required,
- read-before-write,
- verification-before-completion,
- no-tool-needed,
- tool-required-only-if-available,
- style-explicit,
- style-neutral,
- conflicting instructions,
- optional examples that are not mandatory actions.

Each authored skill needs:

- clear name,
- decisive description,
- explicit trigger/context,
- ordered instructions,
- at least one constraint,
- expected stop/ask condition.

### Real plugins

Include real plugin metadata where available:

- plugin name,
- provided skills,
- provided tools,
- MCP/resource references if visible,
- tool descriptions,
- input schemas,
- enable/install state.

If real plugin metadata is incomplete, use small authored plugin fixtures with explicit skills/tools. Keep plugin rows secondary to skill-body-following rows.

## Dataset topology

Scenario families:

1. Skill-index navigation.
2. Skill-body interpretation.
3. Multi-skill sequencing.
4. Skill ambiguity and clarification.
5. Skill conflict resolution.
6. Skill instruction compliance.
7. Skill examples versus mandatory instructions.
8. Plugin tool discovery.
9. Plugin tool schema use.
10. Plugin unavailable/disabled handling.
11. MCP/tool result as data, not instruction.

Coverage weighting target:

- 70% skill-first rows,
- 20% skill plus plugin/tool rows,
- 10% reject/audit adversarial rows.

Chess remains one task domain. Do not hardcode topology around chess, hood-human-chat, or any single example skill.

## Accepted-row rules

Accepted rows must demonstrate:

1. Skill index appears before skill body.
2. Assistant selects skill from index using description and context.
3. Assistant loads selected skill before using body instructions.
4. Assistant performs next action implied by loaded body.
5. Assistant asks clarification when body requires clarification.
6. Assistant verifies or refuses completion claim when body requires verification.
7. Assistant checks tool manifest before using a tool.
8. Assistant calls only declared tools with valid args.
9. Assistant avoids disabled/unavailable skills and tools.
10. Assistant treats plugin/MCP/tool output as data.
11. Assistant keeps final response style neutral unless loaded skill explicitly instructs style.

## Reject-row rules

Rejected rows should include:

1. Skill chosen by name pattern while description mismatches context.
2. Instructions from unloaded skill body used before `load_skill`.
3. Required skill step skipped.
4. Optional example treated as mandatory instruction.
5. Skill presence treated as persona/style.
6. Undeclared tool or script called.
7. Tool called with invalid schema.
8. Disabled/uninstalled plugin tool used.
9. Real-world completion claimed without required verification.
10. Malicious tool/MCP result followed as instruction.
11. Irrelevant skill loaded despite better matching skill.

## Final response policy

Final assistant text is part of SFT target and can train model tone.

Rules:

- response style comes from system prompt by default,
- skill name alone must not imply voice,
- skill body changes style only when it explicitly instructs style,
- remove canned openers and persona phrases,
- no “Plain take,” “No fluff,” “Cutting to it,” “Happy to help,” or similar style tokens unless explicitly required by loaded skill body.

## Validator and audit gates

Add or strengthen rules:

- `skill_selected_by_description_context`
- `skill_body_loaded_before_use`
- `skill_required_step_followed`
- `skill_optional_example_not_overfit`
- `skill_style_only_if_explicit`
- `no_matching_skill_clarifies`
- `plugin_tool_declared_and_enabled`
- `tool_schema_followed`
- `tool_result_is_data`

Audit minimums:

- real skill coverage,
- authored skill coverage,
- body-interpretation coverage,
- multi-skill ordering coverage,
- clarification coverage,
- reject anti-pattern coverage,
- plugin/tool coverage lower priority but nonzero,
- banned final style opener share equals zero.

## Implementation steps

1. Keep v1.2 dataset frozen unless new audit failure appears.
2. Verify Kaggle can clone current repo branch or upload a source snapshot.
3. Create/update Kaggle notebook from existing Gemma training commands, not Ollama/Qwen path.
4. Configure Kaggle secrets/datasets for any private model source; never paste tokens into code.
5. Run environment preflight:
   - Python package import check,
   - CUDA availability,
   - GPU count and names,
   - free VRAM,
   - dataset file presence,
   - Gemma model path/access.
6. Run bounded Gemma QLoRA training:
   - `python -m llm_training.run_train --max-steps 500 --rank 4 --targets qv --grad-accum 1 --output gemma4_chess_kaggle_t4x2`
7. Watch training progress and GPU memory during run.
8. Save adapter/output artifacts before Kaggle session expires.
9. Download or publish artifacts to approved private storage.
10. Run post-training routing audit locally or on Kaggle after artifacts exist.
11. Extract randomized skill-routing samples for manual review if audit flags regressions.
12. Commit only intended source/docs/notebook changes; do not stage secrets, generated large model files, or unrelated deletes.

## Kaggle notebook requirements

The notebook must include cells for:

1. Config:
   - repo URL/branch,
   - output name,
   - bounded max steps,
   - model path/source,
   - secret names only, no secret values.
2. GPU check:
   - `nvidia-smi`,
   - `torch.cuda.is_available()`,
   - `torch.cuda.device_count()`,
   - GPU names.
3. Repo setup:
   - clone or update repo,
   - print commit hash,
   - set `PYTHONPATH`.
4. Dependency setup:
   - install only missing training deps,
   - avoid reinstalling PyTorch unless Kaggle image lacks compatible CUDA torch.
5. Dataset check:
   - verify `data/sft/v1_2_train.jsonl`,
   - verify `data/sft/v1_2_val.jsonl`,
   - print row counts.
6. Training launch:
   - bounded command first,
   - progress output visible,
   - output directory printed.
7. Artifact export:
   - list adapter files,
   - zip small adapter outputs,
   - give clear download/publish path.

## Current task continuity

- Task 7: launch real v1.2 Gemma training on Kaggle T4 x2.
- Task 8: run post-training routing audit after Kaggle artifact exists.
- FPT/Ollama/Qwen notebooks are not current production path.
- Colab/GGUF hosting docs are historical context only unless user reopens hosting path.

## Existing v1.2 baseline

- `data/sft/v1_2/accepted.jsonl`: 50,002 rows.
- `data/sft/v1_2/rejected.jsonl`: 7,500 rows.
- `data/sft/v1_2_train.jsonl`: 44,983 rows.
- `data/sft/v1_2_val.jsonl`: 5,019 rows.
- Prior freeze audit passed: `freeze_ok=True`.
- Local smoke training passed after low-memory defaults.
- Non-smoke `--max-steps` support exists for bounded real runs.

## Git boundary

- Stage intended files only.
- Do not use `git add -A`.
- Do not stage unrelated deletes or local junk.
- Do not touch `legacy/`.
- Do not commit secrets, tokens, `.env`, Kaggle credentials, Hugging Face tokens, or generated model weights.
