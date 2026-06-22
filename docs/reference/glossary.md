# Chess-Coach Agent — Context Glossary

Terms and their canonical meanings in this product. Implementation details belong elsewhere.

## Terms

### Skill
A Markdown file (`SKILL.md`) with `name:` and `description:` frontmatter and a body of instructions, steps, or commands the agent follows strictly when selected. Skills are text-only — no executable code.

### User skill
A skill the end user authored or imported into the local user skill folder (analogue of `.claude/skills/`). Loaded as-is by the agent, treated as trusted user intent.

### Plugin
An installable bundle that can contain skills, tools, hooks, agents, MCP servers, and configuration. Lives under the marketplace folder (analogue of `.claude/plugins/marketplace/<plugin>/`).

### Plugin skill
A skill that ships inside a plugin. Same `SKILL.md` shape as a user skill, but scoped to the plugin and only available when the plugin is installed and enabled.

### Tool
An executable function the agent can call with structured arguments. Tools ship inside plugins. Each tool has a name, description, and argument schema. The agent calls tools via `<tool>NAME args</tool>`.

### Official plugin
The first-party plugin shipped with the product. Bundles the chess-coach `SKILL.md` and the 9 chess tools (`board_state`, `move`, `undo`, `legal_moves`, `list_pieces`, `ask_chessbot`, `eval`, `best_move`, `review_move`, `threats`).

### Marketplace
The user-facing registry of installable plugins. Anthropic-style: browse, install, enable, disable, uninstall.

### Skill index
The list of `(name, description)` pairs the agent sees at the start of a turn. Built from all enabled user skills and plugin skills. The agent picks relevant entries from this index before loading any full skill body.

### Tool manifest
The list of `(name, description, args schema)` entries the agent sees at the start of a turn. Built from all enabled plugins' tools. Agent must only call tools from this manifest.

### Enabled
A skill or plugin is "enabled" when the user has installed it and not disabled it. Enable state is decided at install time, not per session or per turn. Once enabled, user skills and plugin skills merge into a single flat skill index — the agent treats them uniformly.

### Hook gate
The mechanism that decides whether a tool is callable in the current context. Two layers: (1) a static `applies_when` predicate declared in the tool manifest — if false, the tool is hidden from the manifest for that turn; (2) a runtime guard before execution — checks args, state, and permission, returns a tool-error if it fails. The agent never performs a self-check turn before each tool call.

### Adversarial-synthetic name
A skill or tool name in the training corpus that is deliberately unfamiliar (e.g. `tool_zb-19`, `skill-pluto-7`) so the model cannot pattern-match on the name and must read the description to decide. Target ~30% of training rows.
