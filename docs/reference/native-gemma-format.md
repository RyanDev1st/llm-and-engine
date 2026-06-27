# Native Gemma-4 wire format (tool-calling + thinking)

How the Gemma-4 chat template (`src/llm/models/gemma4_e2b/chat_template.jinja`) renders
tool-calling and reasoning. This is the on-the-wire format the v5-native corpus emits and
the serve parser must read. Verified empirically against the local tokenizer
(`GemmaTokenizer`), not assumed — see the verification note at the bottom.

## Control markers are SINGLE special tokens

Every native marker is one token id (so the loss mask can key on ids, and train/serve
tokenize identically):

| Marker | id | Marker | id |
| --- | --- | --- | --- |
| `<\|turn>` | 105 | `<turn\|>` | 106 |
| `<\|tool>` | 46 | `<tool\|>` | 47 |
| `<\|tool_call>` | 48 | `<tool_call\|>` | 49 |
| `<\|tool_response>` | 50 | `<tool_response\|>` | 51 |
| `<\|channel>` | 100 | `<channel\|>` | 101 |
| `<\|think\|>` | 98 | `<\|"\|>` | 52 |

Note the asymmetric pipe convention: opener `<\|X>`, closer `<X\|>`. The OLD Gemma format
`<start_of_turn>`/`<end_of_turn>` is NOT a single token here (7 tokens each) — do not use it.

## Turn / block grammar

```
<bos>
<|turn>system\n  [<|think|>\n]  {system text}  {<|tool>declaration:NAME{...}<tool|>}*  <turn|>\n
<|turn>user\n    {user text}  <turn|>\n
<|turn>model\n   [<|channel>thought\n{cot}\n<channel|>]  {<|tool_call>call:NAME{args}<tool_call|>}
                 {<|tool_response>response:NAME{...}<tool_response|>}  {final answer}  <turn|>\n
```

- **Tool call** (assistant): `<|tool_call>call:NAME{key:value,...}<tool_call|>`. String values wrap
  in the quote token: `move{san:<|"|>e4<|"|>}`. Ints/bools are bare: `best_move{depth:18}`. Keys are
  bare (no quote token). Multiple args comma-separated, **rendered in sorted key order**.
- **Tool response** (environment): `<|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>`
  for a string result, or `{k:v,...}` for a dict. Folded INTO the same model turn right after the call.
- **Thinking**: `<|channel>thought\n{cot}\n<channel|>` — rendered only on an assistant turn that
  (a) has `tool_calls` AND (b) is after the last user message, sourced from the message's `reasoning`
  field. `enable_thinking=True` additionally injects `<|think|>\n` at the top of the system turn.
- **Tool declaration** (system): `<|tool>declaration:NAME{description:<|"|>...<|"|>,parameters:{...}}<tool|>`,
  one per tool passed via the template's `tools=` argument.

## Critical template behaviors (the gotchas)

1. **`role="tool"` SURVIVES iff the preceding assistant message carries structured `tool_calls`.**
   The template forward-scans consecutive `role:"tool"` messages after a tool-calling assistant and
   renders them as native `<|tool_response>` blocks. A bare `role:"tool"` with no structured call before
   it is DROPPED (this was the v1–v4 [[gemma-template-drops-tool-role]] bug). => emit structured
   `tool_calls`, keep `role:"tool"` results; the old `remap_tool_messages` hack is REMOVED, not ported.
2. **`strip_thinking()` deletes any literal `<|channel>...<channel|>` from model CONTENT.** Thinking can
   never live as text in `content` — it must go through the `reasoning` field. => v5-native OMITS thinking
   from training text entirely; the base model's native reasoning is invoked at serve via
   `enable_thinking`, never trained as stubs.
3. **Consecutive assistant messages merge into ONE `<|turn>model ... <turn|>` block** (the template
   suppresses a duplicate `<|turn>model` when the previous non-tool message was also assistant). So a
   multi-step "think → call → (response) → answer" is one model turn.
4. **Generation prompt tail is `<|turn>model\n` for both enable_thinking True and False** — the only
   difference is the `<|think|>` token at the system top.

## Loss mask rule (v5-native)

Train (unmask) ONLY assistant-generated spans: the `<|tool_call>`(48)…`<tool_call|>`(49) runs and the
final answer content (after the last `<tool_response|>`(51) / `<tool_call|>`(49) up to `<turn|>`(106),
inside a model turn). Mask everything else: system, tool declarations, user, `<|channel>`(100)…
`<channel|>`(101) thought, and `<|tool_response>`(50)…`<tool_response|>`(51) env results.

## Reasoning modes

- `fast` → `enable_thinking=False` (no `<|think|>`, direct answer)
- `think` → `enable_thinking=True` (native channel reasoning)
- `auto` → `enable_thinking=True` + contract instruction to reason only on hard steps

## Verification

Reproduce with `scratchpad/probe_native_template.py` (CPU-only, loads just the tokenizer):
single-token check + `apply_chat_template` on a structured conversation. Re-run if the base
tokenizer is ever updated — these ids/strings are the contract both training and serving depend on.
