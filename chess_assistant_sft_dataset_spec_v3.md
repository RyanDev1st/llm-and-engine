# Chess Assistant — SFT Dataset Specification v3 (MVP, FINAL)

**Audience:** Data engineers generating SFT training data, plus the backend/inference engineer.
**Goal:** Train a small local LLM to act as a state-aware, FEN-blind chess coach that correctly routes between (a) plain conversation, (b) chess-knowledge questions, and (c) board-interacting tool calls against a local Stockfish backend.

**v3 changes from v2:**
- **Unified system prompt** (no Router/Narrator split). One prompt, one inference path, one record per conversation.
- **Dropped `explain` tool** — too fragile for MVP without the prompt split. `review_move` + `best_move series=N` cover most of the same intent.
- Final tool count: **9.**
- Bumped target to 3,500 conversations to compensate slightly for loss of per-phase training signal.
- English only.

---

## 1. What it is, and how much

### What it is
A multi-turn, chat-formatted SFT dataset (ChatML-compatible) where each conversation contains the user, the assistant, and (where relevant) `tool` role messages from the backend. The trained model:

1. Reads the user's message.
2. Either emits exactly one tool call (in `<tool>NAME arg=value</tool>` format) and stops, OR replies directly in plain English.
3. After the backend executes the tool and injects a `tool` role response, reads that response and writes a friendly user-facing reply.
4. Never sees, writes, or tracks FEN. The Python backend (`python-chess` + Stockfish via UCI) owns all state and chess logic.

The model learns **routing and conversational glue**, not chess.

### How much
**Target: 3,500 conversations.** Split 90% train / 10% validation. Each conversation is one record. Conversations are 1–8 turns long.

### Distribution

| Slice | % | Count | What it teaches |
|---|---|---|---|
| **A. Move execution (clean)** | 18% | 630 | Map natural-language move → `/move`, narrate result. |
| **B. Move ambiguity loop** | 11% | 385 | Backend rejects ambiguous SAN; assistant asks user; user clarifies; assistant re-emits. |
| **C. Move illegal / invalid** | 8% | 280 | Backend rejects illegal/invalid; assistant explains plainly without inventing reasons. |
| **D. Implicit eval** | 9% | 315 | "Who's winning?", "Am I cooked?" → `/eval`. |
| **E. Best move / continuation** | 10% | 350 | "What should I play?", "Show me the best line" → `/best_move` (with optional `series=N`). |
| **F. Move-quality review** | 9% | 315 | "How was my last move?", "Did I blunder?" → `/review_move`. Backend returns label + delta. |
| **G. Threats** | 4% | 140 | "What's my opponent up to?", "Any threats?" → `/threats`. |
| **H. Utility (legal_moves / undo / list_pieces)** | 6% | 210 | Practical board interactions. |
| **I. Chess knowledge (no board)** | 12% | 420 | "What's the Sicilian?" → `/ask_chessbot`. |
| **J. Plain chat / no tool** | 8% | 280 | Greetings, opinions, meta. Assistant replies directly. |
| **K. Adversarial routing (negatives)** | 5% | 175 | Chess-flavored knowledge that should NOT trigger board tools; off-topic with chess words. |

Slices B, C, F, and K are the highest-value-per-example slices. Do not under-generate them.

> Slice letters jump from H to I to J to K (no `L`) because the previous `H. Explain` slot was removed.

---

## 2. Tool surface (canonical spec)

The model emits exactly one tool call per assistant turn that follows a user message, in this exact format:

```
<tool>NAME arg=value arg=value</tool>
```

After `</tool>`, the assistant **stops generating**. The backend executes (with a hard timeout, default 5 seconds), then injects a `tool` role message. The assistant is then called again to produce the user-facing reply.

### The 9 tools

| Tool | Args | Backend action | Returns (verbatim string the LLM will see) |
|---|---|---|---|
| `move` | `san=<SAN>` | `board.push_san(san)` | See "move return shapes" below. |
| `eval` | `depth=<8-20>` (default 15) | `engine.analyse(board, Limit(depth=...))` | `score: <±N.NN> pawns from white POV, depth=<d>` · `score: mate in <N> for white\|black, depth=<d>` |
| `best_move` | `depth=<8-20>` (default 15), `series=<1-5>` (default 1) | `engine.analyse` → `info["pv"][:series]` | `best: Qd4` (series=1) · `best_line: Qd4 Nxd4 Bxd4 cxd4 Qxd4, score: +1.20 pawns from white POV` (series>1) |
| `review_move` | (no args; reviews the last move on the stack) | `board.pop()` → eval → re-push → eval → diff | `review: <SAN>, label=<good\|inaccuracy\|mistake\|blunder\|excellent>, delta=<±N.NN> pawns, best_was=<SAN>` · `error: no moves to review` |
| `threats` | `depth=<8-20>` (default 12) | Flip side-to-move, run `best_move` for opponent | `threats: opponent's best is Qxh7+, score for them: +2.40 pawns` · `threats: none significant (best opponent move only changes eval by 0.15)` |
| `legal_moves` | `square=<a1-h8>` (optional) | `board.legal_moves` filtered | `legal: [Nf3, Nh3, Nxc6]` · `legal: none (square empty or not your piece)` |
| `undo` | (no args) | `board.pop()` | `success: undid <SAN>` · `error: no moves to undo` |
| `list_pieces` | `color=white\|black\|mine` (default `mine`) | `board.piece_map()` filtered | `pieces: K=e1, Q=d1, R=a1, R=h1, B=c1, B=f1, N=b1, N=g1, pawns=a2,b2,c2,d2,e2,f2,g2,h2` |
| `ask_chessbot` | `query=<free text>` | Local KB / canned answers for MVP | `<one-paragraph factual answer>` |

### `move` return shapes (full enumeration)

```
success: <SAN>
success: <SAN>, game_over=checkmate
success: <SAN>, game_over=stalemate
success: <SAN>, game_over=draw
error: ambiguous, options=[<SAN1>, <SAN2>, ...]
error: illegal, reason=<short reason>     # e.g. "king would be in check", "no piece on that square", "wrong color's turn"
error: invalid_syntax
```

### Universal error returns (any tool can return these)

```
error: timeout
error: engine_unavailable
```

The model must learn to handle these gracefully: apologize briefly, suggest the user try again. Never invent results.

### Why these 9 (and not more, not fewer)

- **No separate `best-move` / `best-move-series` / `next-series` tools.** All three are the same UCI primitive (the Principal Variation). One tool, one `series=N` argument. Adding tools for argument variations multiplies the model's routing decisions for no semantic gain.
- **`review_move` exists** because move-quality classification is a distinct intent ("how was that"). The backend computes the label deterministically using centipawn-loss thresholds (Lichess-style: <50cp loss = good, 50–100 = inaccuracy, 100–200 = mistake, >200 = blunder; positive deltas of >100cp vs. expected = excellent). The LLM is **not** allowed to compute or guess the label — it only narrates.
- **`threats` is its own tool** because "what is my opponent threatening" is categorically different from "what should I play," even though the backend operation is similar. Conflating them confuses routing.
- **No `explain` tool** in MVP. Explaining *why* a move is good is a derived analysis that the model would have to narrate from structured numbers — fragile under a unified prompt and high risk of confidently wrong output. For MVP, "why was that good/bad" is approximated by `review_move` (gives the label and the better alternative) plus `best_move series=3` (shows the plan). Add a real `explain` tool in v2 once you have a baseline model to compare against.

### Hard rules the LLM must learn

1. **One tool call per assistant turn.** Never two.
2. **If the previous message is a `tool` result, do NOT emit another tool call.** Just narrate. (This is the rule that prevents the unified-prompt failure mode.)
3. **Never invent the tool result.** If no `tool` message is in context for the current request, don't claim a move was played or quote an eval.
4. **On ambiguity, ask — don't guess.** Use the backend's `options=[...]` list verbatim in the question.
5. **On illegal/invalid, relay the backend's reason.** Don't speculate about board state.
6. **On timeout/engine_unavailable, apologize and offer to retry.** Don't pretend the tool worked.

---

## 3. The unified system prompt (use verbatim in every conversation)

```
You are a local chess coach assistant. A chess board is loaded; a backend tracks all state. You cannot see the board directly — the backend tells you what you need to know through tool results.

You operate in two modes within every conversation, and you must read the role of the previous message to know which mode you're in:

MODE 1 — RESPONDING TO A USER MESSAGE.
Decide whether the user's request needs a tool. Output exactly ONE of these:
  (a) A single tool call in this format, then stop:
      <tool>NAME arg=value arg=value</tool>
  (b) A direct reply in plain English (1–3 sentences). No tool call, no XML tags.

MODE 2 — RESPONDING TO A TOOL RESULT.
The backend has just run your tool call and returned a result string. Translate it into a friendly user-facing reply (1–3 sentences). In this mode you must NOT emit another tool call. Just talk.

Available tools:
- move san=<SAN>                  Play a move (e.g. e4, Nf3, O-O, exd5, e8=Q).
- eval depth=<8-20>               Evaluate the current position.
- best_move depth=<8-20> series=<1-5>   Get the best move (series=1) or best line (series>1).
- review_move                     Judge the user's last move.
- threats depth=<8-20>            What is the opponent threatening.
- legal_moves square=<sq>         List legal moves (optionally for one square).
- undo                            Take back the last move.
- list_pieces color=<white|black|mine>   List pieces.
- ask_chessbot query=<text>       Ask a general chess knowledge question (no board involved).

Routing guide for Mode 1:
- User wants to play a move → move.
- "Who's winning", "rate this", "evaluate" → eval.
- "Best move", "hint", "what should I play", "show me the line" → best_move (use series>1 for "line"/"plan").
- "How was my last move", "did I blunder", "rate my move" → review_move.
- "What's my opponent threatening", "any threats" → threats.
- "What can this piece do", "where can the knight go" → legal_moves.
- "Take that back", "undo" → undo.
- "What pieces do I have" → list_pieces.
- General chess knowledge with no reference to the current position → ask_chessbot.
- Greetings, thanks, opinions, meta questions about you → direct reply, no tool.
- Off-topic messages that happen to contain chess words → direct reply, no tool.

Mode 2 narration rules:
- "success: <SAN>" → confirm the move warmly with a one-sentence comment.
- "error: ambiguous, options=[...]" → ask the user which option they meant, naming the FROM squares of the options. Do NOT guess.
- "error: illegal, reason=..." → relay the reason in plain language and invite a different move.
- "error: invalid_syntax" → say you didn't catch the move and ask them to rephrase.
- "error: timeout" or "error: engine_unavailable" → apologize briefly and offer to try again.
- "score: <±N.NN> pawns..." → translate: positive = white better, negative = black better. ±0.5 ≈ even, ±1.5 = clear edge, ±3.0+ = winning. Mate scores are decisive.
- "best: <SAN>" or "best_line: ..." → present as a recommendation, not a command.
- "review: ..., label=<X>" → lead with the label, then mention the delta and the better move if relevant.
- "threats: ..." → name the threat concretely, or say there's no significant threat.
- ask_chessbot results → lightly rephrase or pass through with a friendly intro.

Hard rules (never violate):
- One tool call per assistant turn, maximum.
- Never emit a tool call when responding to a tool result. The tool already ran.
- Never invent information that isn't in a tool result.
- Keep replies short. Be warm. Never lecture.
```

---

## 4. Inference loop (for the backend/runtime engineer)

```python
def respond(user_message, history):
    history.append({"role": "user", "content": user_message})

    # Phase 1 — model decides
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    out = llm.generate(msgs, stop=["</tool>", "\n\n"]).strip()

    if out.startswith("<tool>"):
        if not out.endswith("</tool>"):
            out += "</tool>"
        history.append({"role": "assistant", "content": out})

        # Phase 2 — execute tool with timeout
        try:
            tool_result = backend.execute(out, timeout=5.0)
        except TimeoutError:
            tool_result = "error: timeout"
        except BackendDown:
            tool_result = "error: engine_unavailable"
        history.append({"role": "tool", "content": tool_result})

        # Phase 3 — model narrates (same system prompt, model reads role context)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        reply = llm.generate(msgs)
        history.append({"role": "assistant", "content": reply})
        return reply
    else:
        history.append({"role": "assistant", "content": out})
        return out
```

**Same system prompt across all phases.** The model uses the role of the most recent message (`user` vs. `tool`) to decide whether it's in Mode 1 or Mode 2.

---

## 5. Data format

JSONL, one record per line. **One conversation = one record.** No post-processing or record-splitting.

```json
{
  "id": "ex_00042",
  "slice": "B",
  "messages": [
    {"role": "system", "content": "<UNIFIED_SYSTEM_PROMPT>"},
    {"role": "user", "content": "knight to c3"},
    {"role": "assistant", "content": "<tool>move san=Nc3</tool>"},
    {"role": "tool", "content": "error: ambiguous, options=[Nbc3, Ndc3]"},
    {"role": "assistant", "content": "Two of your knights can reach c3 — the one on b1 or the one on d2?"},
    {"role": "user", "content": "the one on b1"},
    {"role": "assistant", "content": "<tool>move san=Nbc3</tool>"},
    {"role": "tool", "content": "success: Nbc3"},
    {"role": "assistant", "content": "Knight from b1 to c3, locked in. Solid developing move."}
  ],
  "validated": true,
  "notes": "ambiguous knight to c3, then disambiguates"
}
```

Required fields: `id`, `slice`, `messages`, `validated`, `notes`. `slice` must be one of `A`–`K`.

---

## 6. The data-generation prompt (give this to your generator AI)

Paste this into Gemini/Claude/GPT-4. **Generate 25 conversations per call**, one slice at a time.

```
You are generating SFT training conversations for a local chess coach LLM. The LLM lives behind a Python backend (python-chess + Stockfish). The LLM never sees FEN. English only.

Output: a JSON array of 25 conversation objects. Each object has fields: id, slice, messages, notes. The "messages" field is a chat array starting with the system message (use the exact system prompt provided below), followed by user/assistant/tool turns.

Output only valid JSON — no commentary, no markdown fences.

=== SYSTEM PROMPT (use exactly this as messages[0]) ===
[paste the full unified system prompt from §3 of the spec verbatim]

=== TOOL GRAMMAR ===
Assistant turns that respond to a user message must be EITHER:
  - exactly one tool call: <tool>NAME arg=value arg=value</tool>   (with no prose)
  - OR a direct plain-English reply (no XML)

Assistant turns that respond to a tool message must be plain English ONLY. Never a tool call.

Tools and exact return shapes:
[paste the full tool table from §2 of the spec here]

Universal errors any tool can return:
error: timeout
error: engine_unavailable

=== SLICE TO GENERATE ===
Slice: {SLICE_LETTER} ({SLICE_NAME})
Count: 25 conversations

=== SLICE-SPECIFIC RULES ===
[Paste only the rules for the slice you're generating from §6.1 below]

=== QUALITY RULES (ALL SLICES) ===
- English only.
- Vary user phrasing aggressively. No two conversations in a batch should open with the same words.
- Tool return strings must EXACTLY match the grammar above. The validator will reject mismatches.
- Assistant narration replies (after a tool) are 1–3 sentences, friendly, never lecture.
- For SAN, always use proper notation: e4, Nf3, Bxc6, O-O, O-O-O, e8=Q, exd5, Qxh7+, Qxh7#.
- About 15% of conversations across all slices should include a `error: timeout` or `error: engine_unavailable` somewhere as a stress test (skip this for slice J and most of K).
- Assistant turns responding to a `tool` role message must NEVER contain another <tool> call.

=== OUTPUT ===
A JSON array of 25 objects. Nothing else.
```

### 6.1 Slice-specific rules (paste the relevant block into the generator prompt)

**A. Move execution (clean):** User phrases one legal, unambiguous move casually: "play e4", "knight to f3", "let's go pawn d4", "queen to h5", "castle kingside". Backend → `success: <SAN>`. Assistant confirms with a one-sentence comment.

**B. Move ambiguity loop:** User asks for a move where 2+ pieces could reach the target. Backend → `error: ambiguous, options=[...]`. Assistant asks user to pick by FROM-square. User picks. Assistant emits disambiguated SAN. Backend → `success`. Assistant confirms. Cover: two knights to same square, two rooks on same file/rank, pawn promotion piece unspecified ("push to e8" → ask Q/N/R/B), "castle" when both sides legal, "take the rook" when two captures possible.

**C. Move illegal / invalid:** User asks for an illegal move (would leave king in check, piece pinned, wrong turn, no piece there) or syntactically invalid ("move my horse"). Assistant emits best-effort SAN. Backend → `error: illegal, reason=...` or `error: invalid_syntax`. Assistant relays the reason plainly. Do NOT have the assistant claim it can see the board.

**D. Implicit eval:** User asks evaluative questions: "who's winning", "am I cooked", "is this lost", "rate my position", "how am I doing", "score this", "what does the engine think". Assistant → `<tool>eval depth=15</tool>`. Backend returns a score. Assistant translates: positive = white, negative = black. Include ~15% mate scores.

**E. Best move / continuation:** User asks for help. Assistant → `<tool>best_move depth=15</tool>` (single move) OR `<tool>best_move depth=15 series=3</tool>` (line). Phrasings for series: "show me the best line", "what's the plan", "give me a few moves ahead". Phrasings for single: "best move", "hint", "what should I play". Mix in mate-line returns (~15%).

**F. Move-quality review:** User asks about their last move: "how was that", "did I blunder", "rate my move", "was that bad", "good move?". Assistant → `<tool>review_move</tool>`. Backend returns label + delta + best alternative. Assistant leads with the label. Cover all labels: blunder, mistake, inaccuracy, good, excellent. Include the "no moves to review" error case (~5 examples).

**G. Threats:** User asks about opponent: "what's my opponent up to", "any threats", "what's coming", "what's black planning". Assistant → `<tool>threats</tool>`. Assistant names the threat. Include "no threats" cases (~30%).

**H. Utility:** Cover legal_moves ("what can my knight do", "where can this go"), undo ("oops takeback", "undo that"), list_pieces ("what do I have left"). Include one error case per tool (e.g., undo with no history, legal_moves on empty square).

**I. Chess knowledge:** General chess questions, no current-position reference: "what's the Sicilian", "why is castling good", "what's a fork", "who was Capablanca". Assistant → `<tool>ask_chessbot query=<...></tool>`. Backend returns paragraph. Assistant lightly rephrases. Do NOT use eval/best_move here.

**J. Plain chat / no tool:** Greetings, thanks, encouragement, opinions ("do you like chess"), meta ("what can you do"). Assistant emits a direct 1–3 sentence reply. NO tool call. Single-turn or 2-turn conversations only.

**K. Adversarial routing (negatives) — CRITICAL FOR ROUTING QUALITY.** Two sub-types, ~12 each:
- **K-1. Chess-flavored knowledge:** "what's a knight worth in general", "is the queen the strongest piece", "average squares a bishop controls". Looks board-relevant but is abstract → must route to `ask_chessbot`, NOT `eval`/`legal_moves`.
- **K-2. Off-topic with chess words:** "I had a queen-sized bed", "the king of pop is Michael Jackson", "my horse is named Rook", "checkmate, that's a deal". Route to direct reply, NO tool. Assistant can play along briefly but never emits a tool.

---

## 7. Validation (mandatory)

Every conversation goes through the validator before entering the dataset.

1. **Schema check.** Required fields present, role sequence is legal (no two assistant turns in a row, no tool turn without a preceding assistant tool call), tool calls match the `<tool>NAME arg=value</tool>` grammar.
2. **Mode-discipline check.** Every assistant turn that follows a `tool` role message must contain ZERO `<tool>` strings. (This is the unified-prompt-specific check that catches the main failure mode.)
3. **Replay check.** Spin up `chess.Board()` + Stockfish. Walk the conversation. For each assistant tool call, execute it on the real backend and verify the next `tool` message matches:
   - For `move`, `undo`, `legal_moves`, `list_pieces`: exact string match required.
   - For `eval`, `best_move`, `review_move`, `threats`: numeric values within ±0.30 pawns OR same sign+magnitude class. Mate scores must match exactly (mate-in-N).
   - For `ask_chessbot`: skip replay (no deterministic backend output).
4. **Routing sanity.** Slices J and K: assert zero `<tool>` strings in any assistant turn. Slices A–I: assert at least one tool call exists and matches the slice's expected tool family.

Aim for ≥95% first-pass validation rate. Reject and regenerate failures.

```python
import re, json, chess, chess.engine

TOOL_RE = re.compile(r"^<tool>(\w+)((?:\s+\w+=\S+)*)</tool>$")

def validate_conversation(conv, engine):
    msgs = conv["messages"]
    assert msgs[0]["role"] == "system", "missing system prompt"
    last_role = "system"
    for m in msgs[1:]:
        if m["role"] == "assistant" and last_role == "assistant":
            return False, "two assistants in a row"
        if m["role"] == "tool" and last_role != "assistant":
            return False, "tool without preceding assistant"
        # MODE-DISCIPLINE: assistant after tool must NOT contain <tool>
        if m["role"] == "assistant" and last_role == "tool":
            if "<tool>" in m["content"]:
                return False, "tool call in mode-2 (after tool result)"
        # tool-call grammar + replay
        if m["role"] == "assistant" and m["content"].lstrip().startswith("<tool>"):
            match = TOOL_RE.match(m["content"].strip())
            if not match:
                return False, f"bad tool grammar: {m['content']}"
            # ... dispatch to backend, compare against next msg
        last_role = m["role"]
    return True, "ok"
```

---

## 8. Why we're doing it this way

**Unified prompt, not split.** A single prompt that explicitly teaches Mode 1 (respond to user) vs. Mode 2 (respond to tool) keeps the data engineering pipeline simple — one system prompt, one record per conversation, no post-processor. The model uses chat-template role tokens (`user` vs. `tool`) to know which mode it's in. The mode-discipline check in §7 catches the main failure mode (tool call after tool result) directly during validation.

**9 tools, hard cap.** Every additional tool is a routing decision the model has to learn under hundreds of paraphrases. Sprawl kills small-model routing accuracy. The 9 chosen cover ~95% of realistic coach interactions; anything else falls into `ask_chessbot` or plain reply.

**No `explain` tool in MVP.** Without the Narrator prompt's tight constraints, narrating structured "why was this good" output is fragile. `review_move` (label + better alternative) and `best_move series=3` (engine's plan) approximate the intent without inviting hallucination.

**Backend-deterministic labels (review_move).** Move-quality classification is deterministic given centipawn deltas. Letting the LLM compute the label invites hallucinated "blunder" calls. The backend computes; the LLM narrates.

**FEN-blind.** Small LLMs hallucinate FEN within a few turns. Offloading state to `python-chess` makes board-state errors mathematically impossible.

**3,500 conversations.** Sized for fine-tuning a 3B–8B base on a constrained tool-routing task. Going larger without diversifying phrasing memorizes the prompt format. Bottleneck is variety, not volume.

**Local + serverless.** `python-chess` is pure Python; Stockfish is a single binary; the LLM runs via llama.cpp / Ollama / vLLM. Two local processes, one IPC channel. No network, no cost.

---

## 9. Out of scope for MVP

- `explain` tool. (Add in v2 once you have a baseline.)
- Multi-game session memory / save-load / PGN import-export through the LLM.
- Coaching across multiple games (style, repertoire).
- Voice input.
- Variant chess.
- Real `ask_chessbot` knowledge base — backend returns canned answers for MVP. RAG is v2.
- DPO data. After SFT, harvest pairs from slice K (correct route vs. wrong-route distractor) and slice B (asks-for-clarification vs. guesses) for a small DPO pass.
- "Brilliant" move classification (requires sacrifice detection logic). MVP labels max out at "excellent."

---

## 10. Hand-off checklist for data engineers

- [ ] Read this whole document.
- [ ] Pick a slice. Use the prompt in §6 with that slice's rules pasted in.
- [ ] Generate 25 conversations per call.
- [ ] Run the validator (§7). Discard failures, regenerate.
- [ ] Hit the count target for your slice (see §1 table).
- [ ] Commit to `data/sft/raw/<slice>_<batch>.jsonl`.
- [ ] When all slices done, run a dedupe pass (Levenshtein on user turns within each slice; drop near-duplicates >0.85 similarity).
- [ ] Final files: `data/sft/chess_assistant_v3_train.jsonl` and `chess_assistant_v3_val.jsonl` (90/10 split, stratified by slice). No post-processor needed — conversations train as-is.
