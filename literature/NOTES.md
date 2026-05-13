# Literature and primary sources — scratchpad

Add one block per major source. Keep URLs full.

## Template entry

**Title:**  
**URL:**  
**Year:**  
**Why it matters for tool calling / agents:**  
**Limitations:**  

---

## Entries

**Title:** tau-bench
**URL:** https://github.com/sierra-research/tau-bench
**Year:** 2024
**Why it matters for tool calling / agents:** Defines tool-agent evaluation as a domain environment with policies, tools, tasks, and user simulation rather than isolated function-call formatting.
**Limitations:** Older benchmark line; related repos suggest using newer tau2/tau3 variants for current work.

---

**Title:** tau2-bench
**URL:** https://github.com/sierra-research/tau2-bench
**Year:** 2025
**Why it matters for tool calling / agents:** Reinforces environment-first agent evaluation with domain policies, tool execution, and task outcomes.
**Limitations:** General task domains, not chess-specific.

---

**Title:** ToolBench / ToolLLaMA
**URL:** https://github.com/OpenBMB/ToolBench
**Year:** 2023
**Why it matters for tool calling / agents:** Provides large-scale tool-use data and evaluator framing for pass/win rates across real API tasks.
**Limitations:** Real/external APIs introduce instability; should not be copied directly into deterministic chess correctness eval.

---

**Title:** StableToolBench
**URL:** https://github.com/THUNLP-MT/StableToolBench
**Year:** 2024
**Why it matters for tool calling / agents:** Shows why stable mirrored/virtual APIs matter for reproducible tool-use evaluation.
**Limitations:** Still general API benchmark, not chess-engine-specific.

---

**Title:** Gorilla / Berkeley Function-Calling Leaderboard
**URL:** https://github.com/ShishirPatil/gorilla
**Year:** 2023
**Why it matters for tool calling / agents:** Tracks function-calling capability, schema adherence, multi-turn behavior, stateful evaluation, and format sensitivity.
**Limitations:** Benchmark scope is broad; chess-specific policy and oracle rules must be built separately.

---

**Title:** VAKRA
**URL:** https://github.com/IBM/vakra
**Year:** 2024
**Why it matters for tool calling / agents:** Supports executable, verifiable tool-use evaluation with local APIs, state, replay, and groundedness checks.
**Limitations:** Not targeted at chess, but architecture strongly transfers.

---

**Title:** tool-eval-bench
**URL:** https://github.com/SeraphimSerapis/tool-eval-bench
**Year:** 2025
**Why it matters for tool calling / agents:** Uses deterministic mock tools, multi-turn orchestration, structured outputs, and safety/boundary categories for local-first eval.
**Limitations:** Scenario set is general-purpose and smaller than needed for chess release gates.

---

**Title:** Local Agent Bench / tool-calling-benchmark
**URL:** https://github.com/MikeVeerman/tool-calling-benchmark
**Year:** 2025
**Why it matters for tool calling / agents:** Demonstrates local-first tool-calling benchmark patterns useful for avoiding cloud API instability.
**Limitations:** Not a chess assistant pipeline.

---

**Title:** python-chess
**URL:** https://github.com/niklasf/python-chess
**Year:** 2026
**Why it matters for tool calling / agents:** Provides legal move generation, SAN/UCI parsing, board state transitions, and UCI engine integration for a local chess oracle.
**Limitations:** Engine determinism and scoring normalization still require product-specific policy.

---

**Title:** vLLM structured outputs
**URL:** https://docs.vllm.ai/en/latest/features/structured_outputs
**Year:** 2026
**Why it matters for tool calling / agents:** Supports JSON-schema-constrained generation through OpenAI-compatible interfaces for strict router outputs.
**Limitations:** Constrained decoding enforces format, not semantic tool choice.

---

**Title:** llama.cpp grammars
**URL:** https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
**Year:** 2026
**Why it matters for tool calling / agents:** Provides GBNF and JSON-schema-to-grammar workflows for local grammar-constrained tool calls.
**Limitations:** Grammar validity alone does not prove policy adherence or chess correctness.

---

**Title:** OpenAI function calling / structured outputs
**URL:** https://developers.openai.com/api/docs/assistants/tools/function-calling
**Year:** 2026
**Why it matters for tool calling / agents:** Documents strict structured outputs, required fields, and rejecting unknown fields through JSON Schema patterns.
**Limitations:** API contract discipline must be paired with backend validation and replay.

---
