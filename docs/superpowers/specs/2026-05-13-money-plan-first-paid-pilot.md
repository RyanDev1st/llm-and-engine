# Money plan: first paid pilot for chess-engine-backed LLM tool-calling project

**Date:** 2026-05-13  
**Goal:** Earn first revenue through a low-risk paid pilot before building full product.  
**Metric:** Lower risk score: fewer build dependencies, shorter path to payment, smaller delivery promise.  
**Guard:** MVP may be planned, but no product implementation in this sprint.

## Executive strategy

Do not start with a full consumer chess SaaS. Start with a paid pilot that sells the strongest current asset: a correctness-first, engine-backed LLM tool-calling evaluation/runtime design for chess coaching products.

Best first customer is not a random chess player. Best first customer is an organization already spending money on chess education, chess apps, tutoring, or AI product experiments and afraid of hallucinated chess advice.

Sell a narrow pilot:

> “We will build and evaluate a small FEN-blind chess coach demo that uses a local chess engine as oracle, proves tool-call correctness on a held-out scenario pack, and reports where your current or planned AI chess coach would fail.”

Target outcome: **one paid pilot at $2,500–$10,000** within 30–45 days.

## Offer ladder

### Offer 1 — Paid diagnostic audit

**Price:** $750–$2,000  
**Timeline:** 3–5 days  
**Buyer:** small chess app, tutoring company, chess content business, AI education startup  
**Deliverable:** report + call

Deliverables:

- current AI/chess workflow risk audit
- tool-calling failure taxonomy
- 20-scenario mini evaluation design
- recommendation: no-build, light MVP, or full pilot

Use this if buyer is skeptical or budget is small.

### Offer 2 — First paid pilot

**Price:** $2,500–$10,000  
**Timeline:** 2–4 weeks  
**Buyer:** product owner or founder with AI chess idea  
**Deliverable:** working demo + evaluation report

Deliverables:

- narrow FEN-blind chess coach MVP for 3–5 user flows
- python-chess/Stockfish-backed truth layer
- router/narrator split or strict tool-call harness
- 50–100 scenario replay pack
- correctness report with blocker/quality buckets
- go/no-go recommendation for productization

Positioning:

> “Not a generic chatbot. A chess-correct assistant prototype with measurable hallucination and routing risk.”

### Offer 3 — Licensing / buildout

**Price:** $15,000–$50,000+  
**Timeline:** after pilot only  
**Buyer:** app/platform with users

Deliverables:

- production runtime design
- larger scenario/evaluator suite
- integration with existing app
- model/router training path
- release gate and regression harness

Do not sell this first unless buyer already has budget and urgency.

## First niche choice

Prioritize buyers in this order:

1. Chess tutoring companies teaching kids or schools.
2. Chess content platforms adding AI feedback.
3. Existing chess apps wanting a safer coach mode.
4. AI edtech startups needing a domain-specific demo.
5. Individual titled coaches with paid communities.

Avoid first:

- mass consumer app from scratch
- generic “AI chess coach for everyone”
- open-ended research contracts without paid scope
- selling to people who only want free advice

## Customer pain map

| Buyer | Pain | Paid pilot angle |
|-------|------|------------------|
| Chess tutoring company | Coaches cannot scale personalized feedback | “AI assistant gives safe first-pass feedback without inventing legal moves.” |
| Chess app | Chatbot hallucination hurts trust | “Evaluation harness proves current-board claims are tool-grounded.” |
| Edtech founder | Needs impressive demo for investors/customers | “Engine-backed coach demo with measurable correctness.” |
| Content creator/coach | Wants premium community feature | “Coach mode for member games with safe explanations.” |
| Existing AI product team | Needs domain eval benchmark | “Chess as deterministic tool-calling benchmark.” |

## Minimum paid pilot scope

Pick exactly one pilot wedge:

### Wedge A — “Explain my last move”

User gives moves or plays through board. Assistant reviews last move using engine evidence.

Tools needed:

- `move`
- `review_move`
- `eval`
- `undo`

Why low risk:

- narrow value
- easy demo
- strong correctness boundary
- useful for coaches and students

### Wedge B — “What are my legal options?”

Assistant explains legal moves and simple candidate ideas.

Tools needed:

- `legal_moves`
- `best_move`
- `eval`

Why low risk:

- less natural-language move ambiguity
- easy validation
- clear value for beginners

### Wedge C — “Spot tactic/threat”

Assistant identifies immediate threats and tactical ideas.

Tools needed:

- `threats`
- `best_move`
- `eval`

Why higher risk:

- threat semantics hard
- more chance of ungrounded claim
- use after first pilot only

Recommended first wedge: **A — Explain my last move**.

## 45-day execution plan

### Days 1–3 — Package offer

Create:

- one-page pilot offer
- short demo storyboard
- example failure report from current research
- pricing menu with three options
- outreach list of 50 prospects

Decision gate:

- If offer cannot be explained in 2 sentences, narrow scope.

### Days 4–10 — Customer discovery and pre-sales

Daily actions:

- contact 10 prospects per day
- ask for 15-minute call
- do not pitch full product first
- ask what chess feedback they already provide and where AI would be risky

Discovery questions:

1. “Where do students most need feedback between lessons?”
2. “Have you tried AI chess feedback? What failed?”
3. “Would wrong legal/eval claims be a deal-breaker?”
4. “If a 2-week pilot proved safe feedback for one workflow, who would approve it?”
5. “What would make this worth $2,500–$10,000?”

Success metric:

- 10 calls booked
- 3 serious pain confirmations
- 1 verbal pilot interest

### Days 11–17 — Sell diagnostic or pilot

Use two close paths:

Path A, buyer hesitant:

- sell $750–$2,000 diagnostic audit
- convert audit into pilot proposal

Path B, buyer urgent:

- sell $2,500–$10,000 pilot directly
- require 50% upfront

Pilot contract must include:

- one narrow workflow
- fixed delivery date
- limited integrations
- no production SLA
- buyer supplies sample positions/games if available
- explicit acceptance criteria

### Days 18–31 — Build MVP pilot

Build only after payment or signed LOI.

MVP scope:

- board/session state with python-chess
- local Stockfish adapter
- router/narrator split or strict structured router
- `move`, `eval`, `review_move`, `legal_moves`, `undo`
- 50–100 scenario pack
- replay/evaluator report
- small demo UI or CLI depending buyer need

Do not build:

- full accounts/auth
- mobile app
- large training dataset
- custom model fine-tuning
- broad `ask_chessbot` fallback
- open-ended educational engine

Acceptance criteria:

- zero illegal state mutation in pilot scenarios
- no ungrounded final current-board claims in held-out pack
- all tool calls schema-valid
- replay pass on all accepted scenarios
- clear list of limitations

### Days 32–38 — Deliver and upsell

Deliver:

- demo walkthrough
- correctness report
- failure taxonomy
- roadmap with 3 options

Upsell options:

1. Expand scenario coverage: $5,000–$15,000.
2. Integrate into existing product: $15,000–$50,000.
3. Monthly eval/runtime retainer: $1,500–$5,000/month.

Close question:

> “Do you want this to become a product feature, an internal benchmark, or a research demo?”

### Days 39–45 — Convert or recycle proof

If buyer converts:

- scope next paid phase
- keep ownership/IP terms clear
- avoid unpaid expansion

If buyer does not convert:

- ask for testimonial or anonymized case study
- turn report into public credibility artifact
- reuse scenario pack and demo for next prospects

## Outreach scripts

### Cold email

Subject: Safer AI chess feedback pilot

Hi {{name}},

I’m building a chess-engine-backed AI coach that avoids a common failure in chess chatbots: confident but wrong board claims.

Instead of letting the model “play chess,” the model routes to python-chess/Stockfish tools, then explains only what the tools prove. I’m looking for 1–2 pilot partners in chess tutoring/apps/content who want safer feedback for one narrow workflow, like “explain my last move.”

Would a 15-minute call be useful to see if this could reduce coach workload or add a premium feature?

### Follow-up

Worth clarifying: I’m not pitching a generic chess chatbot. The pilot is a correctness-first demo plus evaluation report, designed to show where AI chess feedback is safe vs unsafe.

Open to a quick call this week?

### Discovery close

If we can prove one workflow with replayable scenarios and zero ungrounded board claims, would you consider a paid 2–4 week pilot?

## Pricing

Start simple:

| Package | Price | Use when |
|---------|-------|----------|
| Diagnostic audit | $750–$2,000 | buyer unsure, wants advice |
| Narrow paid pilot | $2,500–$10,000 | buyer has clear workflow |
| Productization phase | $15,000–$50,000+ | pilot worked and buyer has users |
| Retainer | $1,500–$5,000/month | ongoing eval/runtime improvement |

First paid pilot should be priced high enough to force seriousness but low enough for fast approval. Recommended first quote: **$5,000 for 3 weeks, 50% upfront**.

## Risk controls

| Risk | Control |
|------|---------|
| Build too much before payment | Sell diagnostic/pilot before MVP build. |
| Buyer wants full app | Restrict pilot to one workflow. |
| Chess correctness failure | Keep model as router/narrator only; engine owns truth. |
| Ambiguous move language explodes scope | Start with “review last explicit move” workflow. |
| No one pays | Run 50-prospect outreach before coding full product. |
| Consumer acquisition too hard | Start B2B/coach/org pilots. |
| Pilot becomes unpaid consulting | Fixed deliverables, upfront payment, change-order rule. |
| Research too abstract | Lead with demo storyboard and failure examples. |

## Risk score

Lower is better.

| Plan path | Build dependency | Sales dependency | Delivery risk | Total risk |
|-----------|------------------|------------------|---------------|------------|
| Full consumer SaaS first | 5 | 5 | 5 | 15 |
| Open-source benchmark first | 2 | 5 | 3 | 10 |
| Consulting only | 1 | 3 | 2 | 6 |
| Diagnostic audit first | 1 | 2 | 2 | 5 |
| Narrow paid pilot first | 3 | 2 | 3 | 8 |

Recommended sequence:

1. Sell diagnostic if trust is low.
2. Sell narrow pilot if buyer has pain and budget.
3. Build MVP only after payment/LOI.
4. Convert pilot into productization or retainer.

## What to do tomorrow

1. Write one-page offer.
2. Create 20-prospect list.
3. Draft demo storyboard for “Explain my last move.”
4. Send 10 outreach messages.
5. Ask for calls, not feedback.
6. Track every reply in a simple sheet.
7. Do not build full MVP until at least 3 discovery calls confirm paid pain.

## End state

By end of first cycle, one of three things should be true:

1. **Best case:** signed $5,000 pilot with 50% upfront.
2. **Acceptable:** paid diagnostic sold and path to pilot identified.
3. **Learning case:** 30–50 prospects reject; revise niche or offer before building.

Do not treat lack of product as blocker. Treat lack of buyer pain as blocker.
