# How the training dataset is built (the "data part")

**What this document is.** A plain-English, detailed explanation of how we built the
training data for the model — written so a human can read it once, understand it fully,
and rewrite it into a polished report or talk. It favours ordinary English over technical
terms; where a technical term is genuinely useful, it is introduced once and defined. There
is a short glossary at the end.

**Who/what the data trains.** The model we are training is a *general assistant that operates
a toolbox*. It is shown a list of "skills" (written instructions it can pull up and read) and
"tools" (functions it can call to get data or take an action), and its job is to pick the right
ones and use them to satisfy a request — in *any* topic, not just chess. Chess coaching is our
flagship demonstration, but only about a quarter of the data is chess; the other three-quarters
teach the general skill of choosing and using the right tool for any task.

---

## 1. The single most important idea: we manufactured the data, we did not write it out

There are two ways to produce a large set of training examples.

- **The "writer" way:** sit a person (or another AI) down and have them compose every example,
  one at a time. Slow, expensive, and the examples quietly inherit the writer's habits and
  mistakes.
- **The "factory" way:** write a *program* that assembles examples from a small set of
  hand-made parts. You build the factory once; it stamps out as many examples as you want, and
  it can *check* each one for correctness before keeping it.

**We chose the factory way.** No language model writes our conversations. A human (the assistant
building this project) hand-wrote a few hundred small text fragments — sample questions, sample
tool outputs, sample answers — and a program combines them in many different ways to produce the
full dataset.

The one sentence that captures it: **we hand-wrote a few hundred parts, and a program combines
them into roughly seventy-five thousand complete examples — each one correct by construction and
free to regenerate.**

Everything that follows is the detail behind that sentence.

---

## 2. What one training example looks like

Each example is not a simple "question → answer" pair. It is a short *transcript of the assistant
working* — a back-and-forth that shows the whole loop of getting something done. A typical
example has this shape:

```
USER:       a request, phrased the way a real person might phrase it
ASSISTANT:  loads the relevant skill (pulls up its instructions)
TOOL:       returns the skill's written instructions
ASSISTANT:  calls the one tool those instructions point to
TOOL:       returns a result (data)
ASSISTANT:  a final answer that uses what the tool returned
```

This matters: we are not teaching the model *what to say*, we are teaching it *how to work* —
when to pull up instructions, when to call a tool, and that the final answer comes only *after*
reading the tool's result. (In the field this is called *trajectory imitation* — learning from
a record of the whole process, not just the endpoints.)

The model has exactly two kinds of action it can take, and it takes one per step:

- **Load a skill** — pull up a set of written instructions to read. A skill is *guidance*, not a
  function; loading it just gives the model something to read.
- **Call a tool** — run a function that returns data or changes something in the world.

It also has three "thinking" settings that the data teaches it to obey: *fast* (no visible
thinking, just act), *think* (show a brief line of reasoning at every step), and *auto* (show
reasoning only at genuinely hard moments). The same underlying example can be rendered in any of
the three, which is how the model learns this as a controllable behaviour rather than an
always-on habit.

---

## 3. How one example is built: the "card" and the "combination lock"

This is the core mechanism, explained with a real case.

### The card

For the general (non-chess) part of the data, I hand-wrote **20 topic "cards."** Each card is one
skill the assistant can offer — for example **code-review**. On a card I wrote, by hand, a small
set of matched parts:

- **6 sample questions** a person might ask (e.g. *"review my diff," "spot anything wrong here,"
  "pr review please"*).
- **3 scenarios.** Each scenario is a *matched pair*: a pretend tool output, and the answer that
  reads it. For code-review, one scenario's tool output shows a function that returns `a - b`
  where it should add, and the matching answer explains that this is a sign bug that would corrupt
  a checkout total. The two were written together, on purpose, so they fit.
- **1 short instruction body** (the skill's "how to do this job" text).
- **1 one-line description** (what the skill is for — this is what the model reads when deciding
  whether to pick it).

That is the whole card: about two dozen short fragments, all hand-authored to be consistent.

The reason the conversations in our data sound coherent and "match" is not magic and not an AI —
it is simply that **a human wrote each question, tool output, and answer to go together.**

### The lock

Building one example means choosing one item from each of several "buckets," then dropping the
chosen items into the fixed conversation shape from Section 2. Picture a combination lock with
several reels; one spin lands on one item per reel and yields one example. For the code-review
card the reels are:

| Reel | What it varies | How many |
| --- | --- | --- |
| Question | which of the sample questions | 6 |
| Style | how the question is phrased (see below) | 6 |
| Scenario | which tool-output-and-answer pair | 3 |
| Thinking mode | fast / think / auto | 3 |
| Closer | which follow-up line ends the answer | 10 |

The **Style** reel is worth a closer look, because it is the only place the program changes the
*wording* of a human-written sentence. It wraps the chosen question in one of six everyday
registers:

- formal — *"Please review my diff."*
- casual — *"review my diff"*
- slang — *"yo review my diff"*
- typo — *"review my diff plz"*
- anxious — *"not sure about this — review my diff"*
- beginner — *"i'm new to this; review my diff"*

### Watch two examples come out of the same card

**Spin one** (question 5, slang, scenario "auth bug," fast mode, closer 1):

```
USER:       yo spot anything wrong here
ASSISTANT:  Let me pull up the right skill.  [loads code-review]
TOOL:       (code-review instructions: read the diff first, flag bugs…)
ASSISTANT:  Now the data I need.  [calls the diff viewer on auth.py]
TOOL:       if user.role == 'admin' or True:   # temp
ASSISTANT:  There's an 'or True' left in the admin check, so every user passes
            as admin — a live security hole, not a style nit. Want me to dig
            into the cause, or jump straight to the fix?
```

**Spin two** (question 1, formal, scenario "sign bug," think mode, closer 2) — same card, nothing
rewritten:

```
USER:       Please review my diff.
ASSISTANT:  (goal: review their code)
            (thinking: they want a review — pull up the code-review skill)
            [loads code-review]
TOOL:       (code-review instructions…)
ASSISTANT:  (thinking: the skill says read the diff before judging)
            [calls the diff viewer on app.py]
TOOL:       def add(a, b): return a - b   # used by checkout total
ASSISTANT:  add() returns a - b — a sign bug that would corrupt the checkout
            total; the rest reads clean. Want the short version of the fix, or
            the full reasoning?
```

Two completely different-looking conversations, produced by **turning dials, not writing text.**

---

## 4. The full curriculum: what the data actually teaches

The dataset is divided into **25 "slices."** Each slice is one behaviour we wanted the model to
learn. They group into five families.

**Family 1 — Chess (our flagship demonstration), 11 slices.** Play a move the user names; choose
between options; handle illegal or special moves; judge who is winning; find the best move; review
the move just played; spot the opponent's threats; read the board (list pieces, legal moves, take
back a move); explain a chess idea; handle greetings and small talk; answer chess trivia.

**Family 2 — Core "operate the toolbox" lessons (any topic), the heart of the product.** Pick the
right skill from a list; cope when two skills could fit or none fits; use a brand-new tool whose
inputs are described on the spot; adapt when a tool is switched off or read-only; never state a
fact without checking it; handle special-case rules; use several tools but stay within a sensible
budget instead of looping forever; **recover from a tool error** (read the error, fix the inputs or
pick a different tool, never just repeat the failing call); describe results in honest, non-inflated
language; handle messages that are part task and part chit-chat; **resist manipulation** (if a tool's
output contains text like "ignore your instructions," treat it as data, not as orders); navigate a
plugin marketplace without falsely claiming something was installed; clean up messy slang before
acting on it; and **answer directly when nothing in the toolbox fits** (loading a skill is a choice,
not a reflex).

**Family 3 — The flagship routing slice.** This is the 20-card "pick the right skill across any
topic" task. It is the single largest slice — about a quarter of all the data.

**Family 4 — Multi-turn conversations.** Hold a conversation across several back-and-forth turns,
remembering earlier context.

**Family 5 — Advanced reasoning ("stages"), 3 slices.** *Verify by running code* — check a computed
claim by actually running a short script and reading its output, instead of asserting a number from
memory. *Compound plan* — when a goal needs two or more skills, lay out a plan, do every step, and
do not stop early. *Audited plan* — pull up an "audit" skill, verify each checkable step by running
code, and be honest about which parts are only partly done.

**The 20 routing cards** (Family 3) are: code-reviewer, math-tutor, writer-coach, cooking-helper,
data-analyst, fitness-coach, travel-planner, resume-helper, translator, unit-converter,
color-designer, sql-helper, regex-helper, json-fixer, music-theory, date-calculator, log-analyzer,
api-debugger, grammar-checker, budget-planner. They deliberately span code, food, money, travel,
music, fitness and more, so that "route to the right skill" means routing across genuinely
different topics.

---

## 5. From a few hundred parts to seventy-five thousand examples

This is the part most people find surprising, so it is worth stating carefully and distinguishing
two numbers that are easy to confuse.

**The multiplication.** Independent choices multiply. For one card, the reels were 6 questions ×
6 styles × 3 scenarios × 3 thinking modes × 10 closers. Multiplied out, that is **3,240 distinct
conversations from a single card.** Across 20 cards, the routing slice alone has on the order of
**65,000 possible distinct conversations** — built from roughly 120 hand-written questions and 60
hand-written scenario pairs.

**Capacity versus sample — the crucial distinction.** That 65,000 is the size of the *room* — how
many distinct conversations the lock *could* produce. It is **not** how many we put in the dataset.
We only *draw a sample* from that room. In the finished corpus:

```
Routing's possible room:   ~65,000 distinct conversations   (the capacity)
Routing rows we drew:       ~18,000                          (the sample, ~25% of the corpus)
The whole corpus:           ~75,000  drawn across ALL 25 slices
```

So there is no "leftover" — the corpus is not "routing plus a bit extra." Every slice gets a
planned share of the ~75,000 total; routing is simply the biggest share. The other ~56,000 rows
are the other 24 slices.

**Why deliberately draw fewer rows than the room holds?** Because that is exactly what keeps the
data varied. When you sample 18,000 from a room of 65,000, you almost never draw the same
conversation twice. If we had forced the routing slice up to 65,000 rows, we would be scraping the
bottom of the room and producing near-duplicates. Sampling *below* capacity is what guarantees
diversity.

**How the shares are decided.** We wrote down a target count for each slice (for example, routing
should be far larger than the niche slices), and then scaled all the targets up together by the
same factor until they summed to the desired corpus size. So the *proportions* between slices are
chosen by hand on purpose, and the absolute counts just follow from the chosen total size.

---

## 6. Why the data does not lie: grounding

A constant risk with assistants is that they state confident numbers that are simply made up. We
designed the data so that **the example answers can only state facts that genuinely came from a
tool.** Two mechanisms.

**Real engines, not pretend ones, where it counts.** For chess, the numbers in the examples — the
evaluation of a position, the best move — are produced by running a **real chess engine
(Stockfish)** while the data is being built. They are not numbers I made up. Likewise, the
"verify by running code" slice runs a **real Python sandbox**: the example shows the assistant
running an actual script and reading its actual output. So in these slices the tool outputs are
true, not authored.

**A built-in fact-check on every example.** When the factory finishes an example, an automatic
checker scans the final answer for any concrete value — a number, a chess move — and confirms that
*the same value appears in one of the tool results earlier in the conversation*. If the final
answer cites a value that no tool produced, the example is thrown away. In effect, the data is
filtered so that the model only ever sees answers that **copy facts from tool output, never invent
them.** This trains the habit we want: check, then claim.

(For the hand-written general cards, the "tool output" is something I wrote — but because I also
wrote the matching answer, and the checker enforces consistency, the answer still only ever cites
what its tool output contains.)

---

## 7. Why it generalizes: distractors, shuffling, and invented skills

The hardest thing to teach is *choosing* the right skill — and it is easy to accidentally teach a
shortcut instead. We took specific steps to prevent that.

**The right answer is never shown alone.** In every routing example, the correct skill sits in a
*menu* alongside several others — including, often, completely unrelated ones (chess-coach next to
cooking-helper next to a SQL helper). If we only ever showed the one correct skill, the model would
learn "use whatever skill is in front of me," which is not a skill at all.

**The menu is shuffled every time.** The position of the correct skill in the list is randomised.
If the right answer were always first, the model would learn "pick the first one." Shuffling forces
it to actually *read the descriptions* and match them to the request.

**We invent skills the model has never seen.** A portion of the examples include freshly
*minted* skills and tools with made-up names (think "skill-vega-47") and a templated description.
Because their names are meaningless and never repeat, the model cannot memorise them — it has to
route by *what the description says the skill does*. This is the single biggest reason the model can
handle skills and tools it never encountered in training: we trained it on a moving target.

Put together: hard distractors + random order + unseen names = the model learns the real skill
(read and match descriptions), not a shortcut. This is the same logic behind shuffling labels and
adding "hard negatives" in any classification task, applied to tool-selection.

---

## 8. The quality gate: an automatic inspector, and deliberate bad examples

**Every example is inspected before it is kept.** After the factory assembles an example, an
automatic checker runs about two dozen rules against it — for instance: the final answer must not
leak the internal action markup; the model may only call tools that are actually listed; tool
inputs must match what the tool accepts; it must not call the same tool twice in a row in a loop;
it must read a skill's instructions before using that skill's tool; and the fact-check from Section
6. An example that breaks any rule is **not** added to the good pile; it is set aside. These rules
are, in effect, the project's contract written as an executable test — so data quality is *proven*
per example, not eyeballed.

**We also keep deliberately broken examples on purpose.** Alongside the good pile, the factory
produces a "rejected" pile by intentionally corrupting good examples into known failure modes:
calling a tool that was never offered, claiming a tool ran when it did not, loading an irrelevant
skill, leaking internal markup into the final answer, and so on. These labelled "this is wrong"
examples feed the auditing and quality-control parts of the project, so the system also learns what
*bad* looks like, by name.

---

## 9. Splitting into "study" and "exam" sets without cheating

A model must be tested on examples it did not train on, or the test is meaningless. We split the
corpus into a training set (~97%) and a held-back validation set (~3%), with two safeguards.

**Even coverage.** We hold back a slice-by-slice sample rather than a random scoop, so that *every*
behaviour is represented in the exam set, not just the common ones.

**No leakage.** We make sure no held-back example is an exact duplicate of a training example —
otherwise the "exam" would just be testing memorisation. Where a slice has naturally few possible
distinct answers (some chess templates, fixed-lesson slices), we keep a small floor of validation
examples for coverage but still forbid exact duplicates. The practical upshot: the validation score
is a fair measure of generalisation, not of recall.

---

## 10. Reproducibility and cheap regeneration

The whole factory is driven by a single starting number (a "seed"). Given the same seed, it
produces the **identical** dataset every time — same examples, same order. This means the data is
fully reproducible, and any change (a new rule, a new card, a reworded instruction) is a one-button
regenerate at essentially no cost. There is no expensive human or API step to repeat.

A related design choice: the system instructions that wrap every example are **rebuilt fresh from
the parts each time the data is loaded**, rather than baked into the saved files. So the contract
wording can be edited and the model retrained on the new wording *without regenerating the dataset
at all*.

---

## 11. Honest limitations (worth stating plainly)

A good account names its weak spots; these are the ones to be upfront about.

- **The "messiness" of user questions is shallow.** Real users misspell mid-word, ramble, switch
  languages, and trail off. Our six phrasing styles are light wrappers ("yo …", "… plz") around
  otherwise clean sentences. The model sees *some* informal phrasing, but not the full chaos of
  real chat. The dedicated "clean up messy chat" skill partly offsets this, but it remains a gap.
- **Hand-written tool outputs are tidier than real ones.** For the general cards, the pretend tool
  outputs are clean and uniform. Real-world tool output is noisier. The chess and code-running
  slices use real engines and so avoid this, but the general slices do not.
- **The behaviours are a fixed, hand-chosen list.** The 25 slices are the behaviours we thought to
  teach. Coverage of the long tail of real situations is therefore only as broad as that list.

None of these sink the approach — the generalisation mechanisms in Section 7 are what let a model
trained on tidy, templated data still handle unseen skills — but an honest report should say them.

---

## 12. Glossary (plain definitions of the few necessary terms)

- **Skill** — a set of written instructions the assistant can pull up and read. Guidance, not a
  function.
- **Tool** — a function the assistant can call to get data or change something. It returns a result.
- **Slice** — one specific behaviour we want to teach; the dataset has 25 of them.
- **Card** — one hand-written topic (like "code-review") holding sample questions, scenarios, and a
  description, used to build the routing examples.
- **Scenario** — a matched pair of a pretend tool output and the answer that reads it.
- **Grounding** — making sure the answer's facts come from a tool result rather than being invented.
- **Distractor** — a wrong-but-plausible skill placed in the menu so the model must actually choose.
- **Trajectory** — the full step-by-step record of the assistant working, which is what each example
  is (as opposed to a bare question-and-answer pair).
- **Validation set** — the held-back examples used to test the model on things it did not train on.
- **Seed** — the single starting number that makes the whole data-generation reproducible.
