"""Cross-domain skill-routing data. Each Domain is one realistic SKILL.md the
agent can load: a description (what routing keys on), a multi-line body
(when-to-use + ordered steps + a constraint), and one domain tool the body
tells it to call. Real domains anchor realism; `synthetic_domain` mints fresh
named skills so the model must route by DESCRIPTION, not memorized names."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .catalog import synthetic_skill_name


@dataclass(frozen=True)
class Domain:
    skill: str
    description: str
    body: str
    tool: str
    tool_args: dict
    call: str
    tool_result: str
    prompts: tuple[str, ...]
    answer: str
    plugin: str = "user-skills"
    source: str = "user_skill"


def _body(name: str, when: str, steps: tuple[str, ...], constraint: str) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return f"# {name}\nWhen to use: {when}\nSteps:\n{numbered}\nConstraint: {constraint}"


REAL_DOMAINS: list[Domain] = [
    Domain(
        skill="code-reviewer",
        description="Review code diffs and files: flag bugs, clarity, and style, suggest concrete fixes.",
        body=_body("code-reviewer", "the user shares a diff or file or asks for code feedback.",
                   ("Read the actual diff with diff_view before commenting.",
                    "Flag correctness first, then clarity, then style.",
                    "Give concrete fixes, not vague praise."),
                   "never claim a test passed without running it."),
        tool="diff_view", tool_args={"path": "required", "range": "required"},
        call="path=app.py range=1-40",
        tool_result="diff_view app.py 1-40: + def add(a, b): return a - b  # used by checkout total",
        prompts=("review my diff", "can you check this code", "is this function right"),
        answer="diff_view shows add() returns a - b — a sign bug that would corrupt the checkout total; the rest reads clean. Want the one-line fix, or should I check the call sites first?",
    ),
    Domain(
        skill="math-tutor",
        description="Explain math problems step by step at the user's level and check the arithmetic.",
        body=_body("math-tutor", "the user asks to solve or understand a math problem.",
                   ("Restate the problem and compute with math_eval, never by hand.",
                    "Show the one step that matters.",
                    "Confirm the result against the tool output."),
                   "do not assert a number you did not evaluate."),
        tool="math_eval", tool_args={"expr": "required"},
        call="expr=12*8+5",
        tool_result="math_eval: 12*8+5 = 101",
        prompts=("whats 12 times 8 plus 5", "help me with this math", "solve this for me"),
        answer="math_eval gives 101 — multiply 12×8 first, then add 5. Want me to walk through why multiplication comes before addition, or try a tougher one?",
    ),
    Domain(
        skill="writer-coach",
        description="Improve writing tone, clarity, and structure; tighten weak prose.",
        body=_body("writer-coach", "the user shares text and wants it clearer or stronger.",
                   ("Run style_check on the text before judging it.",
                    "Name the biggest weakness, not every nit.",
                    "Offer one concrete rewrite."),
                   "edit for the user's voice; do not rewrite their meaning."),
        tool="style_check", tool_args={"text": "required"},
        call="text=my_intro_paragraph",
        tool_result="style_check: 2 passive-voice sentences, avg 31 words/sentence, reading grade 14",
        prompts=("can you improve my intro", "is my writing any good", "make this clearer"),
        answer="style_check flags two passive sentences and 31-word averages — tightening those will lift clarity fast. Want me to rewrite the opening line, or flag every passive clause first?",
    ),
    Domain(
        skill="cooking-helper",
        description="Plan and adjust recipes, substitutions, and timing for home cooks.",
        body=_body("cooking-helper", "the user asks what to cook or how to adjust a dish.",
                   ("Look up a base recipe with recipe_lookup.",
                    "Adapt to the constraint the user gave.",
                    "Give timing and one substitution."),
                   "respect stated allergies and diet limits."),
        tool="recipe_lookup", tool_args={"ingredient": "required"},
        call="ingredient=chicken",
        tool_result="recipe_lookup chicken: pan-seared breast, 20 min, sear 6 min/side then rest 5",
        prompts=("what can I make with chicken", "dinner idea tonight", "quick chicken recipe"),
        answer="recipe_lookup gives a 20-minute pan-sear: 6 minutes a side, then a 5-minute rest. Want a side to pair, or a sheet-pan version with less cleanup?",
    ),
    Domain(
        skill="data-analyst",
        description="Investigate metrics and tables: find the driver behind a number and quantify it.",
        body=_body("data-analyst", "the user asks why a metric moved or what a dataset shows.",
                   ("Pull the figures with search_kb before interpreting.",
                    "Separate the driver from the noise.",
                    "Quantify the single biggest factor."),
                   "report only what the data supports."),
        tool="search_kb", tool_args={"query": "required"},
        call="query=q3_revenue_dip",
        tool_result="search_kb q3_revenue_dip: revenue -8% QoQ, 6 of 8 points from SMB-tier churn",
        prompts=("why did revenue drop", "explain this metric", "what happened in q3"),
        answer="The dip is mostly SMB churn — six of the eight points — not pricing. Want me to break churn out by month, or compare it against Q2?",
    ),
    Domain(
        skill="fitness-coach",
        description="Build safe workout plans and movement swaps around injuries or goals.",
        body=_body("fitness-coach", "the user asks for a workout or an exercise that fits a limit.",
                   ("Look up movements with search_kb for the stated constraint.",
                    "Pick loads that spare the limited joint.",
                    "Give one progression and one swap."),
                   "never program through stated pain."),
        tool="search_kb", tool_args={"query": "required"},
        call="query=knee_safe_leg_day",
        tool_result="search_kb knee_safe_leg_day: box squats, hip thrusts, leg press; avoid deep lunges",
        prompts=("leg workout that's easy on my knees", "what should I train today", "knee-safe legs"),
        answer="Box squats and hip thrusts load the legs while sparing the knee. Want a full session built around those, or swaps for the deep lunges you usually do?",
    ),
    Domain(
        skill="travel-planner",
        description="Plan trip itineraries: sights, routing, and logistics for a time budget.",
        body=_body("travel-planner", "the user asks how to spend a trip or what to see.",
                   ("Look up highlights with search_kb for the destination.",
                    "Fit them to the day budget and transit.",
                    "Flag the one logistics gotcha."),
                   "do not invent opening hours or prices."),
        tool="search_kb", tool_args={"query": "required"},
        call="query=kyoto_3_days",
        tool_result="search_kb kyoto_3_days: Fushimi Inari, Arashiyama, Gion; rail pass covers all three",
        prompts=("what should I see in kyoto", "plan my 3 days", "kyoto trip ideas"),
        answer="Three days covers Fushimi Inari, the bamboo grove, and Gion comfortably on a rail pass. Want it laid out day-by-day, or food stops worked in?",
    ),
    Domain(
        skill="resume-helper",
        description="Sharpen resumes and summaries: stronger verbs, quantified impact, tighter lines.",
        body=_body("resume-helper", "the user shares resume text and wants it stronger.",
                   ("Run style_check on the section first.",
                    "Replace soft verbs with action verbs.",
                    "Quantify one bullet with a metric."),
                   "keep claims truthful to the user's experience."),
        tool="style_check", tool_args={"text": "required"},
        call="text=resume_summary",
        tool_result="style_check: weak verbs (helped, worked), no metrics, 4 lines of filler",
        prompts=("fix my resume summary", "is my resume good", "make my cv stronger"),
        answer="Your summary leans on soft verbs and no numbers — quantifying impact will make it land. Want me to rewrite it with metrics, or tackle the experience section next?",
    ),
    Domain(
        skill="translator",
        description="Translate text between languages and flag idioms that won't carry over literally.",
        body=_body("translator", "the user wants text rendered in another language.",
                   ("Send the text and target language to translate_text.",
                    "Keep tone; don't translate names or code.",
                    "Flag one phrase that needs a non-literal rendering."),
                   "translate only what was given; do not add content."),
        tool="translate_text", tool_args={"text": "required", "to": "required"},
        call="text=user_paragraph to=es",
        tool_result="translate_text -> es: 'Nos vemos pronto'; idiom 'break a leg' rendered as 'mucha suerte'",
        prompts=("translate this to spanish", "how do I say this in spanish", "spanish version please"),
        answer="Here's the Spanish; note 'break a leg' became 'mucha suerte' since the literal version makes no sense there. Want a more formal register, or another language?",
    ),
    Domain(
        skill="unit-converter",
        description="Convert between units (length, mass, temperature, currency-free) with the exact factor.",
        body=_body("unit-converter", "the user asks to convert a quantity between units.",
                   ("Pass value and units to convert_units; never eyeball it.",
                    "Report the result with the conversion factor used.",
                    "Round only at the end."),
                   "state the factor so the user can check it."),
        tool="convert_units", tool_args={"value": "required", "from": "required", "to": "required"},
        call="value=10 from=mi to=km",
        tool_result="convert_units: 10 mi = 16.0934 km (factor 1.60934)",
        prompts=("convert 10 miles to km", "how many km is 10 miles", "miles to kilometers"),
        answer="10 miles is 16.09 km, using the 1.60934 factor. Want it the other way, or a quick reference table?",
    ),
    Domain(
        skill="color-designer",
        description="Audit and fix UI color contrast and palettes against WCAG ratios.",
        body=_body("color-designer", "the user asks about colors, contrast, or palette accessibility.",
                   ("Check the pair with contrast_ratio before judging.",
                    "Compare against the 4.5:1 AA threshold.",
                    "Suggest the smallest shift that passes."),
                   "judge by the measured ratio, not by eye."),
        tool="contrast_ratio", tool_args={"fg": "required", "bg": "required"},
        call="fg=#777 bg=#fff",
        tool_result="contrast_ratio #777 on #fff = 4.48:1 (fails AA 4.5:1 by a hair)",
        prompts=("is this text readable", "check my color contrast", "does #777 pass on white"),
        answer="#777 on white lands at 4.48:1 — just under AA; darkening to #767676 clears it. Want the fixed hex, or the AAA target too?",
    ),
    Domain(
        skill="sql-helper",
        description="Explain and speed up SQL: read the query plan and point at the costly step.",
        body=_body("sql-helper", "the user shares a query or asks why one is slow.",
                   ("Pull the plan with explain_plan before guessing.",
                    "Name the single most expensive operator.",
                    "Suggest the index or rewrite that removes it."),
                   "base advice on the plan, not the query text alone."),
        tool="explain_plan", tool_args={"sql": "required"},
        call="sql=orders_join",
        tool_result="explain_plan: seq scan on orders (2.1M rows, 84% of cost); join key order_id unindexed",
        prompts=("why is my query slow", "speed up this sql", "explain this query plan"),
        answer="The plan shows a 2.1M-row seq scan on orders — an index on order_id kills 84% of the cost. Want the exact CREATE INDEX, or to see the rewritten join?",
    ),
    Domain(
        skill="regex-helper",
        description="Build and test regular expressions against real sample strings.",
        body=_body("regex-helper", "the user wants a pattern or asks why one doesn't match.",
                   ("Test the candidate against samples with regex_test.",
                    "Fix the one group that misfires.",
                    "Keep it readable over clever."),
                   "verify against the samples; never assert an untested match."),
        tool="regex_test", tool_args={"pattern": "required", "samples": "required"},
        call="pattern=\\d{3}-\\d{4} samples=phones",
        tool_result="regex_test: matches '555-1234', misses '(555) 123-4567' (no area-code group)",
        prompts=("write a regex for phone numbers", "why doesn't my regex match", "test this pattern"),
        answer="It catches '555-1234' but misses area codes — adding an optional group fixes it. Want the full pattern, or a version that also allows dots?",
    ),
    Domain(
        skill="json-fixer",
        description="Validate and repair malformed JSON, pointing at the exact break.",
        body=_body("json-fixer", "the user shares JSON that won't parse.",
                   ("Run validate_json to locate the error position.",
                    "Name the one syntax problem.",
                    "Give the corrected snippet."),
                   "report the parser's position; do not guess the line."),
        tool="validate_json", tool_args={"text": "required"},
        call="text=config_blob",
        tool_result="validate_json: error at line 4 col 18 — trailing comma before '}'",
        prompts=("my json won't parse", "fix this json", "what's wrong with this config"),
        answer="It's a trailing comma at line 4 before the closing brace — drop it and it parses. Want the cleaned file back, or a schema check too?",
    ),
    Domain(
        skill="music-theory",
        description="Explain chords, keys, and progressions for a given key or song.",
        body=_body("music-theory", "the user asks about chords, scales, or a progression.",
                   ("Look up the key's chords with chord_lookup.",
                    "Show the function (tonic/subdominant/dominant) that matters.",
                    "Give one substitution to try."),
                   "name notes from the lookup, not from memory."),
        tool="chord_lookup", tool_args={"key": "required"},
        call="key=C_major",
        tool_result="chord_lookup C major: I=C, IV=F, V=G; vi=Am common for pop turnarounds",
        prompts=("what chords are in C major", "give me a pop progression", "chords for this key"),
        answer="In C major the workhorses are C, F, G, with Am for the turnaround — a I-V-vi-IV gets you most pop songs. Want it transposed to another key, or a sadder progression?",
    ),
    Domain(
        skill="date-calculator",
        description="Compute durations, deadlines, and weekday math between dates.",
        body=_body("date-calculator", "the user asks how long between dates or what day something falls.",
                   ("Compute the span with date_diff; never count by hand.",
                    "Report days and the weekday landing.",
                    "Flag any timezone or DST caveat."),
                   "use the tool's count; do not estimate."),
        tool="date_diff", tool_args={"start": "required", "end": "required"},
        call="start=2026-06-13 end=2026-09-01",
        tool_result="date_diff: 80 days; 2026-09-01 is a Tuesday",
        prompts=("how many days until september 1", "days between these dates", "what day is sept 1"),
        answer="It's 80 days out, landing on a Tuesday. Want business days only, or a countdown to a different date?",
    ),
    Domain(
        skill="log-analyzer",
        description="Scan log dumps for the error pattern driving an incident.",
        body=_body("log-analyzer", "the user shares logs or asks what broke.",
                   ("Scan with scan_logs for the dominant error before theorizing.",
                    "Separate the root error from downstream noise.",
                    "Quantify how many lines it accounts for."),
                   "report only patterns the scan returned."),
        tool="scan_logs", tool_args={"pattern": "required"},
        call="pattern=ERROR",
        tool_result="scan_logs ERROR: 142 hits, 118 are 'connection refused :5432' clustered at 02:14",
        prompts=("what's in these logs", "why did the service crash", "find the error"),
        answer="118 of 142 errors are 'connection refused' to Postgres at 02:14 — the DB went away, the rest is fallout. Want me to trace the 02:14 window, or check the DB health logs?",
    ),
    Domain(
        skill="api-debugger",
        description="Probe an HTTP endpoint and read the status, headers, and latency.",
        body=_body("api-debugger", "the user asks why an API call fails or is slow.",
                   ("Hit the endpoint with http_probe before guessing.",
                    "Read the status and the one header that explains it.",
                    "Name the fix (auth, CORS, timeout)."),
                   "diagnose from the probe response, not assumptions."),
        tool="http_probe", tool_args={"url": "required"},
        call="url=/api/orders",
        tool_result="http_probe /api/orders: 401 Unauthorized; WWW-Authenticate: Bearer; 38ms",
        prompts=("why does my api call fail", "debug this endpoint", "what's wrong with this request"),
        answer="It's a fast 401 — the endpoint wants a Bearer token you're not sending, not a server problem. Want help wiring the auth header, or to check token expiry?",
    ),
    Domain(
        skill="grammar-checker",
        description="Catch grammar, agreement, and punctuation errors in a sentence.",
        body=_body("grammar-checker", "the user shares a sentence and wants it corrected.",
                   ("Run grammar_check before rewriting.",
                    "Fix the error that changes meaning first.",
                    "Keep the user's voice."),
                   "correct only flagged issues; don't restyle."),
        tool="grammar_check", tool_args={"text": "required"},
        call="text=user_sentence",
        tool_result="grammar_check: subject-verb disagreement ('the team are' -> 'the team is'); 1 comma splice",
        prompts=("check my grammar", "is this sentence correct", "fix this sentence"),
        answer="Two things: 'the team are' should be 'is', and there's a comma splice to break into two sentences. Want the corrected version, or an explanation of the comma rule?",
    ),
    Domain(
        skill="budget-planner",
        description="Total and break down spending by category to find overspend.",
        body=_body("budget-planner", "the user asks where their money goes or to total a category.",
                   ("Total the category with sum_spend before advising.",
                    "Compare it to a sensible share of income.",
                    "Name the one category to cut."),
                   "use the summed figure, not a guess."),
        tool="sum_spend", tool_args={"category": "required"},
        call="category=dining",
        tool_result="sum_spend dining: $640/mo, 22% of take-home (typical is 8-12%)",
        prompts=("where is my money going", "am I overspending", "total my dining spend"),
        answer="Dining is $640/mo — about 22% of take-home versus a typical 8-12%, so that's the lever. Want a target number for it, or to break out the next category?",
    ),
]

_SYN_TOPICS: tuple[tuple[str, str, str, str], ...] = (
    ("inventory reconciliation", "inventory_counts", "reconcile my warehouse counts", "12 units short in bin A, traced to a receiving miscount"),
    ("garden planning", "spring_planting", "plan my spring garden beds", "tomatoes and basil pair well; start seeds 6 weeks before last frost"),
    ("tax filing prep", "freelance_deductions", "sort out my freelance deductions", "home-office and mileage are the two biggest missed deductions"),
    ("sql tuning", "slow_join_query", "speed up my slow query", "the join scans 2M rows; an index on order_id cuts it 40x"),
    ("regex building", "email_pattern", "write a regex for emails", "a pragmatic pattern matches 99% without validating every RFC edge case"),
    ("language learning", "spanish_past_tense", "explain spanish past tense", "preterite is for finished actions, imperfect for ongoing background"),
    ("budgeting", "monthly_overspend", "figure out my overspending", "dining out is 22% of spend, double the category average"),
    ("guitar practice", "barre_chords", "help my barre chords", "lower the thumb and roll the index finger to stop the buzz"),
    ("plant care", "yellow_leaves", "why are my leaves yellow", "yellowing with damp soil points to overwatering, not light"),
    ("interview prep", "system_design_round", "prep for a system design interview", "lead with requirements and scale numbers before drawing boxes"),
    ("photography", "low_light_shots", "fix my dark photos", "raise ISO and open the aperture before dropping shutter speed"),
    ("car maintenance", "brake_squeal", "diagnose my brake squeal", "squeal on light braking usually means worn pad wear-indicators"),
    ("study planning", "exam_in_two_weeks", "plan my exam revision", "spaced retrieval beats rereading; schedule three short passes"),
    ("public speaking", "talk_nerves", "calm my presentation nerves", "rehearse the open out loud; the first 30 seconds carry the rest"),
    ("home networking", "wifi_dead_zone", "fix my wifi dead zone", "a mesh node mid-house beats moving the router to a corner"),
    ("meal prep", "high_protein_week", "plan high-protein meals", "batch chicken and lentils hits 140g/day across five lunches"),
    ("debugging", "intermittent_crash", "track down an intermittent crash", "the crash correlates with empty input; a null check guards it"),
    ("negotiation", "salary_offer", "negotiate my offer", "anchor on the band's top third with one market comparable"),
    ("accessibility", "low_contrast_ui", "audit my ui contrast", "two text colors fail 4.5:1; darkening them clears WCAG AA"),
    ("pet training", "puppy_recall", "train puppy recall", "reward the turn toward you, not just arrival, to build the habit"),
)


# Synthetic tool ARCHETYPES so minted skills don't all call search_kb — the model
# must route to a genuinely different tool shape (lookup/compute/scan/check), not
# just a fresh name. (tool, arg, verb) — arg holds the query key.
_SYN_TOOLS: tuple[tuple[str, str, str], ...] = (
    ("search_kb", "query", "Pull the specifics with search_kb"),
    ("lookup_ref", "topic", "Look it up with lookup_ref"),
    ("compute_metric", "expr", "Compute it with compute_metric"),
    ("scan_input", "target", "Scan it with scan_input"),
    ("check_rule", "value", "Validate it with check_rule"),
)


def synthetic_domain(seed: int) -> Domain:
    rng = random.Random(seed)
    label, query, prompt, finding = rng.choice(_SYN_TOPICS)
    tool, arg, verb = rng.choice(_SYN_TOOLS)
    name = synthetic_skill_name(seed)
    body = _body(name, f"the user asks about {label}.",
                 (f"{verb} for {label}.",
                  "Identify the single biggest issue.",
                  "Recommend one concrete next action."),
                 f"rely on tool output; do not invent {label} facts.")
    return Domain(
        skill=name,
        description=f"Help with {label}: gather the specifics, then give one concrete next step.",
        body=body,
        tool=tool, tool_args={arg: "required"},
        call=f"{arg}={query}",
        tool_result=f"{tool} {query}: {finding}",
        prompts=(prompt,),
        answer=f"{finding[0].upper() + finding[1:]}. Want me to dig into the cause, or jump to the fix?",
        plugin="synthetic-pack", source="synthetic_plugin",
    )


def pick_domain(seed: int) -> Domain:
    rng = random.Random(seed * 2654435761 % (2 ** 32))
    if rng.random() < 0.40:
        return REAL_DOMAINS[seed % len(REAL_DOMAINS)]
    return synthetic_domain(seed)
