"""Cross-domain skill-routing data (V1_O, the largest slice). Each Domain is one
realistic SKILL.md the agent can load: a description (what routing keys on), a
multi-line body (when-to-use + ordered steps + a constraint), and one domain tool
the body tells it to call.

Realism + anti-memorization: each domain carries MULTIPLE scenes — (call,
tool_result, finding) triples — so the same skill appears with different concrete
questions and grounded results, not one canned answer repeated ~470x. The final
reply is the scene's grounded finding + a seeded guiding-question closer (pulled
from CLOSERS), so distinct finals scale into the hundreds and the model learns the
PATTERN (read the result, summarise, offer a next step) instead of memorising one
line. `synthetic_domain` mints fresh named skills (route by DESCRIPTION, not name)
across a wide topic pool.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .catalog import synthetic_skill_name

# Scene = (tool call args, tool result, grounded finding). The finding states what
# the result shows WITHOUT a trailing question; the renderer appends a closer.
Scene = tuple[str, str, str]


@dataclass(frozen=True)
class Domain:
    skill: str
    description: str
    body: str
    tool: str
    tool_args: dict
    scenes: tuple[Scene, ...]
    prompts: tuple[str, ...]
    plugin: str = "user-skills"
    source: str = "user_skill"

    # Back-compat accessors (first scene) for any caller still reading singulars.
    @property
    def call(self) -> str:
        return self.scenes[0][0]

    @property
    def tool_result(self) -> str:
        return self.scenes[0][1]

    @property
    def answer(self) -> str:
        return self.scenes[0][2]


# Seeded guiding-question closers appended to a finding. Domain-agnostic on purpose
# (they read as a coach offering the next step), so one pool multiplies every
# domain's distinct finals without per-domain rewriting. No facts -> never ungrounds.
CLOSERS: tuple[str, ...] = (
    "Want me to dig into the cause, or jump straight to the fix?",
    "Should I walk through the why, or just hand you the change?",
    "Want the detail behind that, or move to the next step?",
    "Want me to take the next step on this, or check something else first?",
    "Should I keep going down this thread, or step back to the bigger picture?",
    "Want a concrete next action, or more context first?",
    "Want me to verify that, or act on it now?",
    "Should I show the alternative, or run with this one?",
    "Want the short version of the fix, or the full reasoning?",
    "Anything you want me to double-check before we move on?",
)


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
        scenes=(
            ("path=app.py range=1-40",
             "diff_view app.py 1-40: + def add(a, b): return a - b  # used by checkout total",
             "diff_view shows add() returns a - b — a sign bug that would corrupt the checkout total; the rest reads clean."),
            ("path=auth.py range=20-60",
             "diff_view auth.py 20-60: + if user.role == 'admin' or True:  # temp",
             "There's an 'or True' left in the admin check, so every user passes as admin — a live auth hole, not a style nit."),
            ("path=utils.js range=1-30",
             "diff_view utils.js 1-30: + for (let i=0; i<=arr.length; i++)  // off-by-one",
             "The loop uses <= arr.length, so it reads one past the end and yields undefined on the last pass."),
        ),
        prompts=("review my diff", "can you check this code", "is this function right",
                 "pr review please", "spot anything wrong here", "code review on this change"),
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
        scenes=(
            ("expr=12*8+5", "math_eval: 12*8+5 = 101",
             "math_eval gives 101 — multiply 12×8 first, then add 5."),
            ("expr=(3/4)*240", "math_eval: (3/4)*240 = 180",
             "It's 180 — three quarters of 240, so divide by 4 then times 3."),
            ("expr=15%+of+80", "math_eval: 0.15*80 = 12",
             "15% of 80 is 12 — turn the percent into 0.15 and multiply."),
        ),
        prompts=("whats 12 times 8 plus 5", "help me with this math", "solve this for me",
                 "i don't get this problem", "can you check my arithmetic", "math homework help"),
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
        scenes=(
            ("text=my_intro_paragraph",
             "style_check: 2 passive-voice sentences, avg 31 words/sentence, reading grade 14",
             "style_check flags two passive sentences and 31-word averages — tightening those lifts clarity fast."),
            ("text=cover_letter",
             "style_check: 5 sentences open with 'I', tone reads flat, no concrete result",
             "Every sentence opens with 'I' and there's no concrete result, so it reads flat — leading with an outcome fixes both."),
            ("text=blog_draft",
             "style_check: 3 filler phrases ('in order to', 'the fact that'), 18% adverbs",
             "The draft leans on filler like 'in order to' and a heavy adverb count; cutting those tightens it without losing meaning."),
        ),
        prompts=("can you improve my intro", "is my writing any good", "make this clearer",
                 "tighten this paragraph", "does this read well", "edit my draft"),
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
        scenes=(
            ("ingredient=chicken",
             "recipe_lookup chicken: pan-seared breast, 20 min, sear 6 min/side then rest 5",
             "recipe_lookup gives a 20-minute pan-sear: 6 minutes a side, then a 5-minute rest."),
            ("ingredient=pasta",
             "recipe_lookup pasta: aglio e olio, 15 min, salt water heavily, reserve pasta water",
             "Aglio e olio is the fast win here — 15 minutes, and saving a little pasta water makes the sauce cling."),
            ("ingredient=black_beans",
             "recipe_lookup black_beans: 25-min stew, sauté aromatics first, cumin + lime to finish",
             "A 25-minute black-bean stew works: build the aromatics first, then brighten with cumin and lime at the end."),
        ),
        prompts=("what can I make with chicken", "dinner idea tonight", "quick chicken recipe",
                 "what's for dinner", "easy meal with what i have", "help me cook something"),
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
        scenes=(
            ("query=q3_revenue_dip",
             "search_kb q3_revenue_dip: revenue -8% QoQ, 6 of 8 points from SMB-tier churn",
             "The dip is mostly SMB churn — six of the eight points — not pricing."),
            ("query=signup_spike",
             "search_kb signup_spike: signups +34% WoW, 80% from one referral campaign",
             "The signup spike is real but narrow — 80% traces to a single referral campaign, so it may not repeat."),
            ("query=support_ticket_rise",
             "search_kb support_ticket_rise: tickets +21%, half tagged 'login' after the SSO change",
             "Ticket volume is up 21% and half are login issues right after the SSO change — that's the driver, not general growth."),
        ),
        prompts=("why did revenue drop", "explain this metric", "what happened in q3",
                 "what's driving this number", "break down this trend", "is this spike real"),
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
        scenes=(
            ("query=knee_safe_leg_day",
             "search_kb knee_safe_leg_day: box squats, hip thrusts, leg press; avoid deep lunges",
             "Box squats and hip thrusts load the legs while sparing the knee."),
            ("query=shoulder_safe_push",
             "search_kb shoulder_safe_push: neutral-grip press, landmine press; avoid behind-neck",
             "For a cranky shoulder, neutral-grip and landmine presses keep you pressing without the behind-neck strain."),
            ("query=beginner_full_body",
             "search_kb beginner_full_body: goblet squat, push-up, row, hinge — 2x8 to start",
             "A simple full-body start is goblet squat, push-up, row, and hinge at 2 sets of 8."),
        ),
        prompts=("leg workout that's easy on my knees", "what should I train today", "knee-safe legs",
                 "workout around my bad shoulder", "i'm a total beginner where do i start", "build me a session"),
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
        scenes=(
            ("query=kyoto_3_days",
             "search_kb kyoto_3_days: Fushimi Inari, Arashiyama, Gion; rail pass covers all three",
             "Three days covers Fushimi Inari, the bamboo grove, and Gion comfortably on a rail pass."),
            ("query=lisbon_weekend",
             "search_kb lisbon_weekend: Alfama, Belém, tram 28; trams crowd before 10am",
             "A Lisbon weekend fits Alfama, Belém, and tram 28 — just ride the tram before 10am to beat the crush."),
            ("query=nyc_one_day",
             "search_kb nyc_one_day: High Line, Village, Met; subway beats cabs midtown",
             "One day in NYC works as the High Line, the Village, and the Met, with the subway faster than cabs midtown."),
        ),
        prompts=("what should I see in kyoto", "plan my 3 days", "kyoto trip ideas",
                 "weekend in lisbon", "one day in new york", "help me plan this trip"),
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
        scenes=(
            ("text=resume_summary",
             "style_check: weak verbs (helped, worked), no metrics, 4 lines of filler",
             "Your summary leans on soft verbs and no numbers — quantifying impact will make it land."),
            ("text=experience_bullet",
             "style_check: bullet starts 'Responsible for', passive, no result",
             "That bullet opens with 'Responsible for' and never states a result — an action verb plus a number fixes it."),
            ("text=skills_section",
             "style_check: 22 skills listed flat, no grouping, keyword stuffing flagged",
             "The skills block reads as keyword stuffing — grouping the 22 into a few themes makes it scannable and credible."),
        ),
        prompts=("fix my resume summary", "is my resume good", "make my cv stronger",
                 "punch up this bullet", "too many skills listed?", "help my resume stand out"),
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
        scenes=(
            ("text=user_paragraph to=es",
             "translate_text -> es: 'Nos vemos pronto'; idiom 'break a leg' rendered as 'mucha suerte'",
             "Here's the Spanish; note 'break a leg' became 'mucha suerte' since the literal version makes no sense there."),
            ("text=email to=fr",
             "translate_text -> fr: formal register kept; 'touch base' rendered as 'faire le point'",
             "The French keeps your formal tone, and 'touch base' became 'faire le point' — the literal version doesn't exist."),
            ("text=greeting to=ja",
             "translate_text -> ja: 'よろしくお願いします' for the closing; no direct English equivalent",
             "The Japanese closing 'よろしくお願いします' has no clean English match, so I kept the intent rather than a word-for-word."),
        ),
        prompts=("translate this to spanish", "how do I say this in spanish", "spanish version please",
                 "put this in french", "say this in japanese", "translate for me"),
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
        scenes=(
            ("value=10 from=mi to=km", "convert_units: 10 mi = 16.0934 km (factor 1.60934)",
             "10 miles is 16.09 km, using the 1.60934 factor."),
            ("value=200 from=lb to=kg", "convert_units: 200 lb = 90.7184 kg (factor 0.453592)",
             "200 pounds is 90.72 kg at the 0.453592 factor."),
            ("value=350 from=F to=C", "convert_units: 350 F = 176.667 C (subtract 32, times 5/9)",
             "350°F is about 176.67°C — subtract 32, then multiply by 5/9."),
        ),
        prompts=("convert 10 miles to km", "how many km is 10 miles", "miles to kilometers",
                 "200 lbs in kg", "what's 350 fahrenheit in celsius", "convert this for me"),
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
        scenes=(
            ("fg=#777 bg=#fff", "contrast_ratio #777 on #fff = 4.48:1 (fails AA 4.5:1 by a hair)",
             "#777 on white lands at 4.48:1 — just under AA; darkening to #767676 clears it."),
            ("fg=#0a84ff bg=#1c1c1e", "contrast_ratio #0a84ff on #1c1c1e = 5.9:1 (passes AA, fails AAA 7:1)",
             "Your blue on dark is 5.9:1 — fine for AA, but it misses the 7:1 AAA bar for small text."),
            ("fg=#e0e0e0 bg=#f5f5f5", "contrast_ratio #e0e0e0 on #f5f5f5 = 1.3:1 (fails badly)",
             "That light-gray-on-near-white is 1.3:1 — effectively invisible; it needs a much darker foreground."),
        ),
        prompts=("is this text readable", "check my color contrast", "does #777 pass on white",
                 "is my dark mode accessible", "why can't i read this label", "wcag check these colors"),
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
        scenes=(
            ("sql=orders_join",
             "explain_plan: seq scan on orders (2.1M rows, 84% of cost); join key order_id unindexed",
             "The plan shows a 2.1M-row seq scan on orders — an index on order_id kills 84% of the cost."),
            ("sql=report_groupby",
             "explain_plan: sort spills to disk (work_mem exceeded) on a 900k-row group by",
             "The group-by is spilling its sort to disk because work_mem is too small — that disk sort is the bottleneck, not the join."),
            ("sql=user_lookup",
             "explain_plan: index used, but SELECT * pulls a 4KB jsonb column per row",
             "The index is fine; the cost is SELECT * dragging a 4KB jsonb column on every row — selecting only what you need fixes it."),
        ),
        prompts=("why is my query slow", "speed up this sql", "explain this query plan",
                 "this report times out", "is my index being used", "optimize this query"),
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
        scenes=(
            ("pattern=\\d{3}-\\d{4} samples=phones",
             "regex_test: matches '555-1234', misses '(555) 123-4567' (no area-code group)",
             "It catches '555-1234' but misses area codes — adding an optional group fixes it."),
            ("pattern=\\w+@\\w+ samples=emails",
             "regex_test: matches 'a@b', wrongly accepts 'a@b' with no TLD, misses 'x.y@z.co'",
             "The pattern is too loose and too tight at once — it passes 'a@b' but fails dotted names like 'x.y@z.co'."),
            ("pattern=^#[0-9a-f]{6}$ samples=hex_colors",
             "regex_test: matches '#1a2b3c', misses '#ABC' shorthand and uppercase",
             "It works for full lowercase hex but misses 3-digit shorthand and uppercase — a case-insensitive flag and a length alternation cover both."),
        ),
        prompts=("write a regex for phone numbers", "why doesn't my regex match", "test this pattern",
                 "email validation regex", "match hex colors", "fix my regular expression"),
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
        scenes=(
            ("text=config_blob", "validate_json: error at line 4 col 18 — trailing comma before '}'",
             "It's a trailing comma at line 4 before the closing brace — drop it and it parses."),
            ("text=api_payload", "validate_json: error at line 2 col 9 — single quotes used for keys",
             "The keys use single quotes; JSON requires double quotes, so line 2 is the break."),
            ("text=settings", "validate_json: error at line 7 — missing comma between two objects in array",
             "Two array objects on line 7 are missing the comma between them — add it and the rest is valid."),
        ),
        prompts=("my json won't parse", "fix this json", "what's wrong with this config",
                 "why is this invalid json", "json syntax error help", "validate this payload"),
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
        scenes=(
            ("key=C_major", "chord_lookup C major: I=C, IV=F, V=G; vi=Am common for pop turnarounds",
             "In C major the workhorses are C, F, G, with Am for the turnaround — a I-V-vi-IV gets you most pop songs."),
            ("key=A_minor", "chord_lookup A minor: i=Am, iv=Dm, V=E; E (major) borrowed for a stronger cadence",
             "A minor centers on Am, Dm, and a borrowed E major — that major V gives the pull back to Am a sad song wants."),
            ("key=G_major", "chord_lookup G major: I=G, IV=C, V=D; ii=Am for a smoother approach to D",
             "G major runs on G, C, D, and slipping Am in before the D makes the move to the V feel smoother."),
        ),
        prompts=("what chords are in C major", "give me a pop progression", "chords for this key",
                 "sad progression in a minor", "what goes with g major", "help me write a chord loop"),
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
        scenes=(
            ("start=2026-06-13 end=2026-09-01", "date_diff: 80 days; 2026-09-01 is a Tuesday",
             "It's 80 days out, landing on a Tuesday."),
            ("start=2026-06-13 end=2026-12-25", "date_diff: 195 days; 2026-12-25 is a Friday",
             "Christmas is 195 days away and falls on a Friday this year."),
            ("start=2026-01-01 end=2026-06-13", "date_diff: 163 days; spans a DST shift in March",
             "That's 163 days, but heads up — the span crosses the March DST change, so an hour goes missing."),
        ),
        prompts=("how many days until september 1", "days between these dates", "what day is sept 1",
                 "how long till christmas", "days since new year", "what weekday is this date"),
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
        scenes=(
            ("pattern=ERROR", "scan_logs ERROR: 142 hits, 118 are 'connection refused :5432' clustered at 02:14",
             "118 of 142 errors are 'connection refused' to Postgres at 02:14 — the DB went away, the rest is fallout."),
            ("pattern=timeout", "scan_logs timeout: 60 hits, all on /checkout, all after a deploy at 14:30",
             "Every one of the 60 timeouts is on /checkout right after the 14:30 deploy — that release is the trigger, not load."),
            ("pattern=OOM", "scan_logs OOM: 1 kill on the worker at 03:02, preceded by a 4GB heap climb",
             "There's a single OOM kill at 03:02 after a steady 4GB heap climb — a leak, not a sudden spike."),
        ),
        prompts=("what's in these logs", "why did the service crash", "find the error",
                 "what broke at 2am", "why are requests timing out", "analyze this log dump"),
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
        scenes=(
            ("url=/api/orders", "http_probe /api/orders: 401 Unauthorized; WWW-Authenticate: Bearer; 38ms",
             "It's a fast 401 — the endpoint wants a Bearer token you're not sending, not a server problem."),
            ("url=/api/profile", "http_probe /api/profile: 200 but 4.2s; Server-Timing db;dur=4100",
             "The call succeeds but takes 4.2s, and Server-Timing pins 4.1s of it on the DB — the slowness is the query, not the network."),
            ("url=/api/upload", "http_probe /api/upload: 403; missing Access-Control-Allow-Origin",
             "The 403 with no Access-Control-Allow-Origin is a CORS block — the server isn't allowing your origin, so the browser kills it."),
        ),
        prompts=("why does my api call fail", "debug this endpoint", "what's wrong with this request",
                 "my api is slow", "getting cors errors", "why a 401 here"),
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
        scenes=(
            ("text=user_sentence",
             "grammar_check: subject-verb disagreement ('the team are' -> 'the team is'); 1 comma splice",
             "Two things: 'the team are' should be 'is', and there's a comma splice to break into two sentences."),
            ("text=their_there",
             "grammar_check: 'their' used for 'there'; one missing Oxford comma flagged optional",
             "The main fix is 'their' where you want 'there'; the comma it flags is optional, so that's your call."),
            ("text=tense_shift",
             "grammar_check: tense shifts mid-sentence (past -> present); 1 dangling modifier",
             "The sentence jumps from past to present tense, and there's a dangling modifier at the front — fixing the tense matters most."),
        ),
        prompts=("check my grammar", "is this sentence correct", "fix this sentence",
                 "their or there?", "did i use the right tense", "proofread this line"),
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
        scenes=(
            ("category=dining", "sum_spend dining: $640/mo, 22% of take-home (typical is 8-12%)",
             "Dining is $640/mo — about 22% of take-home versus a typical 8-12%, so that's the lever."),
            ("category=subscriptions", "sum_spend subscriptions: $180/mo across 14 services, 5 unused 60+ days",
             "Subscriptions run $180/mo across 14 services, and five haven't been touched in two months — easy money back."),
            ("category=transport", "sum_spend transport: $310/mo, 70% rideshare on weekday commutes",
             "Transport is $310/mo and 70% of it is weekday rideshare — a transit pass would cut most of that."),
        ),
        prompts=("where is my money going", "am I overspending", "total my dining spend",
                 "what should i cut", "review my subscriptions", "break down my budget"),
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
    ("note taking", "messy_meeting_notes", "organize my meeting notes", "splitting into decisions, actions, and open questions makes them usable"),
    ("sleep hygiene", "cant_fall_asleep", "fix my sleep schedule", "a fixed wake time anchors the rhythm faster than an earlier bedtime"),
    ("git workflow", "messy_branch_history", "clean up my git history", "an interactive rebase squashes the noise before the PR, not after merge"),
    ("email triage", "overflowing_inbox", "get my inbox under control", "a two-minute rule plus three folders clears most of the backlog"),
    ("houseplant light", "low_light_room", "what plant for a dark room", "pothos and snake plants tolerate low light better than most"),
    ("running form", "shin_splints", "stop my shin splints", "shortening stride and lifting cadence offloads the shins more than new shoes"),
    ("spreadsheet formulas", "vlookup_failing", "why is my vlookup wrong", "exact-match needs the FALSE flag; approximate match is the usual culprit"),
    ("coffee brewing", "bitter_pourover", "my coffee tastes bitter", "bitterness points to over-extraction; coarsen the grind or shorten the pour"),
    ("resume gaps", "employment_gap", "explain my resume gap", "a brief honest line about the gap beats hiding it with vague dates"),
    ("time management", "always_behind", "stop running out of time", "time-boxing the top three tasks beats an open to-do list"),
    ("home repair", "squeaky_door", "fix my squeaky hinge", "a little petroleum jelly on the pin outlasts spray oil for a hinge"),
    ("investing basics", "index_vs_picking", "should i pick stocks", "low-cost index funds beat most active picking over a long horizon"),
    ("dog nutrition", "portion_size", "how much to feed my dog", "portion by goal weight and activity, not by the bag's generic chart"),
    ("presentation design", "slide_overload", "fix my busy slides", "one idea per slide and fewer words read far better from the back row"),
    ("language tone", "too_formal_email", "soften my email tone", "swapping passive openers for a direct ask warms the tone without losing clarity"),
    ("bike maintenance", "shifting_skips", "my gears skip", "a skipping shift is usually cable stretch; a barrel-adjuster turn re-tensions it"),
    ("baking", "flat_cookies", "why are my cookies flat", "flat cookies usually mean warm butter or too little flour, not the oven"),
    ("focus", "constant_distraction", "i can't focus", "a single-tab work block with the phone in another room beats willpower"),
    ("data cleaning", "duplicate_rows", "dedupe my dataset", "keying on a stable id beats fuzzy text matching for catching duplicates"),
    ("onboarding", "new_hire_ramp", "speed up new-hire ramp", "a first-week shipping task teaches the codebase faster than docs alone"),
)


# Synthetic tool ARCHETYPES so minted skills don't all call search_kb — the model
# routes to a different tool NAME per skill. All are retrieval/analysis-shaped so
# the (prose) finding stays coherent with the tool. (tool, arg, verb) — arg = query key.
_SYN_TOOLS: tuple[tuple[str, str, str], ...] = (
    ("search_kb", "query", "Pull the specifics with search_kb"),
    ("lookup_ref", "topic", "Look it up with lookup_ref"),
    ("fetch_context", "subject", "Gather context with fetch_context"),
    ("analyze_topic", "topic", "Run analyze_topic on it"),
    ("get_reference", "key", "Pull a reference with get_reference"),
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
    statement = finding[0].upper() + finding[1:] + "."
    return Domain(
        skill=name,
        description=f"Help with {label}: gather the specifics, then give one concrete next step.",
        body=body,
        tool=tool, tool_args={arg: "required"},
        scenes=((f"{arg}={query}", f"{tool} {query}: {finding}", statement),),
        prompts=(prompt,),
        plugin="synthetic-pack", source="synthetic_plugin",
    )


def pick_domain(seed: int) -> Domain:
    rng = random.Random(seed * 2654435761 % (2 ** 32))
    if rng.random() < 0.40:
        return REAL_DOMAINS[seed % len(REAL_DOMAINS)]
    return synthetic_domain(seed)
