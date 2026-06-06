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


def synthetic_domain(seed: int) -> Domain:
    rng = random.Random(seed)
    label, query, prompt, finding = rng.choice(_SYN_TOPICS)
    name = synthetic_skill_name(seed)
    body = _body(name, f"the user asks about {label}.",
                 (f"Pull the specifics with search_kb for {label}.",
                  "Identify the single biggest issue.",
                  "Recommend one concrete next action."),
                 f"rely on tool output; do not invent {label} facts.")
    return Domain(
        skill=name,
        description=f"Help with {label}: gather the specifics, then give one concrete next step.",
        body=body,
        tool="search_kb", tool_args={"query": "required"},
        call=f"query={query}",
        tool_result=f"search_kb {query}: {finding}",
        prompts=(prompt,),
        answer=f"{finding[0].upper() + finding[1:]}. Want me to dig into the cause, or jump to the fix?",
        plugin="synthetic-pack", source="synthetic_plugin",
    )


def pick_domain(seed: int) -> Domain:
    rng = random.Random(seed * 2654435761 % (2 ** 32))
    if rng.random() < 0.40:
        return REAL_DOMAINS[seed % len(REAL_DOMAINS)]
    return synthetic_domain(seed)
