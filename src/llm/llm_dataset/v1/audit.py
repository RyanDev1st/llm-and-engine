from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import RULES, SLICES
from .jsonl_io import read_rows
from .paths import OUT
from .profiles import DatasetProfile, V1_2, profile
from .validate import validate_row

# Must stay proportional to generate.DEFAULT_PLAN (general-first 75/25 mix). Chess
# slices keep a tight tolerance; universal slices vary (V1_O dominant) so they are
# only floor-checked. BASE_UNIVERSAL_TARGET = mean universal base (sum/16) so the
# scale denominator matches the real plan total.
CHESS_TARGETS = {"A": 180, "B": 110, "C": 80, "D": 95, "E": 100, "F": 95, "G": 50, "H": 65, "I": 120, "J": 80, "K": 55}
BASE_UNIVERSAL_TARGET = 257  # mean of the 17 V1_* base counts (sum 4370 / 17)
UNIVERSAL_MINIMUM = 60
RULE_MINIMUMS = {"engine_grounded": 200, "skill_body_strict": 200}
GENERALIZATION_MINIMUMS = {
    "human-chat helper accepted coverage": 200,
    "multi-skill composition accepted coverage": 200,
}
GENERIC_FINAL_MAX_SHARE = 0.02
LOADED_SKILL_DIVERSITY_MIN = 50
GROUNDED_WHY_MIN = 0.90
# Reason vocabulary the grounded composer (renderer/grounded.py + review.py) draws from —
# distinct from the standing/eval words, so a regression to "move + line + standing" (no
# WHY) scores ~0. Guards the grounded-answer fraction (the 80%-no-why gap) on the E
# (best-move) + F (review) finals, where a concrete WHY is always expected.
_WHY_VOCAB = (
    "mate", "checkmate", "finishes", "resilient", "stiffest", "resistance", "practical",
    "alive", "fork", "both", "stroke", "wins", "picks off", "nets", "pin", "nails", "freez",
    "queen", "tempo", "harasses", "check", "tucks", "king safe", "develop", "brings",
    "control", "presses", "terms", "soundest", "healthiest", "accurate", "best move",
    "exactly right", "nothing better", "good move", "holds up", "sound choice", "stronger",
    "keeps more", "inaccuracy", "mistake", "blunder",
)
GENERIC_FINAL_PATTERNS = (
    "i read the index",
    "picked the right skill",
    "called only declared tools",
    "answered without xml",
)


def load_rows(path: Path) -> list[dict]:
    return list(read_rows(path))


def audit(gold_dir: Path = OUT, audit_profile: DatasetProfile = V1_2) -> int:
    accepted = load_rows(gold_dir / "accepted.jsonl")
    rejected = load_rows(gold_dir / "rejected.jsonl")
    failures = []
    by_slice = Counter(row["slice"] for row in accepted)
    reject_by_slice = Counter(row["slice"] for row in rejected)
    rule_counts = Counter(rule for row in accepted for rule in row["acceptance_rules"])
    for row in accepted:
        failures.extend((row["id"], v.rule, v.reason) for v in validate_row(row))
    reject_passed = [row["id"] for row in rejected if not validate_row(row)]
    print(f"accepted={len(accepted)} rejected={len(rejected)}")
    print("accepted_by_slice=" + _fmt_counts(by_slice))
    print("rejected_by_slice=" + _fmt_counts(reject_by_slice))
    print("rule_counts=" + _fmt_counts(rule_counts))
    _print_samples(accepted)
    _print_rejects(rejected)
    print(f"accepted_synthetic_share={_synthetic_share(accepted):.3f}")
    print(f"rejected_synthetic_share={_synthetic_share(rejected):.3f}")
    print(f"reject_reason_diversity={_reject_reason_diversity(rejected)}")
    print(f"generic_final_share={_generic_final_share(accepted):.3f}")
    print(f"loaded_skill_diversity={_loaded_skill_diversity(accepted)}")
    print(f"grounded_why_fraction={_grounded_why_fraction(accepted):.3f}")
    missing = _missing(accepted, rejected, by_slice, reject_by_slice, rule_counts, audit_profile)
    for item in missing:
        print(f"MISSING: {item}")
    for row_id, rule, reason in failures:
        print(f"FAIL: {row_id} {rule} {reason}")
    for row_id in reject_passed:
        print(f"FAIL: rejected row passed validation {row_id}")
    ok = not failures and not reject_passed and not missing
    print(f"failures_count={len(failures) + len(reject_passed) + len(missing)}")
    print(f"freeze_ok={ok}")
    return 0 if ok else 1


def _fmt_counts(counts: Counter) -> str:
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _print_samples(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if row["slice"] in seen:
            continue
        seen.add(row["slice"])
        final = row["messages"][-1]["content"]
        print(f"SAMPLE {row['slice']} {row['id']}: {final}")


def _print_rejects(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["slice"]].append(row["reject_reason"])
    for slice_name in sorted(grouped):
        print(f"REJECTS {slice_name}: " + "; ".join(grouped[slice_name]))


def _missing(
    accepted: list[dict], rejected: list[dict], by_slice: Counter, reject_by_slice: Counter, rule_counts: Counter,
    audit_profile: DatasetProfile = V1_2,
) -> list[str]:
    out = []
    pure = audit_profile.pure_chess
    if len(accepted) < audit_profile.accepted_target:
        out.append(f"accepted < {audit_profile.accepted_target}")
    if len(rejected) < audit_profile.rejected_min:
        out.append(f"rejected < {audit_profile.rejected_min}")
    if len(rejected) > audit_profile.rejected_max:
        out.append(f"rejected > {audit_profile.rejected_max}")
    for slice_name in SLICES:
        threshold = UNIVERSAL_MINIMUM if slice_name.startswith("V1_") else 20
        if by_slice[slice_name] < threshold:
            out.append(f"{slice_name} accepted < {threshold}")
    for rule, minimum in RULE_MINIMUMS.items():
        if rule_counts[rule] < minimum:
            out.append(f"rule coverage < {minimum}: {rule}")
    if _reject_reason_diversity(rejected) < 4:
        out.append("reject reason diversity < 4")
    if _generic_final_share(accepted) > GENERIC_FINAL_MAX_SHARE:
        out.append("generic final share > 2%")
    min_div = 4 if pure else LOADED_SKILL_DIVERSITY_MIN
    if _loaded_skill_diversity(accepted) < min_div:
        out.append(f"loaded skill diversity < {min_div}")
    if audit_profile.min_plugin_sources and _plugin_source_diversity(accepted) < audit_profile.min_plugin_sources:
        out.append(f"plugin source diversity < {audit_profile.min_plugin_sources}")
    if _user_prompt_concentration(accepted) > audit_profile.max_prompt_concentration:
        out.append("normalized user prompt concentration too high")
    if pure:
        # v5 keystone gate: the grounded "why" must not silently regress (the 80%-no-why gap).
        if _grounded_why_fraction(accepted) < GROUNDED_WHY_MIN:
            out.append(f"grounded-why fraction < {GROUNDED_WHY_MIN}")
    else:
        # v1.2 cross-domain corpus: synthetic mix, plan-scaled slice tolerance, every-rule
        # coverage, and the generalization-coverage floors (none apply to a pure-chess corpus).
        if _synthetic_share(accepted) < 0.28:
            out.append("accepted synthetic share < 28%")
        if _synthetic_share(rejected) < 0.28:
            out.append("rejected synthetic share < 28%")
        universal_targets = sum(1 for item in SLICES if item.startswith("V1_")) * BASE_UNIVERSAL_TARGET
        target_scale = audit_profile.accepted_target / (sum(CHESS_TARGETS.values()) + universal_targets)
        for slice_name, target in CHESS_TARGETS.items():
            scaled_target = round(target * target_scale)
            if abs(by_slice[slice_name] - scaled_target) > max(20, round(scaled_target * 0.10)):
                out.append(f"{slice_name} outside target tolerance")
        for rule in RULES:
            if rule_counts[rule] == 0:
                out.append(f"rule has no accepted coverage: {rule}")
        for label, minimum in GENERALIZATION_MINIMUMS.items():
            if _generalization_coverage(accepted, label) < minimum:
                out.append(f"{label} < {minimum}")
    return out


def _synthetic_share(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    pattern = re.compile(r"tool_[a-z]+_\d+|\b(skill|ski|plugin|ext)-[a-z]+-\d+")
    hits = sum(1 for row in rows if pattern.search(json.dumps(row)))
    return hits / len(rows)


def _loaded_skill_diversity(rows: list[dict]) -> int:
    """Distinct skills the agent actually LOADS via the native load_skill{name} call
    (was 2 before cross-domain routing). Measures whether routing generalizes by
    description."""
    names: set[str] = set()
    for row in rows:
        for message in row.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if fn.get("name") == "load_skill":
                    name = (fn.get("arguments") or {}).get("name")
                    if name:
                        names.add(name)
    return len(names)


def _reject_reason_diversity(rows: list[dict]) -> int:
    return len({row.get("reject_reason", "") for row in rows if row.get("reject_reason")})


def _plugin_source_diversity(rows: list[dict]) -> int:
    sources = set()
    for row in rows:
        for skill in row.get("skills_index", []):
            if skill.get("source"):
                sources.add(skill["source"])
        for tool in row.get("tool_manifest", []):
            if tool.get("source"):
                sources.add(tool["source"])
    return len(sources)


def _user_prompt_concentration(rows: list[dict]) -> float:
    prompts = Counter()
    for row in rows:
        users = [m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"]
        if users:
            prompts[_normalize_prompt(users[0])] += 1
    if not prompts:
        return 0.0
    return max(prompts.values()) / sum(prompts.values())


def _generalization_coverage(rows: list[dict], label: str) -> int:
    return sum(1 for row in rows if label in row.get("acceptance_rules", []))


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _grounded_why_fraction(rows: list[dict]) -> float:
    """Fraction of E (best-move) + F (review) finals that state a concrete WHY (a reason
    word from _WHY_VOCAB). ~1.0 for the grounded composer; drops toward 0 if finals
    regress to move + line only — the gap that fed serve-time confabulation."""
    answers = [r for r in rows if r.get("slice") in ("E", "F")]
    if not answers:
        return 1.0
    grounded = sum(1 for r in answers
                   if any(w in r["messages"][-1]["content"].lower() for w in _WHY_VOCAB))
    return grounded / len(answers)


def _generic_final_share(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    generic = 0
    for row in rows:
        final = row["messages"][-1]["content"].lower()
        if all(pattern in final for pattern in GENERIC_FINAL_PATTERNS):
            generic += 1
    return generic / len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="v1.2")
    args = parser.parse_args()
    p = profile(args.profile)
    raise SystemExit(audit(p.gold_dir, p))
