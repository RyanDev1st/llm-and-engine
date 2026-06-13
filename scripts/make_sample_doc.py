"""Generate a TRULY-RANDOM, human-inspectable sample of the committed v1.2 train
split — N rows per slice, chosen by seeded random.sample over the whole bucket
(not first-N, not handpicked). Shows real token length (exact gemma4_e2b path),
the offered skills/tools, selected skills, and every conversation turn verbatim.

Run:  python scripts/make_sample_doc.py [N]
Out:  docs/2026-06-13-v1.2-random-sample-inspection.md
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path("src/llm").resolve()))
from llm_training.chat_format import remap_tool_messages  # noqa: E402
from llm_training.system_prompt import build_system  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED = 1313
TRAIN = Path("data/sft/v1_2_train.jsonl.gz")
OUT = Path("docs/2026-06-13-v1.2-random-sample-inspection.md")
tok = AutoTokenizer.from_pretrained("src/llm/models/gemma4_e2b", trust_remote_code=True)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ntok(row):
    system = build_system(row.get("skills_index", []), row.get("tool_manifest", []),
                          row.get("plugin_context", {}), reasoning_mode=row.get("reasoning_mode", ""))
    body = [m for m in row.get("messages", []) if m.get("role") != "system"]
    msgs = remap_tool_messages([{"role": "system", "content": system}, *body])
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def clip(s, n=600):
    s = s.replace("\n", "\n  ")
    return s if len(s) <= n else s[:n] + " …[clipped]"


def skill_names(row):
    out = []
    for s in row.get("skills_index", []):
        nm = s.get("name", "?")
        tags = []
        if s.get("plugin"):
            tags.append(s["plugin"])
        if s.get("enabled") is False:
            tags.append("disabled")
        out.append(nm + (f"({','.join(tags)})" if tags else ""))
    return out


def tool_names(row):
    return [t.get("name", "?") for t in row.get("tool_manifest", [])]


def render_row(row, idx, total):
    lines = []
    lines.append(f"#### {row['slice']} — sample {idx}/{total}")
    lines.append("")
    lines.append(f"`id: {row.get('id','')}`")
    lines.append("")
    lines.append(f"- **reasoning_mode:** `{row.get('reasoning_mode') or '(unset)'}`  ·  "
                 f"**tokens (train path):** {ntok(row)}")
    lines.append(f"- **selected skills:** {row.get('selected_skills', [])}")
    lines.append(f"- **skills offered:** {', '.join(skill_names(row)) or '(none)'}")
    lines.append(f"- **tools offered:** {', '.join(tool_names(row)) or '(none)'}")
    lines.append("")
    lines.append("**conversation:**")
    lines.append("")
    body = [m for m in row["messages"] if m.get("role") != "system"]
    for i, m in enumerate(body):
        role = m.get("role", "?")
        is_final = role == "assistant" and i == len(body) - 1
        tag = "assistant (FINAL)" if is_final else role
        lines.append(f"- **{tag}:** {clip(m.get('content',''))}")
    lines.append("")
    return "\n".join(lines)


def main():
    rows = list(read_rows(TRAIN))
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["slice"]].append(r)

    rng = random.Random(SEED)
    parts = []
    parts.append("# v1.2 train split — truly-random sample for inspection\n")
    parts.append(f"Parent: none\n")
    parts.append(
        f"**{len(rows)} train rows · {len(buckets)} slices · {N} random rows per slice "
        f"(seeded `random.sample`, SEED={SEED} — not first-N, not handpicked).** "
        "Token counts are measured through the exact training path "
        "(`build_system` + tool-role remap + Gemma chat template + gemma4_e2b tokenizer); "
        "train ceiling is 1664. Regenerate with `python scripts/make_sample_doc.py`.\n")
    parts.append("> The per-row **system prompt** (the harness contract: two verbs, the offered\n"
                 "> skills index, the tool manifest, the reasoning-mode line) is ~85% of every\n"
                 "> row and is shown ONCE below. Each sample then lists only its offered\n"
                 "> skills/tools + the conversation turns.\n")

    # one full rendered system prompt (representative: a cross-domain routing row)
    rep = next((r for r in rows if r["slice"] == "V1_O_cross_domain_skill_routing"), rows[0])
    sys_text = build_system(rep.get("skills_index", []), rep.get("tool_manifest", []),
                            rep.get("plugin_context", {}), reasoning_mode=rep.get("reasoning_mode", ""))
    parts.append("## The harness contract (one rendered example)\n")
    parts.append(f"From `{rep.get('id','')}` (slice `{rep['slice']}`, mode "
                 f"`{rep.get('reasoning_mode') or '(unset)'}`):\n")
    parts.append("```\n" + sys_text.strip() + "\n```\n")

    parts.append("## Random samples by slice\n")
    for slice_name in sorted(buckets):
        bucket = buckets[slice_name]
        k = min(N, len(bucket))
        picks = rng.sample(bucket, k)
        parts.append(f"### {slice_name}  ({len(bucket)} rows in split)\n")
        for j, row in enumerate(picks, 1):
            parts.append(render_row(row, j, k))

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
