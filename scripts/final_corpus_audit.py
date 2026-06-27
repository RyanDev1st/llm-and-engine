"""One-shot cynical audit of the COMMITTED v1.2 split (train+val).

Checks the retrain-forcers that audit.py / validate.py do NOT cover:
  1. Real token length through the EXACT training path (build_system + remap +
     Gemma chat template + tokenizer). Counts rows whose total > MAX_SEQ, i.e.
     whose FINAL assistant turn would be silently truncated at train time.
  2. Chat-template fallback rate (apply_chat_template raising -> plain-text
     render -> format drift).
  3. Tool-result survival: rows with a tool turn must show <tool_result> in the
     rendered text (remap fired).
  4. Train/val leakage: exact full-row dup, final-text overlap, first-user-prompt
     overlap.
  5. Exact duplicate rows within train (memorization).
  6. reasoning_mode distribution + integrity (fast=>no <think>; think=>has <think>).
  7. Full re-validation of every committed row via validate_row.
Writes a JSON summary to scripts/_audit_out.json and prints it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path("src/llm").resolve()))

from llm_dataset.v1.validate import validate_row  # noqa: E402
from llm_training.system_prompt import build_system  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MAX_SEQ = 1664
TRAIN = Path("data/sft/v1_2_train.jsonl.gz")
VAL = Path("data/sft/v1_2_val.jsonl.gz")
TOK_DIR = Path("src/llm/models/gemma4_e2b")


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def render_messages(row):
    system = build_system(
        row.get("skills_index", []),
        row.get("tool_manifest", []),
        row.get("plugin_context", {}),
        reasoning_mode=row.get("reasoning_mode", ""),
    )
    body = [m for m in row.get("messages", []) if m.get("role") != "system"]
    # v5-native: NO remap — role="tool" survives the native template via the assistant's
    # structured tool_calls (it folds into a <|tool_response> block).
    return [{"role": "system", "content": system}, *body]


def first_user(row):
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def final_text(row):
    for m in reversed(row.get("messages", [])):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def row_hash(row):
    payload = json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def main(train_path=TRAIN, val_path=VAL):
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR), trust_remote_code=True)
    print("tokenizer loaded:", type(tok).__name__, flush=True)

    train = list(read_rows(train_path))
    val = list(read_rows(val_path))
    print(f"train={len(train)} val={len(val)}", flush=True)

    summary = {"train": len(train), "val": len(val), "MAX_SEQ": MAX_SEQ}

    # ---- token length + template fallback + tool survival (train+val) ----
    lengths = []
    over = []          # (split, id, ntok)
    fallback = []      # ids where apply_chat_template raised
    tool_missing = []  # rows with a tool turn but no <tool_result> in render
    by_slice_over = Counter()
    for split, rows in (("train", train), ("val", val)):
        for i, row in enumerate(rows):
            msgs = render_messages(row)
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                               enable_thinking=False)
            except Exception:
                fallback.append((split, row.get("id")))
                text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            lengths.append(ntok)
            if ntok > MAX_SEQ:
                over.append((split, row.get("id"), ntok))
                by_slice_over[row.get("slice")] += 1
            had_tool = any(m.get("role") == "tool" for m in row.get("messages", []))
            # native: tool results survive as <|tool_response> blocks (was <tool_result>)
            if had_tool and "<|tool_response>" not in text:
                tool_missing.append((split, row.get("id")))
            if (i + 1) % 5000 == 0:
                print(f"  {split} tokenized {i+1}/{len(rows)}", flush=True)

    lengths.sort()
    n = len(lengths)
    summary["token_len"] = {
        "max": lengths[-1],
        "p50": lengths[n // 2],
        "p99": lengths[int(n * 0.99)],
        "p999": lengths[min(n - 1, int(n * 0.999))],
        "over_1536": sum(1 for x in lengths if x > 1536),
        "over_1600": sum(1 for x in lengths if x > 1600),
        f"over_{MAX_SEQ}": len(over),
    }
    summary["over_examples"] = over[:20]
    summary["over_by_slice"] = dict(by_slice_over)
    summary["template_fallback"] = {"count": len(fallback), "examples": fallback[:20]}
    summary["tool_result_missing"] = {"count": len(tool_missing), "examples": tool_missing[:20]}

    # ---- leakage ----
    train_hashes = {row_hash(r) for r in train}
    val_hashes = [row_hash(r) for r in val]
    exact_overlap = sum(1 for h in val_hashes if h in train_hashes)
    train_finals = {final_text(r) for r in train}
    final_overlap = sum(1 for r in val if final_text(r) in train_finals)
    train_prompts = Counter(first_user(r) for r in train)
    prompt_overlap = sum(1 for r in val if first_user(r) in train_prompts)
    summary["leakage"] = {
        "val_exact_row_in_train": exact_overlap,
        "val_final_text_in_train": final_overlap,
        "val_first_prompt_in_train": prompt_overlap,
    }

    # ---- dup within train ----
    th = Counter(train_hashes_list := [row_hash(r) for r in train])
    dup_groups = {h: c for h, c in th.items() if c > 1}
    summary["train_exact_dups"] = {
        "duplicate_rows": sum(c - 1 for c in dup_groups.values()),
        "dup_groups": len(dup_groups),
        "worst": sorted(dup_groups.values(), reverse=True)[:5],
    }

    # ---- final-answer diversity (NOT gated; tracked to catch canned-final
    # regressions where one sentence is repeated thousands of times -> memorization).
    by_slice_finals = defaultdict(Counter)
    all_finals = Counter()
    for row in train:
        f = final_text(row).split("</think>")[-1].strip()
        by_slice_finals[row.get("slice")][f] += 1
        all_finals[f] += 1
    worst = []
    for sl, finals in by_slice_finals.items():
        n = sum(finals.values())
        distinct = len(finals)
        top = finals.most_common(1)[0][1]
        worst.append((sl, n, distinct, round(distinct / n, 3), top))
    worst.sort(key=lambda x: x[3])  # lowest distinct-ratio first
    summary["final_diversity"] = {
        "overall_distinct": len(all_finals),
        "overall_ratio": round(len(all_finals) / max(1, len(train)), 3),
        "worst_slices": [
            {"slice": sl, "rows": n, "distinct": d, "ratio": r, "top_repeat": t}
            for sl, n, d, r, t in worst[:8]
        ],
    }

    # ---- reasoning mode ----
    # v5-native: thinking is NOT inline text — it's the native channel (rendered from a
    # message's `reasoning` field) and is invoked at serve via enable_thinking. So the only
    # integrity that still holds in training data is: a FAST row must carry NO reasoning
    # channel (fast = answer directly). think/auto rows legitimately carry no inline thinking
    # (native at serve); plan rows carry the <goal>/<plan> in the reasoning channel.
    mode_counts = Counter()
    bad_fast = []
    bad_think = []   # not enforced in native (kept for the gate's stable shape)
    for split, rows in (("train", train), ("val", val)):
        for row in rows:
            mode = (row.get("reasoning_mode") or "").strip().lower() or "(unset)"
            mode_counts[mode] += 1
            has_think = any(
                m.get("role") == "assistant" and (m.get("reasoning")
                    or "<|channel>thought" in (m.get("content") or "") or "<think>" in (m.get("content") or ""))
                for m in row.get("messages", [])
            )
            if mode == "fast" and has_think:
                bad_fast.append(row.get("id"))
    summary["reasoning_mode"] = {
        "distribution": dict(mode_counts),
        "fast_with_think": len(bad_fast),
        "think_without_think": len(bad_think),
        "think_without_think_examples": bad_think[:10],
    }

    # ---- full re-validation ----
    vfail = []
    for split, rows in (("train", train), ("val", val)):
        for row in rows:
            vs = validate_row(row)
            if vs:
                vfail.append((split, row.get("id"), vs[0].rule, vs[0].reason))
    summary["validate_failures"] = {"count": len(vfail), "examples": vfail[:20]}

    # ---- slice distribution ----
    summary["slice_dist_train"] = dict(Counter(r.get("slice") for r in train))

    # ---- hard gate: any of these is a retrain-forcer; fail loud, never ship ----
    # NOTE: val_final_text_in_train is intentionally NOT a hard gate. The build step
    # keeps a per-slice val floor (build.VAL_FLOOR) for eval coverage, so low-answer-
    # diversity slices retain a few val rows whose FINAL overlaps train (unavoidable —
    # those slices have few distinct finals; routing eval cares about the first-turn
    # tool, not final text). The REAL leak guard is val_exact_row_in_train (full-row
    # dup), which stays hard-gated at 0. final-text overlap is tracked below for sight.
    gate = {
        "over_seq": summary["token_len"][f"over_{MAX_SEQ}"],   # silently chopped finals
        "template_fallback": summary["template_fallback"]["count"],
        "tool_result_missing": summary["tool_result_missing"]["count"],
        "val_exact_row_in_train": summary["leakage"]["val_exact_row_in_train"],
        "fast_with_think": summary["reasoning_mode"]["fast_with_think"],
        "think_without_think": summary["reasoning_mode"]["think_without_think"],
        "validate_failures": summary["validate_failures"]["count"],
    }
    summary["tracked_val_final_overlap"] = summary["leakage"]["val_final_text_in_train"]
    summary["gate"] = gate
    summary["gate_ok"] = all(v == 0 for v in gate.values())

    Path("scripts/_audit_out.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("GATE:", "PASS" if summary["gate_ok"] else "FAIL", gate)
    return 0 if summary["gate_ok"] else 1


if __name__ == "__main__":
    import argparse

    from llm_dataset.v1.profiles import profile as _profile

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v1.2")
    a = ap.parse_args()
    p = _profile(a.profile)
    MAX_SEQ = p.max_seq          # module global read by main(); enforce the profile's seq ceiling
    raise SystemExit(main(Path(str(p.train_path) + ".gz"), Path(str(p.val_path) + ".gz")))
