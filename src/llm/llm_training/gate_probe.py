"""Pre-flight name-precision gate — runs in its OWN process so the GPU is fully released
before the full train (an in-kernel probe model used to leak ~9 GiB and OOM Cell 7).

Loads the freshly micro-overfit adapter, free-gen probes a few TRAINING rows, and checks the
model emits VALID format with a REAL catalog name. PASS = format-ok AND name-in-catalog on
>= PASS_MIN of the samples — NOT gold-exact: a direct `<tool>move ...>` instead of loading
chess-coach first is valid routing under the 'act this turn' contract, not a failure.

Usage (from a Kaggle cell, after the micro-overfit subprocess):
    python -m llm_training.gate_probe runs/micro_overfit data/sft/v1_2_train.jsonl
Exits 0 on PASS, 2 on FAIL.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys

CONTRACT = {"think", "/think", "goal", "/goal", "plan", "/plan",
            "skill", "/skill", "tool", "/tool"}
PASS_MIN = 7   # of 8 samples


def _fmt_ok(out: str) -> bool:
    foreign = [t for t in re.findall(r"</?([a-zA-Z_][\w]*)", out) if t not in CONTRACT]
    return not foreign and ("<skill>" in out or "<tool>" in out)


def _emitted_name(out: str) -> str | None:
    m = re.search(r"<skill>\s*([\w./-]+)", out) or re.search(r"<tool>\s*([\w./-]+)", out)
    return m.group(1) if m else None


def main() -> None:
    adapter = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else "data/sft/v1_2_train.jsonl"
    os.environ.setdefault("CHESS_REP_PENALTY", "1.0")      # clean decode (rep-penalty corrupts copying)
    os.environ.setdefault("CHESS_NO_REPEAT_NGRAM", "0")
    from backend.model_hf import HFModel
    from llm_training.system_prompt import build_system

    model = HFModel(adapter=adapter, temperature=0.0)

    rows = []
    path = data if os.path.exists(data) else data + ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 256:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def pick(mode: str, n: int) -> list:
        return [x for x in rows if x.get("reasoning_mode") == mode][:n]
    samples = pick("fast", 3) + pick("think", 3) + pick("auto", 2)

    ok_fmt = ok_name = 0
    for x in samples:
        catalog = ({s.get("name") for s in (x.get("skills_index") or [])}
                   | {t.get("name") for t in (x.get("tool_manifest") or [])})
        sysp = build_system(x["skills_index"], x["tool_manifest"], x.get("plugin_context", {}),
                            reasoning_mode=x.get("reasoning_mode", ""))
        user = next(m["content"] for m in x["messages"] if m["role"] == "user")
        out = model.generate([{"role": "system", "content": sysp},
                              {"role": "user", "content": user}],
                             max_new_tokens=160, stop=["</tool>", "</skill>"])
        f_ok = _fmt_ok(out)
        name = _emitted_name(out)
        n_ok = name in catalog
        ok_fmt += int(f_ok)
        ok_name += int(n_ok)
        print("=" * 60)
        print(f"[{x.get('reasoning_mode')}] fmt={'OK' if f_ok else 'BAD'} name={name} in_catalog={n_ok}")
        print("USER :", user[:70])
        print("MODEL:", out[:170])

    n = len(samples)
    print(f"\nGATE: format_ok {ok_fmt}/{n} | real-name (in catalog) {ok_name}/{n}")
    ok = ok_fmt >= PASS_MIN and ok_name >= PASS_MIN
    print("=> PASS — GPU released (subprocess); run Cell 7" if ok
          else "=> FAIL — tell Opus; do not run the full session")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
