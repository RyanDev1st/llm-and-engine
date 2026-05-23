"""Record cleaning + tone warming for the human SFT slices.

Two structural fixes found by audit:
  * every user turn in the slice/slice batch carries a generation-artifact
    suffix like " (c_0)" / " (i_3)" -> strip it.
  * a handful of slice-C error replies are curt ("Please specify ... standard
    notation") -> regenerate them warm + on-spec from the backend error string.

System prompt is canonicalised to the clean SYSTEM_PROMPT (fixes dash mojibake
in the raw files) so all records share byte-identical message[0].
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_training.system_prompt import SYSTEM_PROMPT  # noqa: E402

_ARTIFACT = re.compile(r"\s*\([a-z]{1,4}_\d+\)\s*$", re.IGNORECASE)

# Warm, on-spec replacements keyed on the backend error the assistant is narrating.
_INVALID_SYNTAX = [
    "Hmm, I didn't quite catch that move - could you write it in standard notation like Nf3 or e4?",
    "I'm not sure I read that move right. Mind rephrasing it in standard notation (e.g. Bb5 or O-O)?",
    "That one slipped past me! Could you give it to me in standard chess notation?",
    "Sorry, that didn't land as a move I recognise - try standard notation like exd5 or Qh5 and we're good.",
]


def strip_artifact(text: str) -> str:
    """Remove a trailing ' (x_N)' generation tag from a user message."""
    return _ARTIFACT.sub("", text).strip()


def _warm_illegal(reason: str, rng: random.Random) -> str:
    reason = reason.strip().rstrip(".")
    openers = ["Ah,", "Whoops,", "Not quite -", "Hmm,"]
    closers = [
        "want to try a different move?",
        "let's pick another one!",
        "give me another move and we'll keep rolling.",
        "what would you like to play instead?",
    ]
    return f"{rng.choice(openers)} that move won't work because {reason} - {rng.choice(closers)}"


def warm_slice_c(messages: list[dict], rng: random.Random) -> list[dict]:
    """Rewrite the final narration of a slice-C record warmly from the tool error."""
    out = [dict(m) for m in messages]
    last_tool = next((m["content"] for m in reversed(out) if m["role"] == "tool"), "")
    if not out or out[-1]["role"] != "assistant" or out[-1]["content"].lstrip().startswith("<tool>"):
        return out
    if last_tool.startswith("error: illegal"):
        reason = last_tool.split("reason=", 1)[1] if "reason=" in last_tool else "it isn't legal here"
        out[-1]["content"] = _warm_illegal(reason, rng)
    elif last_tool.startswith("error: invalid_syntax"):
        out[-1]["content"] = rng.choice(_INVALID_SYNTAX)
    return out


def clean_record(record: dict, rng: random.Random) -> dict:
    """Canonicalise system prompt, strip user artifacts, warm slice-C errors."""
    rec = dict(record)
    msgs = [dict(m) for m in rec["messages"]]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = SYSTEM_PROMPT
    for m in msgs:
        if m["role"] == "user":
            m["content"] = strip_artifact(m["content"])
    if rec.get("slice") == "C":
        msgs = warm_slice_c(msgs, rng)
    rec["messages"] = msgs
    return rec
