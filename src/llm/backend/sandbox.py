"""Sandboxed `python` tool: runs a SHORT model-authored script in an isolated
subprocess and returns its captured stdout, so the agent GROUNDS a claim by
running code and reading the REAL output — the way Claude verifies instead of
fabricating. This is the Stage-0 keystone: verification-as-tool-use. The model
proposes the script, a deterministic interpreter produces the value, the model
copies it.

Isolation (LOCAL single-user demo posture): a fresh `python -I` subprocess
(ignores env + user site), a hard wall-clock timeout, output + code size caps, and
a temp cwd. This contains hangs, crashes, and runaway output. It is NOT a security
boundary against HOSTILE code — `-I` still exposes the full stdlib (os, socket,
subprocess). Before exposing this tool to untrusted input or a multi-user
deployment, run it under an OS-level sandbox (container / seccomp / nsjail) or a
restricted interpreter. Adequate for the local 4060 demo where the operator IS the
user.

Output contract: `output: <stdout>` (the model copies the printed value into its
reply; narration grounding then checks the copy) or a clean `error:` string the
turn loop narrates without leaking internals. Scripts must be DETERMINISTIC (no
time/random/network) so the result rendered at training equals the result the live
tool returns at serve."""
from __future__ import annotations

import subprocess
import sys
import tempfile

_TIMEOUT_S = 3      # wall-clock kill: a grounding script finishes in milliseconds
_MAX_CODE = 500     # a short verification script never needs more
_MAX_OUT = 600      # cap echoed stdout so a runaway print can't flood the context


def run_python(code: str) -> str:
    """Execute `code` in an isolated subprocess; return `output: ...` / `error: ...`."""
    text = (code or "").strip()
    if not text or len(text) > _MAX_CODE:
        return "error: python_invalid"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", text],
            capture_output=True, text=True, timeout=_TIMEOUT_S,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return "error: python_timeout"
    except Exception:
        return "error: python_failed"
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["failed"])[-1][:200]
        return f"error: python_error: {last}"
    out = proc.stdout.strip()
    if not out:
        return "output: (no output — use print() to show the result)"
    if len(out) > _MAX_OUT:
        out = out[:_MAX_OUT] + "…(truncated)"
    return "output: " + out
