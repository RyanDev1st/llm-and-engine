"""Force a clean process exit (code 0) after a model run completes.

torch / CUDA / bitsandbytes / llama.cpp can call std::terminate from a C++ destructor during
interpreter shutdown — "terminate called without an active exception" -> SIGABRT (6) — AFTER the
work has finished and every output file is written. Under papermill + subprocess(check=True) that
benign exit-time abort fails the WHOLE notebook and skips every downstream cell (a captured
transcript or eval is on disk, yet the run is marked failed and the report cells never execute).

Once main() has returned and stdout/stderr are flushed, os._exit(0) skips those destructors and
guarantees a 0 exit. Call ONLY from the `if __name__ == "__main__"` path AFTER main() returns — so a
REAL error inside main() still raises and propagates (we never mask a genuine failure), and importing
the module for tests has no os._exit side effect.
"""
from __future__ import annotations

import os
import sys


def flush_and_exit(code: int = 0) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)
