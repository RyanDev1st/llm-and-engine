"""Live dev runner: auto-restarts the weightless app server whenever a backend
*.py changes. The heavy model lives in a separate persistent process
(model_server.py), so these restarts are ~1s and never reload weights.

Workflow:
  Terminal A (once):  python -m backend.model_server "A:/Download/gemma4_chess_kaggle_adapter (1)"
  Terminal B (live):  python -m backend.dev_serve
Edit inference.py / tool_hints.py / ... and save — the app restarts itself.
Frontend (index.html) needs no restart at all; just refresh the browser.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
POLL_SECONDS = 0.5


def _snapshot() -> dict[Path, float]:
    return {p: p.stat().st_mtime for p in BACKEND.glob("*.py")}


def main() -> None:
    # Point the app at the persistent model service unless the user overrode it.
    os.environ.setdefault("CHESS_MODEL_SERVER", "http://127.0.0.1:7861")
    cmd = [sys.executable, "-m", "backend.server", *sys.argv[1:]]
    proc = subprocess.Popen(cmd)
    last = _snapshot()
    print("dev_serve: watching backend/*.py — save a file to auto-restart the app", flush=True)
    try:
        while True:
            time.sleep(POLL_SECONDS)
            now = _snapshot()
            if now != last:
                changed = [p.name for p in now if last.get(p) != now.get(p)]
                print(f"dev_serve: changed {changed or '(files added/removed)'} -> restarting app", flush=True)
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                proc = subprocess.Popen(cmd)
                last = now
            elif proc.poll() is not None:  # app exited on its own — relaunch
                proc = subprocess.Popen(cmd)
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
