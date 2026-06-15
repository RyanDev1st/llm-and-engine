"""Emergency off-kernel checkpoint recovery — push/pull the ckpt mirror BY HAND.

Use when the automatic `_hub_push` in train_unsloth left the Hub repo empty (e.g. an
xet upload that failed silently) and you need to either bank an in-flight checkpoint
before a 12h SIGKILL, or stage a resume from a prior session.

Paste into a Kaggle/Colab cell:

    from llm.llm_training.hub_recover import push, pull, status
    status("RyanDev1st/gemma4-chesscoach-ckpt")                 # what's on the Hub now?
    push("runs/<OUTPUT>", "RyanDev1st/gemma4-chesscoach-ckpt")  # bank checkpoint/ + best/
    pull("runs/<OUTPUT>", "RyanDev1st/gemma4-chesscoach-ckpt")  # restore before --resume

Forces the LFS upload path (xet off) — the same fix the trainer now applies — so this
works on stock Kaggle images without hf_xet.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def status(repo: str) -> list[str]:
    """List files currently in the Hub ckpt repo (so you can SEE if a push landed)."""
    _env()
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        print(f"[hub] cannot list {repo}: {exc!r}", flush=True)
        return []
    print(f"[hub] {repo} holds {len(files)} files:", flush=True)
    for f in sorted(files):
        print(f"   {f}", flush=True)
    return files


def push(run_dir: str, repo: str) -> None:
    """Mirror checkpoint/ and best/ from a local run dir to the Hub. Loud on failure."""
    _env()
    import traceback
    from huggingface_hub import create_repo, upload_folder
    root = Path(run_dir)
    create_repo(repo, repo_type="model", private=True, exist_ok=True)
    pushed = 0
    for tag in ("checkpoint", "best"):
        local = root / tag
        if not local.is_dir() or not any(local.iterdir()):
            print(f"[hub] skip {tag}: {local} absent/empty", flush=True)
            continue
        try:
            upload_folder(folder_path=str(local), repo_id=repo, repo_type="model",
                          path_in_repo=tag, commit_message=f"manual {tag} push")
            print(f"[hub] pushed {tag} -> {repo}", flush=True)
            pushed += 1
        except Exception as exc:
            print(f"[hub] {tag} push FAILED: {exc!r}", flush=True)
            traceback.print_exc()
    print(f"[hub] done — {pushed} folder(s) mirrored", flush=True)


def pull(run_dir: str, repo: str) -> None:
    """Download checkpoint/ + best/ from the Hub into a local run dir for --resume."""
    _env()
    from huggingface_hub import snapshot_download
    dst = Path(run_dir)
    dst.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=repo, repo_type="model", local_dir=str(dst),
                          allow_patterns=["checkpoint/*", "best/*"])
        print(f"[hub] pulled checkpoint/ + best/ from {repo} -> {dst}", flush=True)
    except Exception as exc:
        print(f"[hub] pull FAILED: {exc!r}", flush=True)
        raise
