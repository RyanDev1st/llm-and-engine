from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WORKDIR = Path(os.environ.get("KAGGLE_WORKDIR", "/kaggle/working/llm-and-engine"))
MODEL = "gemma4_e4b"
ENGINE = "unsloth"
DATA_STEM = "v5"
MAX_SEQ = 1920
RANK = 16
TARGETS = "all-linear"
LR = "0.0001"
OUTPUT = "gemma4_chess_e4b_kaggle"
CKPT_REPO = "RyanDev1st/gemma4-chesscoach-ckpt-v5"


@dataclass(frozen=True)
class RunSpec:
    output: str
    max_steps: int
    max_examples: int | None
    max_val: int
    eval_every: int
    save_every: int
    grad_accum: int
    ckpt_repo: str


SMOKE = RunSpec("v5_smoke_20", 20, 64, 0, 100000, 20, 1, "")
FINAL = RunSpec(OUTPUT, 1000, None, 128, 100, 50, 16, CKPT_REPO)


def run_env(workdir: Path = WORKDIR, ckpt_repo: str = CKPT_REPO) -> dict[str, str]:
    env = {**os.environ}
    env["PYTHONPATH"] = f"{workdir.as_posix()}/src/llm"
    env["CHESS_CKPT_REPO"] = ckpt_repo
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["UNSLOTH_COMPILE_DISABLE"] = "1"
    return env


def train_command(spec: RunSpec, resume: bool = False) -> list[str]:
    args = [
        "--model", MODEL, "--engine", ENGINE, "--data-stem", DATA_STEM,
        "--max-seq", str(MAX_SEQ), "--rank", str(RANK), "--targets", TARGETS,
        "--grad-accum", str(spec.grad_accum), "--max-steps", str(spec.max_steps),
        "--max-val", str(spec.max_val), "--eval-every", str(spec.eval_every),
        "--save-every", str(spec.save_every), "--lr", LR, "--output", spec.output,
    ]
    if spec.max_examples is not None:
        args += ["--max-examples", str(spec.max_examples)]
    if resume:
        args.append("--resume")
    return ["accelerate", "launch", "--num_processes", "2", "--multi_gpu",
            "-m", "llm_training.run_train", *args]


def _count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def check_data(workdir: Path = WORKDIR) -> None:
    for split, floor in (("train", 13000), ("val", 700)):
        path = workdir / "data" / "sft" / f"{DATA_STEM}_{split}.jsonl.gz"
        n = _count(path)
        print(f"{path}: {n} rows", flush=True)
        if n < floor:
            raise SystemExit(f"{split} row count {n} below expected floor {floor}")


def checkpoint_is_real(run_dir: Path) -> bool:
    state = run_dir / "checkpoint" / "trainer_state.pt"
    adapter = (run_dir / "checkpoint" / "adapter_model.safetensors").exists() or \
        (run_dir / "checkpoint" / "adapter_model.bin").exists()
    if not state.exists() or not adapter:
        return False
    import torch
    best_val = torch.load(state, map_location="cpu").get("best_val")
    return best_val is not None and math.isfinite(best_val)


def pull_checkpoint(workdir: Path = WORKDIR) -> None:
    from huggingface_hub import snapshot_download

    dst = workdir / "runs" / OUTPUT
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=CKPT_REPO, repo_type="model", local_dir=str(dst),
                      allow_patterns=["checkpoint/*", "best/*"])
    print(f"pulled checkpoint/best from {CKPT_REPO} -> {dst}", flush=True)


def preflight(workdir: Path = WORKDIR) -> None:
    check_data(workdir)
    tok_dir = workdir / "src" / "llm" / "models" / MODEL
    cmd = [sys.executable, "scripts/retrain_preflight.py", "--profile", DATA_STEM,
           "--tok-dir", str(tok_dir)]
    subprocess.run(cmd, check=True, cwd=workdir, env=run_env(workdir, ""))


def run_spec(spec: RunSpec, workdir: Path = WORKDIR, resume: bool = False) -> None:
    cmd = train_command(spec, resume=resume)
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=workdir, env=run_env(workdir, spec.ckpt_repo))


def smoke(workdir: Path = WORKDIR) -> None:
    run_spec(SMOKE, workdir, resume=False)
    state = workdir / "runs" / SMOKE.output / "checkpoint" / "trainer_state.pt"
    if not state.exists():
        raise SystemExit(f"smoke checkpoint missing: {state}")
    print("SMOKE OK: checkpoint saved and no Hub push was armed", flush=True)


def train(workdir: Path = WORKDIR, resume: bool = False) -> None:
    run_dir = workdir / "runs" / OUTPUT
    if resume:
        pull_checkpoint(workdir)
        if not checkpoint_is_real(run_dir):
            raise SystemExit("--resume requested but checkpoint is absent, incomplete, or never validated")
    run_spec(FINAL, workdir, resume=resume)
    result = run_dir / "train_result.json"
    if result.exists():
        print(json.dumps(json.loads(result.read_text()), indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["preflight", "smoke", "train"])
    ap.add_argument("--workdir", type=Path, default=WORKDIR)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    {"preflight": preflight, "smoke": smoke, "train": train}[args.action](args.workdir, args.resume) \
        if args.action == "train" else {"preflight": preflight, "smoke": smoke}[args.action](args.workdir)


if __name__ == "__main__":
    main()
