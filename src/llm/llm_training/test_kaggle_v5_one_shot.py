from __future__ import annotations

from pathlib import Path

from llm_training import kaggle_v5_one_shot as k


def test_smoke_command_is_real_20_step_v5_train_not_cli_smoke():
    cmd = k.train_command(k.SMOKE)
    text = " ".join(cmd)

    assert "--smoke" not in cmd
    assert "--data-stem v5" in text
    assert "--max-steps 20" in text
    assert "--max-examples 64" in text
    assert "--max-seq 1920" in text
    assert "--output v5_smoke_20" in text
    assert cmd[:5] == ["accelerate", "launch", "--num_processes", "2", "--multi_gpu"]


def test_final_train_command_uses_locked_v5_config():
    cmd = k.train_command(k.FINAL)
    text = " ".join(cmd)

    for required in (
        "--model gemma4_e4b",
        "--engine unsloth",
        "--data-stem v5",
        "--max-seq 1920",
        "--rank 16",
        "--targets all-linear",
        "--grad-accum 16",
        "--lr 0.0001",
        "--max-steps 1000",
        "--eval-every 100",
        "--max-val 128",
        "--save-every 50",
        "--output gemma4_chess_e4b_kaggle",
    ):
        assert required in text

    assert "--resume" not in cmd
    assert "--resume" in k.train_command(k.FINAL, resume=True)


def test_smoke_env_never_pushes_to_checkpoint_repo(monkeypatch):
    monkeypatch.setenv("CHESS_CKPT_REPO", "repo/that-must-not-be-used")

    env = k.run_env(Path("/kaggle/working/llm-and-engine"), ckpt_repo="")

    assert env["CHESS_CKPT_REPO"] == ""
    assert env["PYTHONPATH"] == "/kaggle/working/llm-and-engine/src/llm"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["TORCHDYNAMO_DISABLE"] == "1"
    assert env["UNSLOTH_COMPILE_DISABLE"] == "1"
