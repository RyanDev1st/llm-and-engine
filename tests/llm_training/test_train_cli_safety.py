import pytest

from llm_training.train_gemma4_lora import TrainConfig, run_training


def test_train_defaults_safe_cpu() -> None:
    result = run_training(TrainConfig("router"))
    assert result["device"] == "cpu"
    assert result["dry_run"] is True


def test_cuda_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="CUDA requires --allow-cuda"):
        run_training(TrainConfig("router", device="cuda"))
