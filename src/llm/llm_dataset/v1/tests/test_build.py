from __future__ import annotations

from llm_dataset.v1 import build
from llm_dataset.v1.profiles import profile


def test_build_uses_v1_2_profile_paths(tmp_path):
    p = profile("v1.2")
    assert p.train_path.as_posix().endswith("data/sft/v1_2_train.jsonl")
    assert p.val_path.as_posix().endswith("data/sft/v1_2_val.jsonl")


def test_build_cli_profile_resolves_v1_2_paths():
    p = build.profile("v1.2")
    assert p.gold_dir.as_posix().endswith("data/sft/v1_2")
