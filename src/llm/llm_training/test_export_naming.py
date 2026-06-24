"""export_gguf output naming — pure, no model. The names must carry the base tag (E2B/E4B) and the
quant so multiple quants of the same adapter (Q5_K_M + Q6_K for the A/B) coexist without clobbering."""
from pathlib import Path

from llm_training.export_gguf import model_tag, out_names


def test_model_tag_from_base_path():
    assert model_tag(Path("/x/models/gemma4_e2b")) == "E2B"
    assert model_tag(Path("/x/models/gemma4_e4b")) == "E4B"
    assert model_tag(Path("/content/unsloth/gemma-4-E4B-it")) == "E4B"


def test_out_names_carry_tag_and_quant():
    repo = Path("/repo")
    q5 = out_names(repo, Path("/m/gemma4_e4b"), "Q5_K_M")
    q6 = out_names(repo, Path("/m/gemma4_e4b"), "Q6_K")
    assert q5["quant"].name == "gemma4-E4B-chesscoach-Q5_K_M.gguf"
    assert q6["quant"].name == "gemma4-E4B-chesscoach-Q6_K.gguf"
    assert q5["quant"] != q6["quant"]                      # the two quants don't clobber
    assert q5["f16"] == q6["f16"]                          # but share one f16 (merge once)
    assert q5["merged"].name == "gemma4_e4b_chess_merged"
    assert q5["mmproj"].name == "mmproj-gemma4-e4b-vision-f16.gguf"
