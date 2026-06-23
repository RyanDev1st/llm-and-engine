"""Generation stop set: the model finished its reply but kept decoding <pad> to the token cap (the
'NUL' flood in the serve log = dozens-of-seconds latency on EVERY reply) because Gemma's gen_config
omits the turn-ender from eos. _stop_token_ids adds the turn-ender (<end_of_turn>/<turn|>) and <pad>
so generation halts at the real turn end. Verified against the real local tokenizer (CPU, no model)."""
from pathlib import Path

import pytest

_TOK_DIR = Path(__file__).resolve().parents[1] / "models" / "gemma4_e2b"


def _tok():
    if not (_TOK_DIR / "tokenizer_config.json").exists():
        pytest.skip("local gemma tokenizer not present")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(_TOK_DIR), local_files_only=True)


def test_stop_ids_include_turn_ender_eos_and_pad():
    from backend.model_hf import _stop_token_ids
    tok = _tok()
    ids = _stop_token_ids(tok)
    assert tok.eos_token_id in ids                    # <eos>
    assert tok.pad_token_id in ids                    # <pad> — stops the degenerate pad run
    assert tok.convert_tokens_to_ids("<turn|>") in ids  # the chat turn-ender (id 106 here)
    # The <unk> fallback (3, from the absent "<end_of_turn>" literal) must NOT leak in.
    assert tok.unk_token_id not in ids
