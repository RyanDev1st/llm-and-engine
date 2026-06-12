import importlib


def test_model_gguf_default_path_is_q5(monkeypatch):
    monkeypatch.delenv("CHESS_GGUF_PATH", raising=False)
    mod = importlib.reload(importlib.import_module("backend.model_gguf"))

    # Q5_K_M is the serving default (better eval fidelity than the old Q4_0).
    assert mod.default_gguf_path() == mod.REPO / "runs" / "gemma4-E2B-chesscoach-Q5_K_M.gguf"


def test_model_gguf_path_can_come_from_env(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma4-30b-q4.gguf"
    monkeypatch.setenv("CHESS_GGUF_PATH", str(gguf))
    mod = importlib.reload(importlib.import_module("backend.model_gguf"))

    assert mod.default_gguf_path() == gguf


def test_model_gguf_runtime_defaults_are_unchanged(monkeypatch):
    monkeypatch.delenv("CHESS_N_CTX", raising=False)
    monkeypatch.delenv("CHESS_N_GPU_LAYERS", raising=False)
    mod = importlib.reload(importlib.import_module("backend.model_gguf"))

    assert mod.gguf_runtime_config() == (4096, -1)


def test_model_gguf_runtime_accepts_env(monkeypatch):
    monkeypatch.setenv("CHESS_N_CTX", "4096")
    monkeypatch.setenv("CHESS_N_GPU_LAYERS", "60")
    mod = importlib.reload(importlib.import_module("backend.model_gguf"))

    assert mod.gguf_runtime_config() == (4096, 60)


def test_server_bind_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("CHESS_HOST", raising=False)
    monkeypatch.delenv("CHESS_PORT", raising=False)
    server = importlib.reload(importlib.import_module("backend.server"))

    assert server.bind_address() == ("127.0.0.1", 7860)


def test_server_bind_accepts_tunnel_env(monkeypatch):
    monkeypatch.setenv("CHESS_HOST", "0.0.0.0")
    monkeypatch.setenv("CHESS_PORT", "7861")
    server = importlib.reload(importlib.import_module("backend.server"))

    assert server.bind_address() == ("0.0.0.0", 7861)
