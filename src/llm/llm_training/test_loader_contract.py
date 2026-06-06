import json
import re

from llm_training.data_pipeline import load_jsonl_chat

TOOL = re.compile(r"<tool>\s*([a-zA-Z_]\w*)")

ROW = {
    "messages": [
        {"role": "user", "content": "play e4"},
        {"role": "assistant", "content": "<tool>load_skill name=chess-coach</tool>"},
        {"role": "tool", "content": "Use board tools before claims."},
        {"role": "assistant", "content": "<tool>move san=e4</tool>"},
        {"role": "tool", "content": "success: e4"},
        {"role": "assistant", "content": "Played e4."},
    ],
    "skills_index": [
        {"name": "chess-coach", "description": "Analyze positions.",
         "plugin": "chess-official", "source": "official_plugin", "enabled": True}
    ],
    "tool_manifest": [
        {"name": "load_skill", "description": "Load a skill.",
         "args": {"name": "required"}, "applies_when": "always", "plugin": "chess-official", "enabled": True},
        {"name": "move", "description": "Play a move.",
         "args": {"san": "required"}, "applies_when": "always", "plugin": "chess-official", "enabled": True},
    ],
    "plugin_context": {"installed": ["chess-official"], "enabled": ["chess-official"], "marketplace": []},
}


def _write(tmp_path, *rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_system_message_is_built_from_envelope(tmp_path):
    msgs = load_jsonl_chat(_write(tmp_path, ROW), 10)[0]
    assert msgs[0]["role"] == "system"
    sys = msgs[0]["content"]
    assert "chess-coach" in sys           # skills_index serialized
    assert "PLUGIN CONTEXT" in sys        # plugin_context serialized


def test_every_called_tool_is_declared_in_system(tmp_path):
    msgs = load_jsonl_chat(_write(tmp_path, ROW), 10)[0]
    sys = msgs[0]["content"]
    called = {t for m in msgs if m["role"] == "assistant" for t in TOOL.findall(m["content"])}
    assert called == {"load_skill", "move"}
    for tool in called:
        assert tool in sys, f"called tool {tool} not declared in system text"
