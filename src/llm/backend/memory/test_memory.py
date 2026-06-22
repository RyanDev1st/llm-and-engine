"""Persistent user memory: the extractor emits only durable typed facts, the write gate
dedupes/caps/supersedes and rejects board-state/PII, and the rendered block round-trips into
a prompt. Uses CHESS_MEMORY_DIR (a tmp dir) so it never touches the real profile."""
import importlib
import os

import pytest


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_MEMORY_DIR", str(tmp_path))
    from backend.memory import store
    importlib.reload(store)        # re-read _ROOT from the patched env
    return store


# --- extractor: only durable, typed facts (the gate's first half) ------------------------

def test_extract_rating_weakness_pref_style():
    from backend.memory.extract import extract_facts
    facts = dict(extract_facts("I'm rated 1200 and I always hang my queen, keep it short"))
    assert facts.get("rating") == "~1200"
    assert "weakness" in facts and "hang my queen" in facts["weakness"]
    assert facts.get("pref") == "prefers terse replies"

    style = dict(extract_facts("I play the London system"))
    assert style.get("style") == "plays london system"


def test_extract_ignores_transient_and_noise():
    from backend.memory.extract import extract_facts
    # a plain board question carries no durable fact
    assert extract_facts("what's the best move here?") == []
    # a bare number that isn't a rating is not captured
    assert not any(c == "rating" for c, _ in extract_facts("I have 8 pawns left"))


def test_explicit_remember_is_the_only_freeform_path():
    from backend.memory.extract import extract_facts
    facts = dict(extract_facts("remember that I prefer the Italian game"))
    assert facts.get("note") == "prefer the Italian game"


# --- the write gate: dedupe / cap / supersede / reject -----------------------------------

def test_gate_dedupes_and_supersedes_and_caps(mem):
    p = {}
    assert mem.add_fact(p, "rating", "~1200") is True
    assert mem.add_fact(p, "rating", "~1200") is False        # dedupe (single-valued)
    assert mem.add_fact(p, "rating", "~1500") is True and p["rating"] == "~1500"  # supersede
    for i in range(8):                                        # list cap = 6, oldest drops
        mem.add_fact(p, "weakness", f"tends to drop piece {i}")
    assert len(p["weakness"]) == 6 and "piece 0" not in " ".join(p["weakness"])


def test_gate_rejects_board_state_pii_and_unknown_category(mem):
    p = {}
    assert mem.add_fact(p, "note", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") is False
    assert mem.add_fact(p, "note", "email me at foo@bar.com") is False
    assert mem.add_fact(p, "weakness", "x" * 200) is False    # over-long
    assert mem.add_fact(p, "bogus_category", "anything") is False
    assert p == {}                                            # nothing got through


# --- end to end: capture persists, render injects, restart survives ----------------------

def test_capture_persists_and_renders(mem):
    n = mem.capture("I'm rated ~1400 and I keep hanging my queen, keep it brief", "tester")
    assert n >= 2
    block = mem.memory_block("tester")
    assert block.startswith("USER PROFILE") and "1400" in block
    assert "hang" in block.lower() and "terse" in block.lower()
    # a fresh load (simulating a new session / service restart) still has it
    assert mem.load_profile("tester").get("rating") == "~1400"


def test_capture_is_idempotent(mem):
    msg = "I'm rated 1300"
    assert mem.capture(msg, "t2") == 1
    assert mem.capture(msg, "t2") == 0            # re-capturing the same message writes nothing


def test_render_empty_profile_is_blank(mem):
    assert mem.render_profile({}) == ""
