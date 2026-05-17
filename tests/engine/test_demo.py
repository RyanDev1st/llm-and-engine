from engine.research.demo import run_demo


def test_demo_runs_spec_tool_calls() -> None:
    transcript = run_demo()

    assert len(transcript) == 8
    assert transcript[0] == ("<tool>move san=e4</tool>", "success: e4")
    assert transcript[-2][1].startswith("review: Nf3, label=mistake")
    assert transcript[-1][1].startswith("pieces: ")
    assert transcript[-1][1].endswith("R=h8")
