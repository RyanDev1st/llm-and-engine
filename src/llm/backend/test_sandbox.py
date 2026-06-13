"""The `python` sandbox: real stdout grounds the claim, scripts with spaces run,
deterministic outputs match, and hangs/crashes/oversize are contained as clean
errors (never a leaked traceback or an unbounded flood)."""
from backend.sandbox import run_python


def test_returns_real_stdout():
    assert run_python('print(f"{0.15 * 86.40:.2f}")') == "output: 12.96"
    assert run_python('print(f"{sum([78, 92, 85]) / 3:.2f}")') == "output: 85.00"
    assert run_python('print(f"{max([12, 45, 7, 89]) - min([12, 45, 7, 89]):.2f}")') == "output: 82.00"


def test_deterministic_repeat():
    a = run_python('print(f"{128.40 / 3:.2f}")')
    b = run_python('print(f"{128.40 / 3:.2f}")')
    assert a == b == "output: 42.80"


def test_runtime_error_is_clean_no_traceback():
    out = run_python("print(1/0)")
    assert out.startswith("error: python_error:")
    assert "Traceback" not in out


def test_timeout_is_contained():
    assert run_python("while True: pass") == "error: python_timeout"


def test_guards_empty_and_oversize_and_no_output():
    assert run_python("") == "error: python_invalid"
    assert run_python("x = 9" * 200) == "error: python_invalid"   # over code cap
    assert run_python("x = 1 + 1").startswith("output: (no output")  # ran, printed nothing
