"""Unit tests for the isolated execution sandbox."""

from __future__ import annotations

from sandbox import ast_safety_screen, environment_description, run_tests


def test_safe_pytest_suite_passes() -> None:
    solution = """def add(a, b):
    return a + b
"""
    tests = """from solution import add

def test_add_normal():
    assert add(2, 3) == 5

def test_add_zero():
    assert add(0, 0) == 0
"""
    result = run_tests(solution, tests, timeout=5)
    assert result.ok
    assert result.tests_passed
    assert result.exit_code == 0
    assert "2 passed" in result.stdout


def test_failing_suite_is_reported_without_raising() -> None:
    solution = "def add(a, b):\n    return a - b\n"
    tests = """from solution import add

def test_add():
    assert add(2, 3) == 5
"""
    result = run_tests(solution, tests, timeout=5)
    assert result.ok
    assert not result.tests_passed
    assert result.exit_code == 1
    assert "1 failed" in (result.stdout + result.stderr)


def test_dangerous_import_is_blocked_before_execution() -> None:
    result = run_tests(
        'import subprocess\nsubprocess.run(["echo", "escaped"], check=False)\n',
        "def test_placeholder():\n    assert True\n",
        timeout=5,
    )
    assert result.blocked
    assert result.exit_code is None
    assert any("blocked import 'subprocess'" in reason for reason in result.block_reasons)


def test_dangerous_os_call_is_blocked() -> None:
    result = run_tests(
        'import os\nos.system("whoami")\n',
        "def test_placeholder():\n    assert True\n",
        timeout=5,
    )
    assert result.blocked
    assert any("blocked import 'os'" in reason for reason in result.block_reasons)


def test_open_and_absolute_database_path_are_blocked() -> None:
    open_reasons = ast_safety_screen('open("C:/Windows/win.ini").read()', "solution.py")
    assert any("blocked call 'open()'" in reason for reason in open_reasons)

    db_reasons = ast_safety_screen(
        'import sqlite3\nsqlite3.connect("C:/Windows/secret.db")\n',
        "solution.py",
    )
    assert any("blocked absolute path" in reason for reason in db_reasons)


def test_network_imports_are_blocked() -> None:
    for module in ("socket", "requests", "urllib", "httpx"):
        reasons = ast_safety_screen(f"import {module}\n", "test_solution.py")
        assert reasons, module


def test_timeout_is_a_hard_limit() -> None:
    result = run_tests(
        "def identity(value):\n    return value\n",
        "def test_hangs():\n    while True:\n        pass\n",
        timeout=1,
    )
    assert result.timed_out
    assert not result.tests_passed
    assert result.duration_seconds < 4.0


def test_unittest_fallback_runs() -> None:
    solution = "def multiply(a, b):\n    return a * b\n"
    tests = """import unittest
from solution import multiply

class TestMultiply(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(multiply(3, 4), 12)

if __name__ == "__main__":
    unittest.main()
"""
    result = run_tests(solution, tests, timeout=5, framework="unittest")
    assert result.framework == "unittest"
    assert result.tests_passed
    assert result.exit_code == 0


def test_environment_description_documents_isolation() -> None:
    description = environment_description()
    assert description["docker_required"] is False
    assert description["cwd"] == "temporary directory"
    assert "PYTHONPATH" in description["env_allowlist"]
    assert "subprocess" in description["blocked_imports"]
