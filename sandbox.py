"""Isolated, Docker-free execution sandbox for generated tests.

Isolation layers
----------------
1. Every run gets a fresh ``tempfile.TemporaryDirectory`` which is also its CWD.
2. Generated test code *and* candidate solution are AST-screened before writing.
3. The subprocess receives a minimal environment allow-list: no API keys, no
   user site-packages injection, ``PYTHONPATH`` restricted to the temp directory.
4. A hard ``subprocess.run(..., timeout=...)`` limit stops hangs (default 5 s).
5. Only ``solution.py`` and ``test_solution.py`` are created in the temp folder.

This is deliberately portable on Windows 11 — no Docker, no privileged APIs.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 200_000
MAX_OUTPUT = 50_000

BLOCKED_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "threading",
    "requests",
    "urllib",
    "urllib2",
    "http",
    "httpx",
    "ftplib",
    "telnetlib",
    "smtplib",
    "paramiko",
    "pty",
    "resource",
    "signal",
    "webbrowser",
    "importlib",
    "builtins",
    "gc",
    "inspect",
    "pathlib",
    "tempfile",
}

BLOCKED_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "__import__",
    "globals",
    "locals",
    "vars",
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_output",
    "shutil.rmtree",
    "socket.socket",
    "socket.create_connection",
}

BLOCKED_ATTRIBUTES = {
    "__globals__",
    "__builtins__",
    "__subclasses__",
    "__mro__",
    "__bases__",
    "__code__",
    "rmtree",
    "system",
    "popen",
    "spawn",
    "spawnl",
    "spawnv",
    "spawnve",
    "fork",
    "execv",
    "execve",
    "execl",
    "unlink",
    "rmdir",
}


@dataclass(slots=True)
class SandboxResult:
    blocked: bool
    timed_out: bool
    tests_passed: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timeout_seconds: int = 5
    command: str = ""
    framework: str = "pytest"
    block_reasons: list[str] = field(default_factory=list)
    preflight_error: str = ""
    files: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.timed_out and not self.preflight_error

    @property
    def status_label(self) -> str:
        if self.blocked:
            return "BLOCKED BY SAFETY SCREEN"
        if self.timed_out:
            return f"TIMEOUT after {self.timeout_seconds}s"
        if self.preflight_error:
            return "PRE-FLIGHT ERROR"
        if self.tests_passed:
            return "PASSED"
        return "FAILED"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        payload["status_label"] = self.status_label
        return payload

    def trace_record(self) -> dict[str, Any]:
        return {
            "name": f"Isolated Sandbox — {self.framework}",
            "kind": "sandbox",
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "timeout_seconds": self.timeout_seconds,
            "duration_seconds": round(self.duration_seconds, 3),
            "blocked": self.blocked,
            "block_reasons": self.block_reasons,
            "status": self.status_label,
        }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_absolute_path(value: str) -> bool:
    if not value or "\n" in value:
        return False
    if value.startswith(("/", "\\")):
        return True
    return len(value) > 2 and value[1] == ":" and value[2] in {"\\", "/"}


def ast_safety_screen(source: str, label: str = "source") -> list[str]:
    """Return human-readable reasons why the code is unsafe to execute."""
    reasons: list[str] = []
    if "\x00" in source:
        return [f"{label}: NUL byte detected"]
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return [f"{label}: exceeds {MAX_SOURCE_BYTES // 1000} KB source limit"]
    try:
        tree = ast.parse(source, filename=f"{label}.py")
    except SyntaxError as exc:
        return [f"{label}: syntax error at line {exc.lineno}: {exc.msg}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_IMPORT_ROOTS:
                    reasons.append(f"{label}:{node.lineno} blocked import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BLOCKED_IMPORT_ROOTS:
                reasons.append(f"{label}:{node.lineno} blocked import from '{node.module}'")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in BLOCKED_CALL_NAMES:
                reasons.append(f"{label}:{node.lineno} blocked call '{name}()'")
            if name.endswith(".open"):
                reasons.append(f"{label}:{node.lineno} blocked call '{name}()'")
            if name.endswith("connect") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if _is_absolute_path(first.value):
                        reasons.append(
                            f"{label}:{node.lineno} blocked absolute path in '{name}()'"
                        )
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRIBUTES:
                reasons.append(
                    f"{label}:{node.lineno} blocked attribute access '.{node.attr}'"
                )
        elif isinstance(node, ast.Name) and node.id in {
            "__builtins__",
            "__import__",
        }:
            reasons.append(f"{label}:{node.lineno} blocked name '{node.id}'")
    # Preserve order but remove duplicates.
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return unique[:20]


def _environment(temp_dir: str) -> dict[str, str]:
    """A deliberately small environment. Secrets in the parent can never leak."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "PATHEXT": os.environ.get("PATHEXT", ".EXE;.CMD;.BAT;.COM"),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", os.environ.get("WINDIR", "")),
        "WINDIR": os.environ.get("WINDIR", os.environ.get("SYSTEMROOT", "")),
        "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", "C:"),
        "COMSPEC": os.environ.get("COMSPEC", ""),
        "TEMP": temp_dir,
        "TMP": temp_dir,
        "HOME": temp_dir,
        "USERPROFILE": temp_dir,
        "PYTHONPATH": temp_dir,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "NO_COLOR": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    return {key: value for key, value in env.items() if value}


def run_tests(
    solution_code: str,
    test_code: str,
    *,
    timeout: int = 5,
    framework: str = "pytest",
) -> SandboxResult:
    """AST-screen and execute generated tests in a disposable temp directory."""
    timeout = max(1, min(int(timeout), 30))
    reasons = ast_safety_screen(solution_code, "solution.py")
    reasons.extend(ast_safety_screen(test_code, "test_solution.py"))
    if reasons:
        return SandboxResult(
            blocked=True,
            timed_out=False,
            tests_passed=False,
            exit_code=None,
            timeout_seconds=timeout,
            command="not executed — safety screen rejected the submission",
            block_reasons=reasons,
            framework=framework,
            files={"solution.py": solution_code, "test_solution.py": test_code},
        )

    choose_unittest = framework == "unittest" or (
        framework == "auto" and "unittest" in test_code and "pytest" not in test_code
    )
    framework = "unittest" if choose_unittest else "pytest"
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="code_review_sandbox_") as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "solution.py").write_text(solution_code, encoding="utf-8")
        (temp_path / "test_solution.py").write_text(test_code, encoding="utf-8")
        if framework == "unittest":
            args = [sys.executable, "-m", "unittest", "discover", "-v", "-s", "."]
            display = "python -m unittest discover -v -s ."
        else:
            args = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--disable-warnings",
                "-p",
                "no:cacheprovider",
                f"--basetemp={temp_path / 'pytest_tmp'}",
                "test_solution.py",
            ]
            display = (
                "python -m pytest -q --disable-warnings -p no:cacheprovider "
                "--basetemp=<temp>/pytest_tmp test_solution.py"
            )
        env = _environment(temp_dir)
        try:
            completed = subprocess.run(
                args,
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
            elapsed = time.perf_counter() - started
            return SandboxResult(
                blocked=False,
                timed_out=False,
                tests_passed=completed.returncode == 0,
                exit_code=completed.returncode,
                stdout=completed.stdout[:MAX_OUTPUT],
                stderr=completed.stderr[:MAX_OUTPUT],
                duration_seconds=elapsed,
                timeout_seconds=timeout,
                command=display,
                framework=framework,
                files={"solution.py": solution_code, "test_solution.py": test_code},
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            stdout = _coerce_output(exc.stdout)
            stderr = _coerce_output(exc.stderr)
            return SandboxResult(
                blocked=False,
                timed_out=True,
                tests_passed=False,
                exit_code=None,
                stdout=stdout[:MAX_OUTPUT],
                stderr=(stderr + f"\nSandbox terminated after {timeout}s.").strip()[:MAX_OUTPUT],
                duration_seconds=elapsed,
                timeout_seconds=timeout,
                command=display,
                framework=framework,
                files={"solution.py": solution_code, "test_solution.py": test_code},
            )
        except OSError as exc:
            elapsed = time.perf_counter() - started
            return SandboxResult(
                blocked=False,
                timed_out=False,
                tests_passed=False,
                exit_code=None,
                stderr=f"Failed to start test runner: {type(exc).__name__}: {exc}",
                duration_seconds=elapsed,
                timeout_seconds=timeout,
                command=display,
                framework=framework,
                files={"solution.py": solution_code, "test_solution.py": test_code},
            )


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def environment_description() -> dict[str, Any]:
    """UI-facing description of the isolation guarantees."""
    return {
        "temp_directory": "tempfile.TemporaryDirectory (auto-deleted)",
        "cwd": "temporary directory",
        "pythonpath": "temporary directory only",
        "env_allowlist": sorted(_environment("<temp>").keys()),
        "timeout_seconds": "hard limit via subprocess.run(timeout=...)",
        "docker_required": False,
        "blocked_imports": sorted(BLOCKED_IMPORT_ROOTS),
        "blocked_calls": sorted(BLOCKED_CALL_NAMES),
        "blocked_attributes": sorted(BLOCKED_ATTRIBUTES),
    }


__all__ = [
    "SandboxResult",
    "ast_safety_screen",
    "environment_description",
    "run_tests",
]
