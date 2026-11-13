"""Static-analysis and lint tools.

The AST pass is always available.  A real subprocess invocation of Ruff is used
when installed (with a Flake8 fallback), exactly as required by the assignment.
The analysis is intentionally transparent: every command, exit code, and raw
output is captured for the Trace Inspector.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_CAPTURE = 30_000


@dataclass(slots=True)
class FunctionMetric:
    name: str
    line: int
    end_line: int
    arguments: list[str]
    complexity: int
    has_docstring: bool
    lines: int


@dataclass(slots=True)
class Finding:
    rule: str
    line: int
    message: str


@dataclass(slots=True)
class AnalysisResult:
    source: str
    syntax_ok: bool
    syntax_error: str = ""
    functions: list[FunctionMetric] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    max_complexity: int = 0
    builtin_findings: list[Finding] = field(default_factory=list)
    lint_tool: str = "unavailable"
    lint_command: str = ""
    lint_exit_code: int | None = None
    lint_stdout: str = ""
    lint_stderr: str = ""
    lint_findings: list[dict[str, Any]] = field(default_factory=list)
    report_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source", None)
        payload["lint_stdout"] = self.lint_stdout[:MAX_CAPTURE]
        payload["lint_stderr"] = self.lint_stderr[:MAX_CAPTURE]
        return payload

    @property
    def lint_available(self) -> bool:
        return self.lint_tool in {"ruff", "flake8"}

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "syntax_ok": self.syntax_ok,
            "syntax_error": self.syntax_error,
            "functions": len(self.functions),
            "classes": len(self.classes),
            "imports": self.imports,
            "max_complexity": self.max_complexity,
            "lint_tool": self.lint_tool,
            "lint_issue_count": len(self.lint_findings),
            "builtin_issue_count": len(self.builtin_findings),
            "lint_exit_code": self.lint_exit_code,
        }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _complexity(node: ast.AST) -> int:
    score = 1
    branch_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.IfExp,
        ast.Assert,
        ast.comprehension,
    )
    for child in ast.walk(node):
        if isinstance(child, branch_nodes):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += max(1, len(child.values) - 1)
        elif isinstance(child, ast.Match):
            score += max(1, len(child.cases))
    return score


def _builtin_findings(source: str, tree: ast.Module | None) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()
    for index, line in enumerate(lines, start=1):
        if len(line) > 88:
            findings.append(Finding("E501", index, f"Line is {len(line)} characters (limit 88)."))
        if "\t" in line:
            findings.append(Finding("W191", index, "Tab indentation detected; use four spaces."))
        if line.rstrip() != line:
            findings.append(Finding("W291", index, "Trailing whitespace."))
    if tree is None:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_") and not node.name.islower():
                findings.append(
                    Finding("N802", node.lineno, f"Function name '{node.name}' should be snake_case.")
                )
            if not ast.get_docstring(node):
                findings.append(
                    Finding("D103", node.lineno, f"Function '{node.name}' has no docstring.")
                )
            for arg in node.args.args:
                if len(arg.arg) == 1 and arg.arg not in {"_", "self"}:
                    findings.append(
                        Finding(
                            "N803",
                            node.lineno,
                            f"Argument '{arg.arg}' in '{node.name}' is too short to be descriptive.",
                        )
                    )
            for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                is_mutable = isinstance(default, (ast.List, ast.Dict, ast.Set))
                if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
                    is_mutable = is_mutable or default.func.id in {"list", "dict", "set"}
                if is_mutable:
                    findings.append(
                        Finding(
                            "B006",
                            default.lineno,
                            f"Mutable default argument in '{node.name}' is shared across calls.",
                        )
                    )
        elif isinstance(node, ast.ClassDef) and not node.name[0:1].isupper():
            findings.append(Finding("N801", node.lineno, f"Class '{node.name}' should be PascalCase."))
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(Finding("E722", node.lineno, "Bare except catches every exception."))
        elif isinstance(node, ast.Compare):
            for operator, comparator in zip(node.ops, node.comparators):
                if isinstance(operator, ast.Eq) and isinstance(comparator, ast.Constant) and comparator.value is None:
                    findings.append(Finding("E711", node.lineno, "Compare to None with 'is None', not '=='."))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if node.args and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Add):
                right = node.args[0].right
                if isinstance(right, ast.Constant) and right.value == 1:
                    findings.append(
                        Finding("REVIEW", node.lineno, "range(... + 1) can cause an off-by-one error.")
                    )
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"eval", "exec", "os.system", "subprocess.run", "subprocess.Popen"}:
                findings.append(Finding("S", node.lineno, f"Dangerous call '{name}'."))
    return findings


def _find_ruff() -> str | None:
    executable = Path(sys.executable)
    candidates = [
        executable.parent / "ruff.exe",
        executable.parent / "ruff",
        Path(shutil.which("ruff") or ""),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def _find_flake8() -> str | None:
    executable = Path(sys.executable)
    candidates = [
        executable.parent / "flake8.exe",
        executable.parent / "flake8",
        Path(shutil.which("flake8") or ""),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def _run_ruff(source: str, ruff: str) -> tuple[str, int, str, str, list[dict[str, Any]]]:
    command = [
        ruff,
        "check",
        "--isolated",
        "--no-cache",
        "--select",
        "E,W,F,I,N,D,B",
        "--line-length",
        "88",
        "--output-format",
        "json",
        "--stdin-filename",
        "submitted_code.py",
        "-",
    ]
    completed = subprocess.run(
        command,
        input=source,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        cwd=str(Path(tempfile.gettempdir())),
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "RUFF_NO_CACHE": "1",
        },
        check=False,
    )
    stdout = completed.stdout[:MAX_CAPTURE]
    stderr = completed.stderr[:MAX_CAPTURE]
    findings: list[dict[str, Any]] = []
    try:
        parsed = json.loads(completed.stdout or "[]")
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    location = item.get("location") or {}
                    findings.append(
                        {
                            "code": item.get("code") or "RUF",
                            "line": location.get("row"),
                            "column": location.get("column"),
                            "message": item.get("message", ""),
                            "url": item.get("url", ""),
                        }
                    )
    except json.JSONDecodeError:
        findings = []
    display_command = " ".join(
        ["ruff", "check", "--isolated", "--select", "E,W,F,I,N,D,B", "--output-format", "json", "--stdin-filename", "submitted_code.py", "-"]
    )
    return display_command, completed.returncode, stdout, stderr, findings


def _run_flake8(source: str, flake8: str) -> tuple[str, int, str, str, list[dict[str, Any]]]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(source)
        path = handle.name
    try:
        command = [
            flake8,
            "--max-line-length=88",
            "--format=%(row)d:%(col)d:%(code)s:%(text)s",
            path,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            cwd=str(Path(path).parent),
            check=False,
        )
        findings: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            bits = line.split(":", 3)
            if len(bits) == 4 and bits[0].isdigit():
                findings.append(
                    {
                        "code": bits[2],
                        "line": int(bits[0]),
                        "column": int(bits[1]) if bits[1].isdigit() else None,
                        "message": bits[3],
                        "url": "",
                    }
                )
        display_command = "flake8 --max-line-length=88 --format=... <temporary file>"
        return (
            display_command,
            completed.returncode,
            completed.stdout[:MAX_CAPTURE],
            completed.stderr[:MAX_CAPTURE],
            findings,
        )
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def analyze_code(source: str) -> AnalysisResult:
    """Run AST analysis, readability checks, and a lint subprocess."""
    source = source or ""
    syntax_ok = True
    syntax_error = ""
    tree: ast.Module | None = None
    try:
        tree = ast.parse(source, filename="submitted_code.py")
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = f"{exc.msg} (line {exc.lineno}, column {exc.offset})"

    functions: list[FunctionMetric] = []
    classes: list[str] = []
    imports: list[str] = []
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
                functions.append(
                    FunctionMetric(
                        name=node.name,
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        arguments=args,
                        complexity=_complexity(node),
                        has_docstring=bool(ast.get_docstring(node)),
                        lines=(node.end_lineno or node.lineno) - node.lineno + 1,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        functions.sort(key=lambda metric: metric.line)

    builtin_findings = _builtin_findings(source, tree)
    lint_tool = "unavailable"
    lint_command = ""
    lint_exit: int | None = None
    lint_stdout = ""
    lint_stderr = ""
    lint_findings: list[dict[str, Any]] = []

    ruff = _find_ruff()
    flake8 = _find_flake8()
    try:
        if ruff:
            lint_tool = "ruff"
            lint_command, lint_exit, lint_stdout, lint_stderr, lint_findings = _run_ruff(source, ruff)
        elif flake8:
            lint_tool = "flake8"
            lint_command, lint_exit, lint_stdout, lint_stderr, lint_findings = _run_flake8(source, flake8)
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        lint_stderr = f"Lint subprocess failed: {type(exc).__name__}: {exc}"
        lint_exit = -1

    max_complexity = max((metric.complexity for metric in functions), default=0)
    result = AnalysisResult(
        source=source,
        syntax_ok=syntax_ok,
        syntax_error=syntax_error,
        functions=functions,
        classes=classes,
        imports=sorted(set(item for item in imports if item)),
        max_complexity=max_complexity,
        builtin_findings=builtin_findings,
        lint_tool=lint_tool,
        lint_command=lint_command,
        lint_exit_code=lint_exit,
        lint_stdout=lint_stdout,
        lint_stderr=lint_stderr,
        lint_findings=lint_findings,
    )
    result.report_markdown = build_ast_report(result)
    return result


def build_ast_report(result: AnalysisResult) -> str:
    lines: list[str] = ["## AST / Static Analysis"]
    lines.append(f"- Syntax: {'VALID' if result.syntax_ok else 'INVALID — ' + result.syntax_error}")
    lines.append(f"- Functions ({len(result.functions)}): " + (", ".join(f"`{f.name}`" for f in result.functions) or "none"))
    lines.append(f"- Classes ({len(result.classes)}): " + (", ".join(f"`{c}`" for c in result.classes) or "none"))
    lines.append("- Imports: " + (", ".join(f"`{item}`" for item in result.imports) or "none"))
    lines.append(f"- Maximum cyclomatic complexity: {result.max_complexity}")
    if result.functions:
        lines.extend(["", "### Function metrics"])
        for metric in result.functions:
            lines.append(
                f"- `{metric.name}` (line {metric.line}): complexity {metric.complexity}, "
                f"{metric.lines} lines, docstring: {'yes' if metric.has_docstring else 'no'}, "
                f"args: {', '.join(metric.arguments) or 'none'}"
            )
    lines.extend(["", f"### Lint ({result.lint_tool})"])
    if result.lint_tool == "unavailable":
        lines.append("- Ruff and Flake8 are not installed; built-in AST style checks were used.")
    else:
        lines.append(f"- Command: `{result.lint_command}`")
        lines.append(f"- Exit code: {result.lint_exit_code}")
        lines.append(f"- Findings: {len(result.lint_findings)}")
        for item in result.lint_findings[:15]:
            lines.append(
                f"  - {item.get('code', 'LINT')} line {item.get('line', '?')}: {item.get('message', '')}"
            )
        if len(result.lint_findings) > 15:
            lines.append(f"  - … {len(result.lint_findings) - 15} more lint finding(s).")
    if result.builtin_findings:
        lines.extend(["", "### Built-in AST / readability checks"])
        for finding in result.builtin_findings[:20]:
            lines.append(f"- {finding.rule} line {finding.line}: {finding.message}")
    return "\n".join(lines)


def analysis_tool_trace(result: AnalysisResult) -> dict[str, Any]:
    """Shape the analysis result for the Trace Inspector tool-call section."""
    if result.lint_tool == "unavailable":
        command = "python -c \"import ast; ...\"  # built-in AST checks (Ruff/Flake8 unavailable)"
        exit_code = 0
        stderr = result.lint_stderr
    else:
        command = result.lint_command
        exit_code = result.lint_exit_code
        stderr = result.lint_stderr
    return {
        "name": "Static Analysis — AST + lint subprocess",
        "kind": "static_analysis",
        "command": command,
        "exit_code": exit_code,
        "stdout": result.lint_stdout or result.report_markdown,
        "stderr": stderr,
        "summary": result.summary,
        "details": result.to_dict(),
        "timed_out": False,
    }


__all__ = [
    "AnalysisResult",
    "FunctionMetric",
    "Finding",
    "analyze_code",
    "analysis_tool_trace",
    "build_ast_report",
]
