"""Deterministic offline chat model used for credential-free demonstrations.

This is intentionally *not* hidden: the UI labels it "Offline Demo".  It is a
real ``BaseChatModel`` implementation, so every LCEL chain, parser, prompt, and
routing path is exercised exactly as it is with Gemini, OpenAI, or Ollama.
The responses are heuristic and deterministic, which also makes the repository
unit tests reproducible without network access.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ---------------------------------------------------------------------------
# Prompt parsing helpers
# ---------------------------------------------------------------------------

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", item))))
        return "\n".join(parts)
    return str(content)


def _last_human(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type in {"human", "user"}:
            return _content_to_text(message.content)
    return ""


def _section(text: str, tag: str, default: str = "") -> str:
    match = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL)
    return match.group(1).strip() if match else default


def _request_from_prompt(prompt: str) -> str:
    return _section(prompt, "request")


def _code_from_prompt(prompt: str, tag: str = "code") -> str:
    return _section(prompt, tag)


def _parse_ast(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _line_of(source: str, needle: str) -> int:
    for index, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return index
    return 0


def _slug(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------------------
# Demonstration analysis
# ---------------------------------------------------------------------------

def _scan_bugs(source: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    tree = _parse_ast(source)

    def add(severity: str, title: str, line: int, why: str, trigger: str, fix: str) -> None:
        issues.append(
            {
                "severity": severity,
                "title": title,
                "line": line,
                "why": why,
                "trigger": trigger,
                "fix": fix,
            }
        )

    if tree is None:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            add(
                "Critical",
                "Syntax error prevents execution",
                exc.lineno or 1,
                exc.msg,
                "Any call or import of this module.",
                "Correct the reported syntax before running tests.",
            )
        return issues

    # Mutable default arguments and class-level shared state.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in list(node.args.defaults) + list(node.args.kw_defaults):
                if default is None:
                    continue
                mutable = isinstance(default, (ast.List, ast.Dict, ast.Set))
                if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
                    mutable = mutable or default.func.id in {"list", "dict", "set"}
                if mutable:
                    add(
                        "High",
                        f"Mutable default argument in '{node.name}'",
                        node.lineno,
                        "Default containers are created once when the module is imported and are shared by every call.",
                        "Call the function twice and observe state leaking between calls, or inspect User.roles across instances.",
                        "Use None as the default and create a fresh container inside the function.",
                    )

    # Common high-risk patterns.
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Call) and isinstance(node.right.func, ast.Name) and node.right.func.id == "len":
                add(
                    "High",
                    "Division by len() can divide by zero",
                    node.lineno,
                    "An empty input makes len(...) zero, raising ZeroDivisionError.",
                    "Call the function with an empty list, string, or dictionary.",
                    "Validate that the input is non-empty before dividing, or return a documented default.",
                )
            elif isinstance(node.right, ast.Name):
                add(
                    "Medium",
                    f"Division by '{node.right.id}' has no zero guard",
                    node.lineno,
                    "A caller can pass zero as the divisor, producing ZeroDivisionError.",
                    f"Call the function with {node.right.id}=0.",
                    "Check for zero (or catch ZeroDivisionError) and raise a descriptive error.",
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if node.args and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Add):
                right = node.args[0].right
                if isinstance(right, ast.Constant) and right.value == 1:
                    add(
                        "High",
                        "Possible off-by-one in range(... + 1)",
                        node.lineno,
                        "Iterating one element past the end of a sequence raises IndexError; for a numeric range it may include one extra value.",
                        "Use an input whose length equals the loop bound.",
                        "Remove the '+ 1' when iterating over a sequence, or make the inclusive bound explicit and safe.",
                    )
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 0:
            parent_guard = any(
                isinstance(other, ast.If)
                and isinstance(other.test, ast.UnaryOp)
                and isinstance(other.test.op, ast.Not)
                for other in ast.walk(tree)
            )
            if not parent_guard:
                add(
                    "Medium",
                    "First-element access without an empty-input guard",
                    node.lineno,
                    "Indexing element 0 of an empty sequence raises IndexError.",
                    "Pass [] (or another empty sequence) to the function.",
                    "Check that the sequence is non-empty before indexing it.",
                )

    # SQL injection and leaked database connections.
    has_select = bool(re.search(r"SELECT\s+.+FROM", source, re.IGNORECASE))
    dynamic_sql = bool(
        re.search(r"SELECT[^\n]*(\+|\.format\(|f[\"'])", source, re.IGNORECASE)
    )
    if has_select and dynamic_sql:
        add(
            "Critical",
            "SQL query built by string concatenation/formatting",
            _line_of(source, "SELECT") or _line_of(source, "execute") or 1,
            "Untrusted values become SQL syntax, enabling SQL injection.",
            "Pass a value such as \"' OR '1'='1\" as the username.",
            "Use a parameterized query: execute(sql, (value,)).",
        )
    if "sqlite3.connect" in source and not (".close()" in source or "closing(" in source):
        add(
            "High",
            "Database connection is never closed",
            _line_of(source, "sqlite3.connect") or 1,
            "The connection and cursor leak file handles and locks on every call.",
            "Repeated callers exhaust resources or leave the database locked.",
            "Use try/finally with conn.close(), or contextlib.closing.",
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            add(
                "Medium",
                "Bare except swallows all failures",
                node.lineno,
                "KeyboardInterrupt, SystemExit, and programming errors are caught indiscriminately.",
                "An unexpected error is silently hidden and debugging becomes difficult.",
                "Catch the narrow exception type and log or re-raise it.",
            )
        if isinstance(node, ast.ExceptHandler) and isinstance(node.body[0], ast.Pass):
            add(
                "Medium",
                "Exception is swallowed with pass",
                node.lineno,
                "Failures silently produce an incorrect or missing result.",
                "Cause the protected operation to fail and observe the missing error signal.",
                "Handle the exception, return a documented fallback, or re-raise it.",
            )
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) and comparator.value is None:
                    add(
                        "Low",
                        "Comparison to None uses == instead of is",
                        node.lineno,
                        "Identity comparison is the correct and conventional way to test None.",
                        "Custom objects can override equality and produce surprising results.",
                        "Write 'value is None' / 'value is not None'.",
                    )

    dangerous = {
        "eval": "eval executes arbitrary code",
        "exec": "exec executes arbitrary code",
        "os.system": "os.system launches shell commands",
        "subprocess.run": "subprocess launches external processes",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in dangerous:
                add(
                    "Critical",
                    f"Dangerous call: {name}",
                    node.lineno,
                    dangerous[name],
                    "Untrusted input can execute arbitrary commands.",
                    "Remove the dynamic execution or strictly validate all inputs.",
                )

    # De-duplicate by title + line while preserving order.
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for issue in issues:
        key = (issue["title"], issue["line"])
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _style_issues(source: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    tree = _parse_ast(source)
    if tree is None:
        return [
            {
                "rule": "Syntax",
                "line": "1",
                "message": "Cannot evaluate style until the syntax error is fixed.",
                "fix": "Correct the syntax error.",
            }
        ]
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", node.name):
                issues.append(
                    {
                        "rule": "Naming (PEP 8)",
                        "line": str(node.lineno),
                        "message": f"Function '{node.name}' is not snake_case.",
                        "fix": "Rename it using lowercase words separated by underscores.",
                    }
                )
            if not ast.get_docstring(node):
                issues.append(
                    {
                        "rule": "Docstrings (PEP 257)",
                        "line": str(node.lineno),
                        "message": f"Function '{node.name}' has no docstring.",
                        "fix": "Add a one-line imperative docstring describing behavior and errors.",
                    }
                )
            short_args = [
                arg.arg
                for arg in list(node.args.args) + list(node.args.kwonlyargs)
                if len(arg.arg) == 1 and arg.arg != "self"
            ]
            if short_args:
                issues.append(
                    {
                        "rule": "Naming readability",
                        "line": str(node.lineno),
                        "message": f"Single-character parameter(s): {', '.join(short_args)}.",
                        "fix": "Use descriptive names such as left, right, operator.",
                    }
                )
            if node.end_lineno and node.end_lineno - node.lineno > 35:
                issues.append(
                    {
                        "rule": "Function length",
                        "line": str(node.lineno),
                        "message": f"Function '{node.name}' is longer than 35 lines.",
                        "fix": "Extract cohesive helper functions.",
                    }
                )
    for index, line in enumerate(lines, start=1):
        if len(line) > 88:
            issues.append(
                {
                    "rule": "Line length (E501)",
                    "line": str(index),
                    "message": f"Line is {len(line)} characters (limit 88).",
                    "fix": "Wrap the expression across lines.",
                }
            )
        if "\t" in line:
            issues.append(
                {
                    "rule": "Indentation (W191)",
                    "line": str(index),
                    "message": "Tab indentation used.",
                    "fix": "Use four spaces per indentation level.",
                }
            )
    elif_chain = sum(1 for line in lines if re.match(r"\s*elif\s+", line))
    if elif_chain >= 3:
        issues.append(
            {
                "rule": "Clean code",
                "line": str(_line_of(source, "elif") or 1),
                "message": f"Long if/elif chain ({elif_chain} branches).",
                "fix": "Use a dispatch dictionary or match statement.",
            }
        )
    if "print(" in source:
        issues.append(
            {
                "rule": "Clean code",
                "line": str(_line_of(source, "print(") or 1),
                "message": "Debug print() left in library code.",
                "fix": "Use logging or remove the print.",
            }
        )
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for issue in issues:
        key = (issue["rule"], issue["line"], issue["message"])
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique[:8]


# ---------------------------------------------------------------------------
# Code fixing helpers
# ---------------------------------------------------------------------------

def _guard_for_empty(name: str) -> str:
    return f'if not {name}:\n    raise ValueError("{name} must not be empty")'


def _insert_at_body(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef, snippet: str) -> str:
    """Insert a dedented snippet as the first statement of a function."""
    lines = source.splitlines()
    if not node.body:
        return source
    first = node.body[0]
    insert_index = first.lineno - 1
    indent = " " * (first.col_offset if first.col_offset else 4)
    block = "\n".join(indent + line if line else line for line in snippet.splitlines())
    lines.insert(insert_index, block)
    return "\n".join(lines)


def _fix_mutable_defaults(source: str) -> str:
    tree = _parse_ast(source)
    if tree is None:
        return source
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    functions.sort(key=lambda node: node.lineno, reverse=True)
    for node in functions:
        mutable_defaults: dict[str, str] = {}
        defaults = node.args.defaults or []
        paired_args = node.args.args[-len(defaults):] if defaults else []
        for arg, default in zip(paired_args, defaults):
            mutable = isinstance(default, (ast.List, ast.Dict, ast.Set))
            kind = "list"
            if isinstance(default, ast.Dict):
                kind = "dict"
            elif isinstance(default, ast.Set):
                kind = "set"
            if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
                mutable = mutable or default.func.id in {"list", "dict", "set"}
                if default.func.id in {"list", "dict", "set"}:
                    kind = default.func.id
            if mutable:
                mutable_defaults[arg.arg] = kind
        if not mutable_defaults:
            continue
        lines = source.splitlines()
        header_start = node.lineno - 1
        body_line = node.body[0].lineno - 1 if node.body else node.lineno
        header = "\n".join(lines[header_start:body_line])
        for name in mutable_defaults:
            header = re.sub(
                rf"(\b{re.escape(name)}\s*=\s*)(?:\[\]|\{{\}}|(?:list|dict|set)\(\))",
                r"\1None",
                header,
            )
        new_lines = lines[:header_start] + header.splitlines() + lines[body_line:]
        source = "\n".join(new_lines)
        # Re-parse so the body offset reflects the rewritten source.
        tree2 = _parse_ast(source)
        if tree2 is None:
            continue
        current = next(
            (
                item
                for item in ast.walk(tree2)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == node.name
            ),
            None,
        )
        if current is not None:
            for name, kind in mutable_defaults.items():
                factory = {"list": "[]", "dict": "{}", "set": "set()"}[kind]
                snippet = f"if {name} is None:\n    {name} = {factory}"
                source = _insert_at_body(source, current, snippet)
                tree2 = _parse_ast(source)
                if tree2 is None:
                    break
                current = next(
                    (
                        item
                        for item in ast.walk(tree2)
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == node.name
                    ),
                    None,
                )
                if current is None:
                    break
    return source


def _mutable_kind_from_source(source: str, function_name: str, arg_name: str) -> str:
    tree = _parse_ast(source)
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                for arg, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
                    if arg.arg == arg_name:
                        if isinstance(default, ast.Dict) or (
                            isinstance(default, ast.Call)
                            and isinstance(default.func, ast.Name)
                            and default.func.id == "dict"
                        ):
                            return "dict"
                        if isinstance(default, ast.Set) or (
                            isinstance(default, ast.Call)
                            and isinstance(default.func, ast.Name)
                            and default.func.id == "set"
                        ):
                            return "set"
    return "list"


def _fix_off_by_one(source: str) -> str:
    source = re.sub(r"range\(\s*len\(([^)]+)\)\s*\+\s*1\s*\)", r"range(len(\1))", source)
    return source


def _add_empty_guards(source: str) -> str:
    tree = _parse_ast(source)
    if tree is None:
        return source
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    functions.sort(key=lambda node: node.lineno, reverse=True)
    for node in functions:
        body_source = ast.get_source_segment(source, node) or ""
        arg_names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        guard_names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div):
                if isinstance(child.right, ast.Call) and isinstance(child.right.func, ast.Name) and child.right.func.id == "len":
                    if child.right.args:
                        target = child.right.args[0]
                        if isinstance(target, ast.Name):
                            guard_names.add(target.id)
                elif isinstance(child.right, ast.Name):
                    guard_names.add(child.right.id)
            if isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Constant) and child.slice.value == 0:
                if isinstance(child.value, ast.Name):
                    guard_names.add(child.value.id)
        guard_names &= arg_names
        for name in guard_names:
            # Avoid duplicate guards and skip self/cls-like values.
            if name in {"self", "cls"} or re.search(rf"if not {re.escape(name)}\b", body_source):
                continue
            if name.isidentifier() and name.islower():
                source = _insert_at_body(source, node, _guard_for_empty(name))
        # Re-parse for subsequent insertions.
        tree = _parse_ast(source) or tree
        functions_next = [
            item
            for item in ast.walk(tree)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        node_map = {item.name: item for item in functions_next}
        node = node_map.get(node.name, node)
    return source


def _fix_calc_dispatch(source: str) -> str:
    """Replace the tutorial if/elif calculator with a clean dispatch version."""
    if not re.search(r"def\s+calc\s*\(", source):
        return source
    if not re.search(r"op\s*==\s*['\"]div['\"]", source):
        return source
    tree = _parse_ast(source)
    if tree is None:
        return source
    node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "calc"), None)
    if node is None:
        return source
    args = [arg.arg for arg in node.args.args]
    if len(args) != 3:
        return source
    replacement = '''def calc(left: float, right: float, operator: str) -> float:
    """Evaluate a simple arithmetic operation safely.

    Raises:
        ValueError: If the operator is not supported.
        ZeroDivisionError: If the operator is division and right is zero.
    """
    operations = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
    }
    try:
        operation = operations[operator]
    except KeyError as exc:
        raise ValueError(f"Unsupported operation: {operator}") from exc
    if operator == "div" and right == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return operation(left, right)'''
    lines = source.splitlines()
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "\n".join(lines[:start] + replacement.splitlines() + lines[end:])


def _fix_sql(source: str) -> str:
    if "sqlite3" not in source or not re.search(r"SELECT\s+.+FROM", source, re.IGNORECASE):
        return source
    tree = _parse_ast(source)
    if tree is None:
        return source
    function = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            segment = ast.get_source_segment(source, node) or ""
            if "execute" in segment and "SELECT" in segment.upper():
                function = node
                break
    if function is None or not function.args.args:
        return source
    arg_names = [arg.arg for arg in function.args.args]
    lookup_arg = arg_names[0]
    table_match = re.search(r"FROM\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.IGNORECASE)
    table = table_match.group(1) if table_match else "users"
    replacement = f'''def {function.name}({lookup_arg}: str) -> list[tuple]:
    """Look up rows by name with a parameterized SQL query."""
    query = "SELECT * FROM {table} WHERE name = ?"
    connection = sqlite3.connect("app.db")
    try:
        cursor = connection.cursor()
        return cursor.execute(query, ({lookup_arg},)).fetchall()
    finally:
        connection.close()'''
    lines = source.splitlines()
    start = function.lineno - 1
    end = function.end_lineno or function.lineno
    source = "\n".join(lines[:start] + replacement.splitlines() + lines[end:])
    if "import sqlite3" not in source:
        source = "import sqlite3\n\n\n" + source
    return source


def _add_docstrings(source: str) -> str:
    tree = _parse_ast(source)
    if tree is None:
        return source
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("__")
        and not ast.get_docstring(node)
    ]
    functions.sort(key=lambda node: node.lineno, reverse=True)
    for node in functions:
        doc = _slug(node.name) + "."
        snippet = f'"""{doc}"""'
        source = _insert_at_body(source, node, snippet)
    return source


def _make_fix(source: str) -> str:
    code = source.strip("\n")
    for transform in (
        _fix_mutable_defaults,
        _fix_off_by_one,
        _fix_calc_dispatch,
        _fix_sql,
        _add_empty_guards,
        _add_docstrings,
    ):
        try:
            code = transform(code)
        except Exception:
            # A demo fix must never make the application itself fail.  Leave the
            # last valid transformation in place if an edge case is unsupported.
            continue
    # Add a module docstring only when the code is valid and has no docstring.
    tree = _parse_ast(code)
    if tree is not None and ast.get_docstring(tree) is None:
        code = '"""Reviewed and corrected by the Code Review Squad."""\n\n' + code
    return code.strip() + "\n"


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

def _test_code_for(source: str) -> tuple[str, str]:
    tree = _parse_ast(source)
    if tree is None:
        return (
            'def test_submission_imports():\n    assert False, "submission has a syntax error"\n',
            "The submission cannot be imported until its syntax error is fixed.",
        )
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    }
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    lines: list[str] = [
        '"""Generated pytest suite: normal paths + boundary/edge cases."""',
        "import pytest",
        "",
    ]
    if "sqlite3" in source:
        lines.extend(["import sqlite3", ""])
    imports = [name for name in sorted(function_names | class_names)]
    if imports:
        lines.append(f"from solution import {', '.join(imports)}")
    else:
        lines.append("import solution")
    lines.extend(["", ""])
    tests_added = 0

    def add(name: str, body: list[str]) -> None:
        nonlocal tests_added
        lines.extend([f"def {name}():", *[f"    {row}" for row in body], "", ""])
        tests_added += 1

    if "calculate_average" in function_names:
        add(
            "test_calculate_average_normal_path",
            ["assert calculate_average([1, 2, 3, 4]) == pytest.approx(2.5)"],
        )
        add(
            "test_calculate_average_empty_input_is_rejected",
            [
                "with pytest.raises(ValueError):",
                "    calculate_average([])",
            ],
        )
        add(
            "test_calculate_average_single_value",
            ["assert calculate_average([7]) == pytest.approx(7.0)"],
        )
    if "find_max" in function_names:
        add(
            "test_find_max_normal_and_negative_values",
            [
                "assert find_max([3, 1, 7, 2]) == 7",
                "assert find_max([-9, -2, -5]) == -2",
            ],
        )
        add(
            "test_find_max_empty_input_is_rejected",
            [
                "with pytest.raises(ValueError):",
                "    find_max([])",
            ],
        )
        add(
            "test_find_max_single_value",
            ["assert find_max([42]) == 42"],
        )
    if "add_item" in function_names:
        add(
            "test_add_item_default_lists_are_independent",
            [
                "first = add_item('a')",
                "second = add_item('b')",
                "assert first == ['a']",
                "assert second == ['b']",
            ],
        )
        add(
            "test_add_item_respects_a_supplied_list",
            [
                "existing = ['x']",
                "result = add_item('y', existing)",
                "assert result == ['x', 'y']",
                "assert existing is result",
            ],
        )
    if "User" in class_names:
        add(
            "test_user_default_roles_are_not_shared",
            [
                "alice = User('Alice')",
                "bob = User('Bob')",
                "alice.roles.append('admin')",
                "assert bob.roles == []",
            ],
        )
    if "get_user" in function_names:
        add(
            "test_get_user_returns_matching_row",
            [
                "connection = sqlite3.connect('app.db')",
                "try:",
                "    connection.execute('CREATE TABLE users (name TEXT, age INTEGER)')",
                "    connection.execute('INSERT INTO users VALUES (?, ?)', ('alice', 30))",
                "    connection.commit()",
                "finally:",
                "    connection.close()",
                "assert get_user('alice') == [('alice', 30)]",
            ],
        )
        add(
            "test_get_user_treats_sql_injection_as_data",
            [
                "assert get_user(\"' OR '1'='1\") == []",
            ],
        )
    if "calc" in function_names:
        add(
            "test_calc_normal_operations",
            [
                "assert calc(2, 3, 'add') == 5",
                "assert calc(7, 2, 'sub') == 5",
                "assert calc(3, 4, 'mul') == 12",
                "assert calc(8, 2, 'div') == 4",
            ],
        )
        add(
            "test_calc_rejects_division_by_zero",
            [
                "with pytest.raises(ZeroDivisionError):",
                "    calc(1, 0, 'div')",
            ],
        )
        add(
            "test_calc_rejects_unknown_operator",
            [
                "with pytest.raises(ValueError):",
                "    calc(1, 2, 'pow')",
            ],
        )
    if "add" in function_names:
        add(
            "test_add_normal_and_boundary_values",
            [
                "assert add(2, 3) == 5",
                "assert add(0, 0) == 0",
                "assert add(-2, 2) == 0",
            ],
        )
    if "safe_divide" in function_names:
        add(
            "test_safe_divide_normal_path",
            ["assert safe_divide(10, 2) == 5"],
        )
        add(
            "test_safe_divide_zero_divisor",
            [
                "with pytest.raises((ValueError, ZeroDivisionError)):",
                "    safe_divide(1, 0)",
            ],
        )
    if tests_added == 0:
        public = ", ".join(sorted(function_names)) or "no public functions"
        add(
            "test_module_imports_and_exposes_public_api",
            [f"# Public API detected: {public}", "assert True"],
        )
    rationale = (
        "Targeted normal-path, empty-input, boundary, and state-isolation tests "
        "were generated from the submitted functions."
    )
    return "\n".join(lines).rstrip() + "\n", rationale


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _intent_response(prompt: str) -> str:
    request = _request_from_prompt(prompt)
    text = request.lower()
    if any(term in text for term in ("full squad", "full review", "complete review", "review everything", "all agents")):
        intent, confidence, rationale = "full_review", 0.98, "The request explicitly asks for the complete squad workflow."
    elif "debug" in text:
        intent, confidence, rationale = "debugging", 0.96, "The request explicitly asks for debugging and a corrected version."
    elif any(term in text for term in ("test", "pytest", "unit test", "generate tests")):
        intent, confidence, rationale = "testing", 0.95, "The request focuses on generating and running tests."
    elif any(term in text for term in ("explain", "what does", "walk through", "how does", "understand")):
        intent, confidence, rationale = "code_explanation", 0.94, "The user wants an explanation rather than edits."
    elif any(term in text for term in ("style", "lint", "pep 8", "pep8", "format", "readability", "naming")):
        intent, confidence, rationale = "style_review", 0.93, "The request focuses on conventions and readability."
    elif any(term in text for term in ("bug", "fix", "error", "fails", "crash", "wrong", "root cause")):
        intent, confidence, rationale = "debugging", 0.94, "The request asks to find and correct a defect."
    elif any(term in text for term in ("review", "check")):
        intent, confidence, rationale = "full_review", 0.82, "A broad review request maps to the full squad."
    else:
        intent, confidence, rationale = "full_review", 0.55, "The request is broad, so the safest complete route is full review."
    return json.dumps({"intent": intent, "confidence": confidence, "rationale": rationale})


def _rag_response(prompt: str) -> str:
    request = _request_from_prompt(prompt)
    code = _code_from_prompt(prompt)
    lower = request.lower()
    nonempty = [line for line in code.splitlines() if line.strip()]
    trivial = len(nonempty) <= 8 and not re.search(
        r"(range\([^)]*\+\s*1|=\s*\[\]|=\s*\{\}|SELECT|sqlite3|except\s*:|/|eval\()",
        code,
    )
    wants_explanation = any(word in lower for word in ("explain", "what does", "walk through", "how does"))
    if wants_explanation and trivial:
        return json.dumps(
            {
                "should_retrieve": False,
                "query": "",
                "reason": "This is a short explanation task on a trivial, low-risk snippet. The model can answer directly, so the RAG gate skips an unnecessary vector lookup.",
            }
        )
    if any(word in lower for word in ("full squad", "full review", "complete review", "review everything", "all agents")):
        query = "Python mutable default arguments off-by-one zero division PEP 8 naming docstrings pytest edge cases"
    elif any(word in lower for word in ("test", "pytest", "unit test")):
        query = "pytest edge cases boundary tests parametrization assertions"
    elif any(word in lower for word in ("style", "lint", "pep", "format", "readability")):
        query = "PEP 8 naming docstrings line length readability clean code"
    elif any(word in lower for word in ("debug", "bug", "fix", "error")):
        query = "Python debugging mutable default arguments off-by-one zero division resource leaks exception handling"
    elif any(word in lower for word in ("explain", "what does", "walk through")):
        query = "Python code explanation edge cases best practices"
    else:
        query = "Python code review bug patterns PEP 8 pytest edge cases debugging examples"
    return json.dumps(
        {
            "should_retrieve": True,
            "query": query,
            "reason": "The task involves code review decisions where project guidelines and known Python failure patterns improve the analysis, so curated context should be retrieved.",
        }
    )


def _bug_report(prompt: str) -> str:
    code = _code_from_prompt(prompt)
    issues = _scan_bugs(code)
    lines = code.splitlines()
    trace = [
        "## Chain-of-Thought Execution Trace",
        "1. **Parse & structure** — parsed the submitted source, listed functions/classes, and inspected the AST evidence supplied by the static-analysis tool.",
        "2. **Control-flow walkthrough** — traced each branch and return path, including normal, empty, and boundary inputs.",
        "3. **Data-flow review** — checked mutable state, arithmetic domains, sequence bounds, exception paths, external I/O, and SQL construction.",
        "4. **Edge-case matrix** — empty input, one element, zero divisor, repeated calls, malicious input, and failure paths.",
        "",
    ]
    if not issues:
        lines_out = trace + [
            "## Findings",
            "No high-confidence logical defects were found in this snippet.",
            "",
            "### Residual risks",
            "- Edge cases are only as complete as the visible code; inputs are assumed to be of the expected type.",
        ]
        return "\n".join(lines_out)
    trace.append(f"5. **Conclusion** — {len(issues)} issue(s) survived the walkthrough and are reported below.")
    trace.extend(["", "## Findings"])
    for index, issue in enumerate(issues, start=1):
        source_line = ""
        if 0 < issue["line"] <= len(lines):
            source_line = lines[issue["line"] - 1].strip()
        trace.extend(
            [
                "",
                f"### BUG-{index} — {issue['title']}",
                f"- **Severity:** {issue['severity']}",
                f"- **Location:** line {issue['line']}" + (f" (`{source_line}`)" if source_line else ""),
                f"- **Why it fails:** {issue['why']}",
                f"- **Trigger:** {issue['trigger']}",
                f"- **Recommended fix:** {issue['fix']}",
            ]
        )
    trace.extend(
        [
            "",
            "## Summary",
            f"The walkthrough found {len(issues)} issue(s), ordered by risk. The fixer must resolve every finding without weakening existing behavior.",
        ]
    )
    return "\n".join(trace)


def _style_report(prompt: str) -> str:
    code = _code_from_prompt(prompt)
    issues = _style_issues(code)
    tree = _parse_ast(code)
    functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))) if tree else 0
    max_line = max((len(line) for line in code.splitlines()), default=0)
    score = max(35, 100 - 7 * len(issues))
    lines = [
        "## Style / Lint Report",
        "",
        "### Few-shot-informed assessment",
        "The examples anchor the review on PEP 8 naming, docstrings, wrapping, and readable dispatch structure.",
        "",
        f"- Functions reviewed: **{functions}**",
        f"- Longest line: **{max_line}** characters (PEP 8 guidance: 88 for this project)",
        f"- Readability score: **{score}/100**",
        "",
        "### Findings",
    ]
    if not issues:
        lines.append("No style findings. The snippet follows the project's PEP 8 conventions.")
    for index, issue in enumerate(issues, start=1):
        lines.extend(
            [
                "",
                f"### STYLE-{index} — {issue['rule']}",
                f"- **Line:** {issue['line']}",
                f"- **Finding:** {issue['message']}",
                f"- **Fix:** {issue['fix']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Priority",
            "Apply naming/docstring fixes first, then formatting. Preserve the public API unless a rename is coordinated with callers.",
        ]
    )
    return "\n".join(lines)


def _fixer_response(prompt: str) -> str:
    code = _code_from_prompt(prompt, "original_code") or _code_from_prompt(prompt, "code")
    fixed = _make_fix(code)
    return f"""### Fix rationale
The rewrite addresses the reported defects and readability findings: empty-input guards and sequence bounds are explicit, mutable defaults are isolated per call, and database/SQL handling is safe. Public function names are preserved.

### Reflection checklist
- Every reported bug was mapped to a concrete code change.
- Existing normal-path behavior is preserved.
- New guards raise descriptive exceptions instead of silently returning wrong data.
- The result is valid Python and ready for isolated tests.

===FIXED CODE===
```python
{fixed.rstrip()}
```
===END FIXED CODE===
"""


def _reflection_response(prompt: str) -> str:
    proposed = _code_from_prompt(prompt, "proposed_code") or _code_from_prompt(prompt, "code")
    bug_report = _section(prompt, "bug_report")
    style_report = _section(prompt, "style_report")
    test_failure = _section(prompt, "test_failure")
    fixed = _make_fix(proposed) if proposed else proposed
    findings = bug_report.count("### BUG-") + style_report.count("### STYLE-")
    if test_failure:
        verdict = "The sandbox failure was used to harden the edge-case behavior, and the corrected function set is returned for one final test run."
    else:
        verdict = "The proposed fix maps to every reported finding, preserves normal behavior, and introduces no new public API or syntax issue."
    return f"""## Reflection Verdict
{verdict}

## Self-Critique
- **Bug-Finder findings addressed:** {bug_report.count('### BUG-') or 'none reported'}
- **Style/Lint findings addressed:** {style_report.count('### STYLE-') or 'none reported'}
- **Total findings reviewed:** {findings}
- **New-risk check:** no dangerous imports, no mutable defaults, and all empty-input paths fail descriptively.
- **Sandbox feedback:** {'provided and incorporated' if test_failure else 'not needed before test generation'}

## Final Status
PASS — the revised code is ready for isolated pytest execution.

===FINAL CODE===
```python
{fixed.rstrip()}
```
===END FINAL CODE===
"""


def _test_response(prompt: str) -> str:
    source = (
        _code_from_prompt(prompt, "fixed_code")
        or _code_from_prompt(prompt, "code")
        or _code_from_prompt(prompt, "proposed_code")
    )
    test_code, rationale = _test_code_for(source)
    return json.dumps({"test_code": test_code, "rationale": rationale})


def _explain_response(prompt: str) -> str:
    code = _code_from_prompt(prompt)
    tree = _parse_ast(code)
    if tree is None:
        return "## Explanation\n\nThe snippet does not parse as valid Python, so its behavior cannot be traced until the syntax error is fixed."
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    lines = [
        "## What this code does",
        "The module defines "
        + (", ".join(f"`{node.name}`" for node in functions) if functions else "no top-level functions")
        + (" and class " + ", ".join(f"`{node.name}`" for node in classes) if classes else "")
        + ".",
        "",
        "## Step-by-step walkthrough",
    ]
    if not functions and not classes:
        lines.append("The module contains only top-level statements; they execute in source order when imported.")
    for node in functions:
        args = [arg.arg for arg in node.args.args]
        doc = ast.get_docstring(node) or "No docstring is provided."
        lines.extend(
            [
                f"### `{node.name}({', '.join(args)})`",
                f"- **Documented purpose:** {doc}",
                f"- **Control flow:** {len(node.body)} top-level statement(s), with branches/loops evaluated in order.",
                f"- **Returns:** {_return_summary(node)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Edge cases and risks",
            "- Empty or `None` inputs should be checked before indexing or dividing.",
            "- Mutable defaults, if present, persist across calls; prefer `None` plus internal initialization.",
            "- External I/O should be closed on every path and SQL values should be parameterized.",
            "",
            "## Bottom line",
            "Read the functions from their inputs to their return values. The names and docstrings describe intent; the conditions and loops describe the exact path taken.",
        ]
    )
    return "\n".join(lines)


def _return_summary(node: ast.FunctionDef) -> str:
    values = [
        ast.unparse(child.value) if child.value is not None else "None"
        for child in ast.walk(node)
        if isinstance(child, ast.Return)
    ]
    if not values:
        return "no explicit return (returns None)"
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return ", ".join(f"`{value}`" for value in unique[:4])


def _generate(system: str, user: str) -> str:
    role_line = " ".join(system.strip().splitlines()[:1]).lower()
    lower_user = user.lower()
    if "intent router" in role_line:
        return _intent_response(user)
    if "agentic rag gate" in role_line:
        return _rag_response(user)
    if "fixer agent performing a reflection pass" in role_line:
        return _reflection_response(user)
    if "style/lint agent" in role_line:
        return _style_report(user)
    if "bug-finder agent" in role_line:
        return _bug_report(user)
    if "fixer agent" in role_line:
        if "reflection pass" in lower_user or "proposed_code" in user:
            return _reflection_response(user)
        return _fixer_response(user)
    if "test-writer agent" in role_line:
        return _test_response(user)
    if "explainer agent" in role_line:
        return _explain_response(user)
    return "Offline demo model: no matching task signature was found in the prompt."


class DemoChatModel(BaseChatModel):
    """A deterministic BaseChatModel for fully offline GUI evaluation."""

    model_name: str = "code-review-squad-demo"

    @property
    def _llm_type(self) -> str:
        return "code-review-squad-demo"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "deterministic": True}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_parts = [
            _content_to_text(message.content)
            for message in messages
            if message.type in {"system", "developer"}
        ]
        user = _last_human(messages)
        system = "\n\n".join(system_parts)
        response_text = _generate(system, user)
        if stop:
            for marker in stop:
                if marker and marker in response_text:
                    response_text = response_text.split(marker, 1)[0]
        message = AIMessage(
            content=response_text,
            response_metadata={"demo_model": True, "model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


__all__ = ["DemoChatModel"]
