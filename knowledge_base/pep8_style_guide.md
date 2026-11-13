# Python Style and Clean Code Guidelines

Source: Code Review Squad curated PEP 8 / PEP 257 handbook.

## Naming conventions

- `snake_case` for functions, methods, local variables, and module names.
- `PascalCase` for classes and exception types.
- `UPPER_SNAKE_CASE` for module-level constants.
- A leading underscore marks a non-public helper (`_normalise_username`).
- Avoid single-character names except conventional loop indices (`i`, `j`) or
  coordinates (`x`, `y`) in a tightly scoped mathematical expression.
- Names should describe intent, not implementation. Prefer `active_users` over
  `user_list_2`; prefer `calculate_average` over `calc`.
- Do not shadow built-ins such as `list`, `dict`, `id`, `input`, `type`, or
  `format`.

Example:

```python
# Prefer
def calculate_average(values: list[float]) -> float:
    ...

# Avoid
def Calc(v):
    ...
```

## Docstrings (PEP 257)

Every public module, class, and function should have a docstring. The first line
is a short imperative summary ending with a period. Document non-obvious
arguments, return values, raised exceptions, side effects, and complex edge
cases. Keep the docstring consistent with the implementation; an outdated
docstring is worse than none.

```python
def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers and reject a zero denominator.

    Args:
        numerator: Value being divided.
        denominator: Non-zero divisor.

    Raises:
        ZeroDivisionError: If denominator is zero.
    """
    if denominator == 0:
        raise ZeroDivisionError("denominator must not be zero")
    return numerator / denominator
```

## Layout and formatting

- Use four spaces per indentation level. Never mix tabs and spaces.
- Limit lines to 88 characters (project standard), and shorter for prose.
- Surround top-level functions and classes with two blank lines; methods with
  one blank line.
- Put one space after commas and around binary operators. Do not add spaces
  around `=` when passing keyword arguments.
- Import one module per line. Order imports as standard library, third-party,
  then local application imports, with blank lines between groups.
- Avoid wildcard imports (`from module import *`).
- End files with a single newline and remove trailing whitespace.

## Readability and clean code

- Prefer small functions with one responsibility. Extract helpers when a
  function exceeds roughly 30–40 lines or nests more than three levels.
- Replace long `if`/`elif` chains with a dispatch dictionary when each branch
  maps an input to a value or callable.
- Remove dead code, commented-out code, debugging `print()` calls, and unused
  imports before committing.
- Make boundary conditions explicit. A guard clause (`if not values: ...`) is
  clearer than deeply nested error handling.
- Avoid magic numbers; name them as constants and explain unusual thresholds.
- Use `is None` / `is not None` for identity checks, and truthiness (`if items:`)
  when an empty container is genuinely equivalent to absence.
- Prefer f-strings for interpolation and `enumerate()` when both the index and
  value are needed.
- Return consistently typed values. Do not return a number on one path and
  `None` on another unless the API documents that contract.
- Raise specific exceptions with actionable messages instead of returning
  sentinel values that callers may forget to check.

## Constants and configuration

Keep secrets out of source code. Read API keys and deployment URLs from the
environment, validate them at startup, and never log their values. Defaults
belong in one configuration module so tests and production share behaviour.

## Lint workflow

Run Ruff (or Flake8) before review. A practical Ruff selection is:

```text
ruff check --select E,W,F,I,N,D,B --line-length 88 .
```

Treat lint findings as prompts for judgement, not an automatic rewrite: a
suppression is acceptable only with a comment that explains why the rule does
not apply. Do not disable a rule globally to hide one problematic line.

## Review checklist

1. Can a new reader identify the purpose from names and docstrings?
2. Are inputs, outputs, exceptions, and side effects explicit?
3. Does every branch have a clear owner and a test?
4. Are imports ordered and unused dependencies removed?
5. Would a smaller helper make the main path easier to follow?
