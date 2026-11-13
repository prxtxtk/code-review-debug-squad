# Common Python Bug Patterns and Defensive Fixes

Source: Code Review Squad curated defect catalogue.

## Mutable default arguments

Default values are evaluated once, when the function object is created. A list,
dict, or set default is therefore shared by every call.

```python
# Bug: all calls share one list
def add_item(item, items=[]):
    items.append(item)
    return items

# Fix: create a fresh container per call
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

The same problem applies to class instance defaults such as
`def __init__(self, roles=[])`. Use `None` and initialise inside the constructor.
Never mutate a mutable argument unless the function contract explicitly says the
argument is owned and modified in place.

## Off-by-one and sequence bounds

`range(len(values) + 1)` visits one index past the final element and raises
`IndexError`. Be explicit about inclusive versus exclusive bounds.

```python
# Bug: IndexError after the final element
for index in range(len(values) + 1):
    print(values[index])

# Fix: exclusive bound
for value in values:
    print(value)
```

Related patterns:

- `values[len(values)]` is always out of bounds.
- `values[0]` on an empty sequence raises `IndexError`; guard first.
- Slicing is forgiving (`values[:n]`), but subsequent arithmetic on a short
  slice may still be wrong.
- A loop that starts at `0` but increments a count before the first iteration
  often produces an off-by-one total.

## Division by zero

Every division potentially has a zero denominator. This includes
`total / len(items)` when `items` is empty.

```python
def calculate_average(values):
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
```

Document whether empty input should raise, return `0.0`, or return `None`.
Never silently return a wrong value to avoid the exception.

## None handling and truthiness

- `if value == None` should be `if value is None`.
- `None.get(...)`, `None[0]`, and iterating `None` raise `AttributeError` or
  `TypeError`; validate optional values at the API boundary.
- `if value:` is not equivalent to `if value is not None:`; numeric `0`, empty
  strings, and empty containers are falsy but valid data.
- Use `dict.get(key, default)` and `collections.defaultdict` deliberately.
- Avoid catching `AttributeError` to hide a missing object. Check and report the
  real precondition.

## Resource leaks

Files, sockets, database connections, and locks must be closed on every path,
including exceptions.

```python
import sqlite3

def get_user(username):
    connection = sqlite3.connect("app.db")
    try:
        cursor = connection.cursor()
        return cursor.execute(
            "SELECT * FROM users WHERE name = ?", (username,)
        ).fetchall()
    finally:
        connection.close()
```

Use a context manager for files. Note that `sqlite3.Connection.__enter__` is a
transaction context, not a close-connection context; use `try/finally` or
`contextlib.closing` when closure must be guaranteed. Cursor objects also
benefit from context management when long-lived.

## SQL injection and query construction

Concatenating or formatting untrusted values into SQL creates an injection
vulnerability. Parameterization sends values separately from SQL syntax.

```python
# Vulnerable
query = "SELECT * FROM users WHERE name = '" + username + "'"

# Safe
cursor.execute("SELECT * FROM users WHERE name = ?", (username,))
```

Never build table or column names from raw input; allow-list identifiers if
dynamic identifiers are unavoidable.

## Exception anti-patterns

- A bare `except:` catches `KeyboardInterrupt` and `SystemExit` as well as
  programming errors. Catch the narrowest exception that you can handle.
- `except Exception: pass` hides failures and makes data corruption silent.
- Catching an exception and returning `None` changes the API contract without
  telling callers.
- Re-raising with `raise` preserves the traceback; `raise exc` resets it.
- Use `raise ... from exc` when translating a low-level error into a domain
  error, and include the original exception as the cause.
- Do not use exceptions for ordinary control flow when a simple condition is
  clearer.

## Unbounded loops and timeouts

A loop whose bound comes from untrusted input can hang a service. Validate a
maximum, add a deadline, or enforce a timeout at the execution boundary. Tests
should cover the terminating and error paths, not only the happy path.

## File paths and untrusted input

Reject absolute paths and `..` traversal when accepting uploads. Resolve paths
against a fixed base directory and verify the final resolved path remains inside
it. Do not use `eval`, `exec`, or dynamic imports on user-controlled strings.

## Concurrency pitfalls

Shared mutable state accessed by threads needs a lock or immutable hand-off.
Mutable default arguments are a single-shot example of the same aliasing bug.
Document whether a function is thread-safe; do not assume that a Python context
switch cannot occur between a check and an action.

## Review heuristics

1. Which inputs are allowed to be empty, `None`, zero, or negative?
2. Which resources must be released on every path?
3. Where can an index exceed the sequence length?
4. Is all external data separated from executable syntax (SQL, shell, HTML)?
5. Are errors specific, observable, and handled at the right layer?
