# Debugging Examples: Before, After, and Why

Source: Code Review Squad curated before/after repair examples.

## Example 1 — Empty sequence causes ZeroDivisionError

Before:

```python
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)
```

After:

```python
def calculate_average(numbers):
    """Return the arithmetic mean of a non-empty sequence."""
    if not numbers:
        raise ValueError("numbers must not be empty")
    return sum(numbers) / len(numbers)
```

Why: `len(numbers)` is zero for an empty list, and Python raises
`ZeroDivisionError` before returning a value. The guard makes the contract
explicit. `sum()` also removes index bookkeeping and the risk of an off-by-one
loop. The test should call `calculate_average([])` and assert `ValueError`.

## Example 2 — Off-by-one loop reads past the end

Before:

```python
def find_max(values):
    max_value = values[0]
    for i in range(len(values) + 1):
        if values[i] > max_value:
            max_value = values[i]
    return max_value
```

After:

```python
def find_max(values):
    """Return the largest value in a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    max_value = values[0]
    for value in values[1:]:
        if value > max_value:
            max_value = value
    return max_value
```

Why: `range(len(values) + 1)` visits index `len(values)`, which is one past the
last valid index. The empty-input guard also prevents `values[0]` from raising
`IndexError`. A boundary test with one element and a test with an empty list
cover both failure modes.

## Example 3 — Shared mutable default leaks state between calls

Before:

```python
def add_item(item, items=[]):
    items.append(item)
    return items
```

After:

```python
def add_item(item, items=None):
    """Append an item to a list, creating a fresh list by default."""
    if items is None:
        items = []
    items.append(item)
    return items
```

Why: The list literal is evaluated once, so every call without an explicit list
appends to the same object. A test that calls the function twice and compares
the first result before the second call exposes the bug. Do not "fix" this by
clearing the default; callers may have intentionally passed a list.

## Example 4 — Constructor aliases a shared list

Before:

```python
class User:
    def __init__(self, name, roles=[]):
        self.name = name
        self.roles = roles
```

After:

```python
class User:
    def __init__(self, name, roles=None):
        if roles is None:
            roles = []
        self.name = name
        self.roles = roles
```

Why: Two users created without roles would share the same list. Mutating one
user's roles would change the other's. The test creates two users, appends to
one, and asserts the other remains empty. Pass `roles=list(roles)` if callers
should not retain a reference to their input list either.

## Example 5 — SQL injection and leaked connection

Before:

```python
import sqlite3

def get_user(username):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return cursor.execute(query).fetchall()
```

After:

```python
import sqlite3

def get_user(username: str) -> list[tuple]:
    """Look up a user by name using a parameterized query."""
    conn = sqlite3.connect("app.db")
    try:
        cursor = conn.cursor()
        return cursor.execute(
            "SELECT * FROM users WHERE name = ?", (username,)
        ).fetchall()
    finally:
        conn.close()
```

Why: String concatenation lets a value such as `' OR '1'='1` change the query's
logic. Parameterized execution sends the value as data. The `finally` block
closes the connection on both success and exception paths. The context manager
alone is insufficient for sqlite3 because it manages the transaction, not the
connection lifetime. A test should insert a fixture row and also query with an
injection-shaped string, expecting no extra rows.

## Example 6 — Messy conditional calculator

Before:

```python
def calc(x,y,op):
 if op=='add': return x+y
 elif op=='sub': return x-y
 elif op=='mul': return x*y
 elif op=='div': return x/y
 else: return None
```

After:

```python
def calc(left: float, right: float, operator: str) -> float:
    """Evaluate a simple arithmetic operation safely."""
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
    return operation(left, right)
```

Why: The dispatch dictionary makes the supported operations obvious and removes
the repeated comparisons. Unknown operators now raise a specific exception
instead of silently returning `None`, and division by zero has an explicit,
testable contract. The public function name is kept to avoid breaking callers.

## General debugging workflow

1. Reproduce with the smallest input that fails; record the exact exception and
   traceback.
2. Read the failing line and its inputs; distinguish a symptom from the root
   cause.
3. Check boundaries first: empty, one item, zero, and one past the end.
4. Inspect state that survives across calls (defaults, caches, globals, class
   attributes).
5. Write a failing regression test before changing the implementation.
6. Apply the smallest fix that addresses the cause, then rerun the full suite.
7. Review the patch for new resource, security, and readability problems.
