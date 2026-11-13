# Pytest Guide: Edge Cases, Boundaries, and Assertions

Source: Code Review Squad curated testing handbook.

## Test the contract, not the implementation

Each test should state an observable behaviour: an input, a call, and an expected
result or exception. Avoid asserting on private attributes unless they are part
of the documented contract. A useful naming pattern is
`test_<unit>_<scenario>_<expected_result>`.

```python
def test_calculate_average_empty_values_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_average([])
```

## Build an edge-case matrix

For every input, ask:

| Dimension | Typical cases |
| --- | --- |
| Emptiness | `[]`, `""`, `{}`, `None` if allowed |
| Size | one element, two elements, many elements |
| Boundary | first, last, exactly at a limit, one past the limit |
| Numeric | `0`, `1`, `-1`, very large, NaN/infinity where relevant |
| State | first call, repeated call, independent instances |
| Errors | expected exception type and message |
| External I/O | missing file, empty table, malformed row, cleanup |

Cover the normal path first, then the boundaries that are most likely to fail.
Do not write ten variations of the same happy path while ignoring empty input.

## Boundary-value analysis

For an inclusive range `[low, high]`, test `low - 1`, `low`, a middle value,
`high`, and `high + 1` when those values are legal to represent. For a sequence,
test length 0, 1, and the exact cutoff used by the implementation. Off-by-one
bugs live at these seams.

## Parametrisation

Use `pytest.mark.parametrize` for data-driven cases. Keep the case IDs readable
so a failure points directly at the scenario.

```python
import pytest

@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([1], 1.0, id="single"),
        pytest.param([1, 2, 3], 2.0, id="odd-count"),
        pytest.param([2, 4], 3.0, id="even-count"),
    ],
)
def test_calculate_average_values(values, expected):
    assert calculate_average(values) == pytest.approx(expected)
```

## Assertions

- Use plain `assert expression` with pytest's introspection.
- Use `pytest.approx` for floating-point comparisons.
- Assert the specific exception with `pytest.raises(ValueError)` and a message
  pattern where the contract specifies one.
- Assert that an exception is not raised when that is the behaviour under test.
- Check both result and relevant side effects; a function can return the right
  value yet mutate shared state incorrectly.
- Prefer one logical behaviour per test. Multiple independent assertions are
  fine when they describe the same behaviour.

## Fixtures and temporary state

Use `tmp_path` for filesystem tests and function-scoped fixtures for isolated
state. Never write to a developer's home directory or a fixed absolute path.
Close resources in fixture teardown. Tests must pass in any order and in parallel
where practical.

```python
def test_report_is_written(tmp_path):
    target = tmp_path / "report.txt"
    write_report(target, ["ok"])
    assert target.read_text(encoding="utf-8") == "ok\n"
```

## Testing exceptions and invalid input

Test the documented failure mode, not merely "some error happened". Avoid broad
`except Exception` in tests. Ensure the invalid input would pass a naive happy
path, so the test actually exercises validation.

## State-isolation tests for mutable defaults

A regression test for a mutable default should prove that two calls do not share
state:

```python
def test_add_item_default_lists_are_independent():
    first = add_item("a")
    second = add_item("b")
    assert first == ["a"]
    assert second == ["b"]
```

Test supplied mutable arguments separately: the function may intentionally
mutate a caller-owned list, and that contract should be explicit.

## Deterministic tests

- Freeze time, random seeds, and external services.
- Do not rely on network access in unit tests.
- Do not assert on dictionary ordering unless insertion order is part of the
  contract.
- Keep output messages stable or match only the important fragment.
- Use relative temporary paths and clean up any database or file created.

## Test suite quality checklist

1. Is there at least one test for every public function and documented branch?
2. Are empty, one-element, zero, negative, and boundary inputs covered?
3. Are error types and messages asserted, not just "it raises"?
4. Does any test leak state between cases?
5. Can the suite run offline in an isolated temporary directory?
6. Would a real defect make at least one test fail for the right reason?
