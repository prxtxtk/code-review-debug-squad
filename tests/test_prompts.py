"""Static tests for the required prompt-engineering techniques."""

from __future__ import annotations

from prompts import (
    BUG_FINDER_PROMPT,
    BUG_FINDER_SYSTEM,
    FIXER_REFLECTION_PROMPT,
    FIXER_REFLECTION_SYSTEM,
    INTENT_ROUTER_PROMPT,
    INTENT_ROUTER_SYSTEM,
    STYLE_AGENT_PROMPT,
    STYLE_AGENT_SYSTEM,
)


def test_intent_router_has_system_user_and_few_shots() -> None:
    messages = INTENT_ROUTER_PROMPT.format_messages(request="debug", code="x = 1")
    assert messages[0].type == "system"
    assert messages[-1].type == "human"
    assert "Few-shot examples" in INTENT_ROUTER_SYSTEM
    assert "system" in messages[0].type


def test_style_agent_has_few_shot_examples() -> None:
    messages = STYLE_AGENT_PROMPT.format_messages(
        request="style",
        code="def Calc(x,y): return x+y",
        lint_report="lint",
        retrieved_context="context",
    )
    assert messages[0].type == "system"
    assert messages[-1].type == "human"
    assert "Few-shot examples" in STYLE_AGENT_SYSTEM


def test_bug_finder_explicitly_requires_chain_of_thought() -> None:
    assert "Chain of Thought" in BUG_FINDER_SYSTEM
    assert "step-by-step execution trace" in BUG_FINDER_SYSTEM
    messages = BUG_FINDER_PROMPT.format_messages(
        request="debug",
        code="x = 1",
        ast_report="ast",
        retrieved_context="context",
    )
    assert messages[0].type == "system"
    assert messages[-1].type == "human"


def test_fixer_has_explicit_reflection_pass() -> None:
    assert "REFLECTION PASS" in FIXER_REFLECTION_SYSTEM
    assert "Critique" in FIXER_REFLECTION_SYSTEM
    messages = FIXER_REFLECTION_PROMPT.format_messages(
        request="fix",
        bug_report="bug",
        style_report="style",
        proposed_code="x = 1",
        test_failure="",
    )
    assert messages[0].type == "system"
    assert messages[-1].type == "human"
    assert "test_failure" in messages[-1].content
