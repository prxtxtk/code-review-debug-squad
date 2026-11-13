"""Offline integration tests for all five RunnableBranch routes.

These use the deterministic DemoChatModel and a temporary ChromaDB collection;
no network or API key is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import CodeReviewSquad
from config import AppConfig, build_llm, llm_identity
from demo_llm import DemoChatModel
from sandbox import SandboxResult
from vector_store import HashingEmbeddings, VectorStoreManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"

OFF_BY_ONE_CODE = '''def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)


def find_max(values):
    max_value = values[0]
    for i in range(len(values) + 1):
        if values[i] > max_value:
            max_value = values[i]
    return max_value
'''

TRIVIAL_CODE = '''def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
'''

MESSY_CODE = '''def calc(x,y,op):
 if op=='add': return x+y
 elif op=='sub': return x-y
 elif op=='mul': return x*y
 elif op=='div': return x/y
 else: return None
'''


@pytest.fixture(scope="module")
def squad_context(tmp_path_factory):
    directory = tmp_path_factory.mktemp("squad_chroma")
    store = VectorStoreManager(
        persist_directory=directory / "db",
        collection_name="integration_guidelines",
        knowledge_base_dir=KNOWLEDGE_BASE,
        embeddings=HashingEmbeddings(),
        auto_ingest=True,
    )
    store.initialize()
    config = AppConfig(
        provider="demo",
        model="code-review-squad-demo",
        top_k=3,
        chroma_dir=directory / "db",
        knowledge_base_dir=KNOWLEDGE_BASE,
        collection_name="integration_guidelines",
        sandbox_timeout=5,
    )
    squad = CodeReviewSquad(DemoChatModel(), store, config)
    yield squad, store, config
    store.close()


def test_demo_llm_factory_returns_lcel_compatible_model() -> None:
    config = AppConfig(provider="demo", model="code-review-squad-demo")
    llm = build_llm(config)
    assert isinstance(llm, DemoChatModel)
    assert llm_identity(config)["transport"] == "offline-demo"


def test_full_squad_route_runs_all_agents_parallel_static_and_sandbox(squad_context) -> None:
    squad, _, _ = squad_context
    result = squad.review(
        "Perform a full squad review: find bugs, check style, fix the code, and run tests.",
        OFF_BY_ONE_CODE,
    )
    assert result.errors == []
    assert result.intent.intent == "full_review"
    assert "full_review" in result.trace["route_path"]
    assert result.route_state.get("parallel_reviewers_executed") is True
    assert result.rag.should_retrieve is True
    assert result.rag.hits
    assert "Chain-of-Thought" in result.bug_report
    assert "Style / Lint Report" in result.style_report
    assert "if not numbers" in result.fixed_code
    assert "range(len(values))" in result.fixed_code
    assert result.test_plan is not None
    assert result.sandbox_attempts
    assert result.final_sandbox.tests_passed
    prompt_keys = {prompt["key"] for prompt in result.trace["prompts"]}
    assert {
        "intent_router",
        "rag_gate",
        "bug_finder",
        "style_agent",
        "fixer",
        "fixer_reflection",
        "test_writer",
    } <= prompt_keys
    tool_kinds = {call["kind"] for call in result.trace["tool_calls"]}
    assert "static_analysis" in tool_kinds
    assert "sandbox" in tool_kinds
    assert result.trace["sandbox_retry"]["attempted"] is False


def test_explanation_route_skips_rag_and_tools_for_trivial_code(squad_context) -> None:
    squad, _, _ = squad_context
    result = squad.review("Explain this short add function.", TRIVIAL_CODE)
    assert result.errors == []
    assert result.intent.intent == "code_explanation"
    assert result.rag.should_retrieve is False
    assert result.rag.hits == []
    assert result.trace["tool_calls"] == []
    assert result.explanation
    assert result.fixed_code == ""
    assert result.sandbox_attempts == []
    assert "skipped_tools_reason" in result.trace


def test_debugging_route_uses_bug_finder_and_fixer(squad_context) -> None:
    squad, _, _ = squad_context
    result = squad.review("Find the bug in calculate_average and fix it.", OFF_BY_ONE_CODE)
    assert result.errors == []
    assert result.intent.intent == "debugging"
    assert result.bug_report
    assert not result.style_report
    assert result.fixed_code
    assert result.reflection_report
    assert result.sandbox_attempts == []
    prompt_keys = {prompt["key"] for prompt in result.trace["prompts"]}
    assert {"bug_finder", "fixer_reflection"} <= prompt_keys
    assert "style_agent" not in prompt_keys


def test_style_route_uses_few_shot_style_agent_and_fixer(squad_context) -> None:
    squad, _, _ = squad_context
    result = squad.review("Clean up this code for PEP 8 style and readability.", MESSY_CODE)
    assert result.errors == []
    assert result.intent.intent == "style_review"
    assert result.style_report
    assert not result.bug_report
    assert result.fixed_code
    prompt_keys = {prompt["key"] for prompt in result.trace["prompts"]}
    assert {"style_agent", "fixer_reflection"} <= prompt_keys
    assert "bug_finder" not in prompt_keys


def test_testing_route_generates_and_executes_tests(squad_context) -> None:
    squad, _, _ = squad_context
    result = squad.review("Write pytest tests for normal and edge cases and run them.", TRIVIAL_CODE)
    assert result.errors == []
    assert result.intent.intent == "testing"
    assert result.test_plan is not None
    assert result.sandbox_attempts
    assert result.final_sandbox.tests_passed
    assert result.fixed_code == ""
    prompt_keys = {prompt["key"] for prompt in result.trace["prompts"]}
    assert "test_writer" in prompt_keys
    assert "fixer_reflection" not in prompt_keys


def test_full_review_retries_once_with_sandbox_feedback(squad_context, monkeypatch) -> None:
    squad, _, _ = squad_context
    calls = {"count": 0}

    def fake_run_tests(solution_code, test_code, *, timeout=5, framework="pytest"):
        calls["count"] += 1
        passed = calls["count"] >= 2
        return SandboxResult(
            blocked=False,
            timed_out=False,
            tests_passed=passed,
            exit_code=0 if passed else 1,
            stdout="1 passed" if passed else "1 failed — synthetic first-run feedback",
            stderr="" if passed else "AssertionError: synthetic failure",
            duration_seconds=0.01,
            timeout_seconds=timeout,
            command="python -m pytest ...",
            framework=framework,
            files={"solution.py": solution_code, "test_solution.py": test_code},
        )

    monkeypatch.setattr("agents.run_tests", fake_run_tests)
    result = squad.review("Run a full review and fix any failing tests.", OFF_BY_ONE_CODE)
    assert result.errors == []
    assert calls["count"] == 2
    assert len(result.sandbox_attempts) == 2
    assert result.sandbox_attempts[0].tests_passed is False
    assert result.sandbox_attempts[1].tests_passed is True
    assert result.trace["sandbox_retry"]["attempted"] is True
    reflection_prompts = [
        prompt for prompt in result.trace["prompts"] if prompt["key"] == "fixer_reflection"
    ]
    assert len(reflection_prompts) == 2
    retry_prompt_text = reflection_prompts[-1]["messages"][-1]["content"]
    assert "synthetic first-run feedback" in retry_prompt_text
